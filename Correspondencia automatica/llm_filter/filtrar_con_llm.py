#!/usr/bin/env python3
"""
Filtro LLM para clasificacion CNP -> ICCS.
Utiliza gpt-5-mini para elegir el mejor match entre los top-10 candidatos.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuracion
# IMPORTANTE: La API key se lee desde el archivo .env en la raiz del proyecto
# Si no existe, se intenta leer desde variable de entorno OPENAI_API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY or API_KEY == "TU_API_KEY_AQUI":
    raise ValueError(
        "ERROR: API key no configurada. Por favor:\n"
        "1. Crea un archivo .env en la raiz del proyecto\n"
        "2. Agrega la linea: OPENAI_API_KEY=tu-api-key-aqui\n"
        "O configura la variable de entorno OPENAI_API_KEY"
    )
MODEL_NAME = "gpt-5-mini"
TOP_K = 10  # numero fijo de candidatos a evaluar
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

# Rutas
REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_DETALLADO_PATH = REPO_ROOT / "Correspondencia automatica" / "embeddings" / "artifacts" / "matches_detallado.csv"
ICCS_DESCRIPCION_PATH = REPO_ROOT / "Correspondencia automatica" / "outputs" / "iccs_descripcion.csv"
CORRESP_MANUAL_PATH = REPO_ROOT / "Correspondencia manual" / "2024" / "07102025_TC_Final_2023-2024_v1.2.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga matches detallado e ICCS descripcion."""
    print("Cargando datos...")

    if not MATCHES_DETALLADO_PATH.exists():
        raise FileNotFoundError(f"No se encuentra: {MATCHES_DETALLADO_PATH}")
    if not ICCS_DESCRIPCION_PATH.exists():
        raise FileNotFoundError(f"No se encuentra: {ICCS_DESCRIPCION_PATH}")

    matches = pd.read_csv(MATCHES_DETALLADO_PATH, encoding="utf-8-sig")
    iccs_full = pd.read_csv(ICCS_DESCRIPCION_PATH, encoding="utf-8-sig")

    print(f"  - Matches cargados: {len(matches)} filas")
    print(f"  - ICCS descripcion: {len(iccs_full)} codigos")

    return matches, iccs_full


def normalizar_codigo_iccs(codigo: Any) -> str:
    """Normaliza codigos ICCS eliminando ceros a la izquierda y espacios."""
    if codigo is None:
        return ""
    codigo_str = str(codigo).strip()
    if not codigo_str:
        return ""
    codigo_norm = codigo_str.lstrip("0")
    return codigo_norm if codigo_norm else "0"


def build_iccs_glosa_map(iccs_full_df: pd.DataFrame) -> dict[str, str]:
    """Construye un dict codigo->glosa desde iccs_descripcion."""
    codigo_series = iccs_full_df["codigo_iccs"].astype(str).str.strip().str.lstrip("0")
    if "glosa_iccs" in iccs_full_df.columns:
        glosa_series = iccs_full_df["glosa_iccs"].astype(str)
    elif "iccs_glosa" in iccs_full_df.columns:
        glosa_series = iccs_full_df["iccs_glosa"].astype(str)
    else:
        glosa_series = pd.Series([""] * len(iccs_full_df))
    return dict(zip(codigo_series, glosa_series))


def preparar_candidatos(matches_df: pd.DataFrame, iccs_full_df: pd.DataFrame, top_k: int = TOP_K) -> dict[str, list[dict]]:
    """
    Agrupa matches por codigo CNP y hace JOIN con ICCS para obtener exclusiones/notas.

    Returns:
        Dict con cnp_codigo como key y lista de candidatos como value.
    """
    print(f"\nPreparando candidatos (top-{top_k} por codigo CNP)...")

    matches_df = matches_df.copy()
    iccs_full_df = iccs_full_df.copy()
    matches_df["iccs_codigo"] = matches_df["iccs_codigo"].astype(str)
    iccs_full_df["codigo_iccs"] = iccs_full_df["codigo_iccs"].astype(str)

    candidatos_por_cnp = {}
    codigos_cnp_unicos = matches_df["cnp_codigo"].unique()

    for cnp_codigo in codigos_cnp_unicos:
        cnp_matches = matches_df[matches_df["cnp_codigo"] == cnp_codigo].nsmallest(top_k, "rank")
        if len(cnp_matches) < top_k:
            print(f"  Aviso: CNP {cnp_codigo} tiene solo {len(cnp_matches)} candidatos (se esperaban {top_k})")

        cnp_info = cnp_matches.iloc[0]

        candidatos = []
        for _, match_row in cnp_matches.iterrows():
            iccs_codigo = str(match_row["iccs_codigo"])
            iccs_info = iccs_full_df[iccs_full_df["codigo_iccs"] == iccs_codigo]

            if len(iccs_info) == 0:
                print(f"  Error: Codigo ICCS {iccs_codigo} no encontrado en iccs_descripcion.csv")
                exclusiones = ""
                notas = ""
            else:
                iccs_info = iccs_info.iloc[0]
                exclusiones = str(iccs_info.get("exclusiones", "")).strip()
                notas = str(iccs_info.get("notas", "")).strip()

            candidato = {
                "rank": int(match_row["rank"]),
                "iccs_codigo": iccs_codigo,
                "iccs_glosa": str(match_row["iccs_glosa"]),
                "iccs_descripcion": str(match_row["iccs_descripcion"]),
                "iccs_inclusiones": str(match_row["iccs_inclusiones"]),
                "iccs_exclusiones": exclusiones,
                "iccs_notas": notas,
                "similarity_score": float(match_row["similarity_score"]),
            }
            candidatos.append(candidato)

        candidatos_por_cnp[str(cnp_codigo)] = {
            "cnp_codigo": str(cnp_info["cnp_codigo"]),
            "cnp_glosa": str(cnp_info["cnp_glosa"]),
            "cnp_descripcion": str(cnp_info["cnp_descripcion"]),
            "cnp_familia": str(cnp_info["cnp_familia"]),
            # El articulado no se envia al prompt para ahorrar tokens
            "cnp_articulado": str(cnp_info.get("cnp_articulado", "")),
            "candidatos": candidatos,
        }

    print(f"  OK {len(candidatos_por_cnp)} codigos CNP preparados")
    return candidatos_por_cnp


def construir_prompt(cnp_data: dict) -> str:
    """Construye el prompt para el LLM."""

    candidatos_texto = ""
    for cand in cnp_data["candidatos"]:
        candidatos_texto += f"""
{cand['rank']}. Codigo ICCS: {cand['iccs_codigo']}
   Glosa: {cand['iccs_glosa']}
   Descripcion: {cand['iccs_descripcion']}
   Inclusiones: {cand['iccs_inclusiones']}
   EXCLUSIONES: {cand['iccs_exclusiones']}
   NOTAS: {cand['iccs_notas']}
   Score similitud embeddings: {cand['similarity_score']:.4f}
"""

    prompt = f"""Eres un experto en clasificacion de delitos penales. Tu tarea es mapear un delito del Codigo Penal Nacional (CNP) chileno a la Clasificacion Internacional de Delitos con Fines Estadisticos (ICCS) de la ONU.

DELITO NACIONAL (CNP):
- Codigo: {cnp_data['cnp_codigo']}
- Glosa: {cnp_data['cnp_glosa']}
- Descripcion: {cnp_data['cnp_descripcion']}
- Familia: {cnp_data['cnp_familia']}

CANDIDATOS ICCS (Top {len(cnp_data['candidatos'])} por similitud semantica):
{candidatos_texto}

INSTRUCCIONES CRITICAS:
1. Elige el codigo ICCS que MEJOR se aproxime a la definicion del delito CNP.
2. NO busques el codigo mas especifico; busca el MAS PRECISO (puede ser general si es mas exacto).
3. Considera ESPECIALMENTE las EXCLUSIONES y NOTAS de cada candidato.
4. Si una exclusion descarta el delito CNP, ese candidato NO es valido.
5. Las NOTAS dan contexto sobre cuando aplicar cada codigo.
6. DELITOS MAS GRAVOSOS: Si dos codigos se excluyen mutuamente (ej: robo vs hurto), elige el delito MAS GRAVOSO. Por ejemplo: robo es mas gravoso que hurto, homicidio es mas gravoso que lesiones, violacion es mas gravosa que abuso sexual.
7. DELITOS SIN DESCRIPCION: Si el delito CNP no tiene descripcion o dice "sin descripcion":
   a) Intenta clasificar usando la GLOSA y FAMILIA aunque sea en terminos genericos.
   b) Solo devuelve "NINGUNO" si el delito es completamente generico (ej: "otros delitos") y no hay contexto suficiente para clasificar.
8. Si NINGUN candidato es apropiado tras aplicar todas las reglas anteriores, devuelve "NINGUNO" y explica por que.
9. No inventes informacion que no este en los insumos. Tu analisis legal es prioritario sobre el score de similitud.
10. Si alguna exclusion o inclusion menciona un codigo ICCS que describe mejor el delito CNP, elige ese codigo aunque no aparezca en la lista. Elimina ceros iniciales antes de reportarlo (ej: 0509 -> 509).

RESPONDE UNICAMENTE CON UN OBJETO JSON (sin markdown ni explicaciones adicionales):
{{
  "iccs_elegido": "codigo ICCS elegido o NINGUNO",
  "confianza": "alta|media|baja",
  "justificacion": "Explicacion breve de por que elegiste este codigo, menciona si aplicaste alguna exclusion",
  "exclusiones_aplicadas": ["lista de exclusiones que descartaron otros candidatos, vacio si no aplica"]
}}"""

    return prompt


def llamar_llm(client: OpenAI, prompt: str, cnp_codigo: str) -> dict[str, Any] | None:
    """Llama al LLM con reintentos y parsea la respuesta JSON."""

    for intento in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un experto en clasificacion de delitos penales. "
                            "Respondes solo en JSON valido. No inventes datos fuera de los insumos. "
                            "Cuando hay exclusion mutua, prefiere el delito mas gravoso. "
                            "Para delitos sin descripcion, clasifica con glosa y familia. "
                            "Solo responde NINGUNO si el delito es completamente generico sin contexto suficiente."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            contenido = response.choices[0].message.content
            resultado = json.loads(contenido)

            campos_requeridos = ["iccs_elegido", "confianza", "justificacion"]
            if not all(campo in resultado for campo in campos_requeridos):
                raise ValueError(f"Respuesta JSON incompleta: {resultado}")

            return resultado

        except json.JSONDecodeError as e:
            print(f"  Error: CNP {cnp_codigo}: JSON invalido (intento {intento}/{MAX_RETRIES})")
            if intento == MAX_RETRIES:
                print(f"      Error: {e}")
                return None

        except Exception as e:
            print(f"  Error: CNP {cnp_codigo}: fallo en API (intento {intento}/{MAX_RETRIES})")
            print(f"      Error: {e}")
            if intento < MAX_RETRIES:
                time.sleep(RETRY_DELAY * intento)
            else:
                return None

    return None


def procesar_batch(
    candidatos_por_cnp: dict[str, dict],
    client: OpenAI,
    iccs_glosa_map: dict[str, str],
    limite: int | None = None,
    checkpoint_file: Path | None = None,
) -> list[dict]:
    """Procesa lote de codigos CNP con LLM."""

    codigos_a_procesar = list(candidatos_por_cnp.keys())
    if limite:
        codigos_a_procesar = codigos_a_procesar[:limite]

    print(f"\nProcesando {len(codigos_a_procesar)} codigos CNP con {MODEL_NAME}...")

    resultados = []
    procesados = set()
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            resultados = checkpoint_data.get("resultados", [])
            procesados = set(checkpoint_data.get("procesados", []))
        print(f"  OK Checkpoint cargado: {len(procesados)} codigos ya procesados")

    errores = []
    for cnp_codigo in tqdm(codigos_a_procesar, desc="Procesando"):
        if cnp_codigo in procesados:
            continue

        cnp_data = candidatos_por_cnp[cnp_codigo]
        prompt = construir_prompt(cnp_data)

        respuesta_llm = llamar_llm(client, prompt, cnp_codigo)

        if respuesta_llm is None:
            errores.append(
                {"cnp_codigo": cnp_codigo, "error": "No se pudo obtener respuesta valida del LLM"}
            )
            continue

        iccs_elegido = respuesta_llm["iccs_elegido"]
        iccs_glosa_elegida = ""
        if iccs_elegido != "NINGUNO":
            for cand in cnp_data["candidatos"]:
                if cand["iccs_codigo"] == iccs_elegido:
                    iccs_glosa_elegida = cand["iccs_glosa"]
                    break
            if not iccs_glosa_elegida:
                iccs_glosa_elegida = iccs_glosa_map.get(normalizar_codigo_iccs(iccs_elegido), "")

        top_refs: dict[str, Any] = {}
        for idx in range(TOP_K):
            codigo = cnp_data["candidatos"][idx]["iccs_codigo"] if idx < len(cnp_data["candidatos"]) else ""
            score = cnp_data["candidatos"][idx]["similarity_score"] if idx < len(cnp_data["candidatos"]) else ""
            top_refs[f"top{idx + 1}_codigo"] = codigo
            top_refs[f"top{idx + 1}_score"] = score

        resultado = {
            "cnp_codigo": cnp_data["cnp_codigo"],
            "cnp_glosa": cnp_data["cnp_glosa"],
            "cnp_descripcion": cnp_data["cnp_descripcion"],
            "cnp_familia": cnp_data["cnp_familia"],
            "iccs_elegido": iccs_elegido,
            "iccs_glosa_elegida": iccs_glosa_elegida,
            "confianza": respuesta_llm["confianza"],
            "justificacion": respuesta_llm["justificacion"],
            "exclusiones_aplicadas": respuesta_llm.get("exclusiones_aplicadas", []),
            **top_refs,
        }

        resultados.append(resultado)
        procesados.add(cnp_codigo)

        if checkpoint_file and len(procesados) % 10 == 0:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"procesados": list(procesados), "resultados": resultados}, f, indent=2, ensure_ascii=False)

    if errores:
        print(f"\nError: {len(errores)} codigos con errores:")
        for error in errores[:5]:
            print(f"  - CNP {error['cnp_codigo']}: {error['error']}")

        error_file = OUTPUT_DIR / "errores.log"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errores, f, indent=2, ensure_ascii=False)
        print(f"  Log completo en: {error_file}")

    return resultados


def guardar_resultados(resultados: list[dict], output_dir: Path) -> pd.DataFrame:
    """Guarda resultados en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(resultados)

    completo_path = output_dir / "clasificacion_con_justificacion.csv"
    df.to_csv(completo_path, index=False, encoding="utf-8-sig")
    print(f"\nOK Clasificacion completa guardada: {completo_path}")

    columnas_compacto = [
        "cnp_codigo",
        "cnp_glosa",
        "iccs_elegido",
        "iccs_glosa_elegida",
        "confianza",
        "top1_codigo",
        "top1_score",
        "top2_codigo",
        "top2_score",
    ]
    df_compacto = df[columnas_compacto]
    compacto_path = output_dir / "clasificacion_final.csv"
    df_compacto.to_csv(compacto_path, index=False, encoding="utf-8-sig")
    print(f"OK Clasificacion compacta guardada: {compacto_path}")

    print(f"\n{'=' * 60}")
    print("ESTADISTICAS:")
    print(f"  Total procesados: {len(df)}")
    print(f"  NINGUNO asignado: {(df['iccs_elegido'] == 'NINGUNO').sum()}")
    print(f"  Confianza alta: {(df['confianza'] == 'alta').sum()}")
    print(f"  Confianza media: {(df['confianza'] == 'media').sum()}")
    print(f"  Confianza baja: {(df['confianza'] == 'baja').sum()}")

    coincide_top1 = (df["iccs_elegido"] == df["top1_codigo"]).sum()
    print(f"  Coincide con top-1 embedding: {coincide_top1} ({100 * coincide_top1 / len(df):.1f}%)")
    print(f"{'=' * 60}")
    return df


MANUAL_COLS = ["N4-2024 UNODC", "N3-2024 UNODC", "N2-2024 UNODC", "N1-2024 FINAL"]


def _extraer_manual_codigos(row: pd.Series) -> list[str]:
    """Devuelve los codigos manuales desde el nivel mas granular al menos granular."""
    codigos = []
    for col in MANUAL_COLS:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            cod_norm = normalizar_codigo_iccs(val)
            codigos.append(cod_norm)
    return codigos


def evaluar_contra_manual(
    df_resultados: pd.DataFrame,
    iccs_full_df: pd.DataFrame,
    iccs_glosa_map: dict[str, str],
    output_dir: Path,
) -> None:
    """Compara resultados LLM vs correspondencia manual y genera estadisticas + xlsx."""
    if not CORRESP_MANUAL_PATH.exists():
        print(f"Aviso: No se encontro correspondencia manual en {CORRESP_MANUAL_PATH}")
        return

    manual_df = pd.read_excel(
        CORRESP_MANUAL_PATH,
        sheet_name="TC_2024",
        skiprows=1,
    )
    manual_df["cnp_codigo"] = manual_df["CUM"].astype(str).str.strip()
    manual_df["manual_codigos"] = manual_df.apply(_extraer_manual_codigos, axis=1)
    manual_df["manual_codigo_granular"] = manual_df["manual_codigos"].apply(lambda xs: xs[0] if xs else "")
    manual_df["glosa_manual"] = manual_df["GLOSA 2024"].astype(str)
    manual_filtrado = manual_df[manual_df["manual_codigo_granular"] != ""].copy()

    df_resultados = df_resultados.copy()
    df_resultados["iccs_elegido"] = df_resultados["iccs_elegido"].fillna("")
    df_resultados["iccs_elegido_norm"] = df_resultados["iccs_elegido"].apply(normalizar_codigo_iccs)
    df_resultados["iccs_glosa_elegida"] = df_resultados.apply(
        lambda r: r["iccs_glosa_elegida"] or iccs_glosa_map.get(r["iccs_elegido_norm"], ""),
        axis=1,
    )

    comparacion = manual_filtrado.merge(df_resultados, on="cnp_codigo", how="left")
    comparacion["llm_codigo"] = comparacion["iccs_elegido_norm"].fillna("")
    comparacion["coincide"] = comparacion.apply(
        lambda r: bool(r["llm_codigo"]) and r["llm_codigo"] in set(r.get("manual_codigos", [])),
        axis=1,
    )

    total_manual = len(manual_filtrado)
    con_llm = (comparacion["llm_codigo"] != "").sum()
    coincidencias = comparacion["coincide"].sum()
    discrepancias = comparacion[
        (comparacion["manual_codigo_granular"] != "") & (comparacion["llm_codigo"] != "") & (~comparacion["coincide"])
    ]
    sin_clas_llm = comparacion[(comparacion["manual_codigo_granular"] != "") & (comparacion["llm_codigo"] == "")]

    print(f"\n{'=' * 60}")
    print("EVALUACION VS CORRESPONDENCIA MANUAL:")
    print(f"  Total con etiqueta manual: {total_manual}")
    print(f"  LLM con codigo asignado: {con_llm}")
    print(f"  Coincidencias: {coincidencias}")
    print(f"  Discrepancias: {len(discrepancias)}")
    print(f"  Manual con NINGUNA respuesta LLM: {len(sin_clas_llm)}")
    if len(discrepancias) > 0:
        print("  Muestras de discrepancias (codigo_manual -> codigo_llm):")
        for _, row in discrepancias.head(5).iterrows():
            print(
                f"    CNP {row['cnp_codigo']}: {row['manual_codigo_granular']} ({row['glosa_manual']}) "
                f"vs {row['llm_codigo']} ({row['iccs_glosa_elegida']})"
            )
    print(f"{'=' * 60}")

    detalle_cols = [
        "cnp_codigo",
        "glosa_manual",
        "manual_codigo_granular",
        "manual_codigos",
        "llm_codigo",
        "iccs_glosa_elegida",
        "confianza",
        "justificacion",
        "exclusiones_aplicadas",
        "cnp_glosa",
        "cnp_descripcion",
    ]
    detalle = comparacion.copy()
    detalle = detalle[[c for c in detalle_cols if c in detalle.columns]]
    salida_xlsx = output_dir / "comparacion_llm_vs_manual.xlsx"
    detalle.to_excel(salida_xlsx, index=False)
    print(f"OK Comparacion manual guardada: {salida_xlsx}")


def estimar_costo(n_codigos: int) -> dict:
    """Estima el costo de procesamiento (ajusta precios a la tarifa vigente)."""
    tokens_por_request_input = 2000
    tokens_por_request_output = 200

    total_input = n_codigos * tokens_por_request_input
    total_output = n_codigos * tokens_por_request_output

    precio_input = 0.150  # USD por 1M tokens (placeholder; ajustar)
    precio_output = 0.600  # USD por 1M tokens (placeholder; ajustar)

    costo_input = (total_input / 1_000_000) * precio_input
    costo_output = (total_output / 1_000_000) * precio_output
    costo_total = costo_input + costo_output

    return {
        "n_codigos": n_codigos,
        "total_tokens_estimado": total_input + total_output,
        "costo_usd": round(costo_total, 3),
        "tiempo_estimado_min": round(n_codigos * 2 / 60, 1),  # ~2 seg por codigo
    }


def main():
    parser = argparse.ArgumentParser(description="Filtro LLM para clasificacion CNP -> ICCS (top-10, gpt-5-mini)")
    parser.add_argument("--test", action="store_true", help="Modo test: solo 10 codigos")
    parser.add_argument("--limite", type=int, help="Limite de codigos a procesar")
    args = parser.parse_args()

    matches_df, iccs_full_df = cargar_datos()

    candidatos_por_cnp = preparar_candidatos(matches_df, iccs_full_df, top_k=TOP_K)
    iccs_glosa_map = build_iccs_glosa_map(iccs_full_df)

    if args.test:
        limite = 10
    elif args.limite:
        limite = args.limite
    else:
        limite = None

    n_a_procesar = limite if limite else len(candidatos_por_cnp)

    estimacion = estimar_costo(n_a_procesar)
    print(f"\n{'=' * 60}")
    print("ESTIMACION DE PROCESAMIENTO:")
    print(f"  Codigos CNP a procesar: {estimacion['n_codigos']}")
    print(f"  Tokens estimados: ~{estimacion['total_tokens_estimado']:,}")
    print(f"  Costo estimado: ${estimacion['costo_usd']} USD")
    print(f"  Tiempo estimado: ~{estimacion['tiempo_estimado_min']} minutos")
    print(f"  Modelo: {MODEL_NAME}")
    print(f"{'=' * 60}\n")

    if args.test:
        print("Aviso: MODO TEST (solo 10 codigos)\n")

    respuesta = input("Proceder con el procesamiento? (s/n): ")
    if respuesta.lower() != "s":
        print("Cancelado.")
        sys.exit(0)

    client = OpenAI(api_key=API_KEY)

    checkpoint_file = OUTPUT_DIR / "checkpoint.json"
    resultados = procesar_batch(
        candidatos_por_cnp,
        client,
        iccs_glosa_map,
        limite=limite,
        checkpoint_file=checkpoint_file,
    )

    df_resultados = guardar_resultados(resultados, OUTPUT_DIR)
    evaluar_contra_manual(df_resultados, iccs_full_df, iccs_glosa_map, OUTPUT_DIR)

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("\nOK Checkpoint eliminado (procesamiento completo)")


if __name__ == "__main__":
    main()
