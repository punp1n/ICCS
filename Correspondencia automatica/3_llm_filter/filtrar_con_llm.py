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
MODEL_NAME = "gpt-5-mini"
TOP_K = 10  # numero fijo de candidatos a evaluar
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

# Rutas
REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_DETALLADO_PATH = REPO_ROOT / "Correspondencia automatica" / "2_embeddings" / "outputs" / "matches_detallado.csv"
ICCS_DESCRIPCION_PATH = REPO_ROOT / "Correspondencia automatica" / "1_iccs" / "outputs" / "iccs_descripcion.csv"
CORRESP_MANUAL_PATH = REPO_ROOT / "Correspondencia manual" / "2024" / "28072025_TC_Final_2023-2024_version completa.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def get_api_key() -> str:
    """Obtiene la API key desde .env o variable de entorno."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "TU_API_KEY_AQUI":
        raise ValueError(
            "ERROR: API key no configurada. Por favor:\n"
            "1. Crea un archivo .env en la raiz del proyecto\n"
            "2. Agrega la linea: OPENAI_API_KEY=tu-api-key-aqui\n"
            "O configura la variable de entorno OPENAI_API_KEY"
        )
    return api_key


def get_openai_client() -> OpenAI:
    """Crea el cliente OpenAI solo cuando se necesita."""
    return OpenAI(api_key=get_api_key())


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


def cargar_iccs_descripcion() -> pd.DataFrame:
    """Carga solo ICCS descripcion (para modo comparar)."""
    print("Cargando ICCS descripcion...")
    if not ICCS_DESCRIPCION_PATH.exists():
        raise FileNotFoundError(f"No se encuentra: {ICCS_DESCRIPCION_PATH}")
    iccs_full = pd.read_csv(ICCS_DESCRIPCION_PATH, encoding="utf-8-sig")
    print(f"  - ICCS descripcion: {len(iccs_full)} codigos")
    return iccs_full


def cargar_salida_llm(path: Path) -> pd.DataFrame:
    """Carga la salida LLM desde CSV/XLSX y reporta el archivo usado."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra salida LLM: {path}")

    print(f"Usando salida LLM: {path}")
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  - Salida LLM cargada: {len(df)} filas")
    return df


def seleccionar_modo() -> str:
    """Panel de opciones interactivo."""
    print("\nPANEL DE OPCIONES")
    print("1. Ejecutar LLM + comparar (genera salidas)")
    print("2. Solo comparar (usa salida LLM existente)")
    print("3. Salir")
    while True:
        opcion = input("Selecciona una opcion [1-3]: ").strip()
        if opcion == "1":
            return "llm"
        if opcion == "2":
            return "comparar"
        if opcion == "3":
            return "salir"
        print("Opcion invalida. Intenta nuevamente.")


def normalizar_codigo_iccs(codigo: Any) -> str:
    """Normaliza codigos ICCS: elimina espacios y convierte a string."""
    if codigo is None:
        return ""
    codigo_str = str(codigo).strip()
    return codigo_str


def codigo_a_int(valor: Any) -> int | None:
    """Convierte un codigo a int; devuelve None si no es valido."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    if isinstance(valor, (int,)) and not isinstance(valor, bool):
        return int(valor)
    valor_str = str(valor).strip()
    if not valor_str or valor_str.upper() == "NINGUNO":
        return None
    try:
        return int(float(valor_str))
    except ValueError:
        return None


def serie_a_int(serie: pd.Series) -> pd.Series:
    """Convierte una serie a Int64 (nullable)."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64")


def extraer_codigos_iccs_de_texto(texto: str) -> set[str]:
    """Extrae codigos ICCS mencionados en un texto (exclusiones/inclusiones/notas)."""
    import re
    if not texto or pd.isna(texto):
        return set()
    # Busca patrones como: "codigo 0501", "(0501)", "clasificar como 0501", etc.
    # Los codigos ICCS son numericos, pueden tener 3-5 digitos
    patron = r'\b\d{3,5}\b'
    codigos_encontrados = re.findall(patron, str(texto))
    return set(codigos_encontrados)


def build_iccs_glosa_map(iccs_full_df: pd.DataFrame) -> dict[Any, str]:
    """Construye un dict codigo->glosa desde iccs_descripcion."""
    codigo_series = iccs_full_df["codigo_iccs"].astype(str).str.strip()
    if "glosa_iccs" in iccs_full_df.columns:
        glosa_series = iccs_full_df["glosa_iccs"].astype(str)
    elif "iccs_glosa" in iccs_full_df.columns:
        glosa_series = iccs_full_df["iccs_glosa"].astype(str)
    else:
        glosa_series = pd.Series([""] * len(iccs_full_df))
    mapa: dict[Any, str] = {}
    for codigo, glosa in zip(codigo_series, glosa_series):
        codigo_str = str(codigo).strip()
        mapa[codigo_str] = glosa
        codigo_int = codigo_a_int(codigo_str)
        if codigo_int is not None:
            mapa[codigo_int] = glosa
    return mapa


def completar_glosas_topk(
    df: pd.DataFrame,
    iccs_glosa_map: dict[str, str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Completa columnas topX_glosa a partir de topX_codigo si faltan o estan vacias."""
    df = df.copy()
    for idx in range(1, top_k + 1):
        codigo_col = f"top{idx}_codigo"
        glosa_col = f"top{idx}_glosa"
        if codigo_col not in df.columns:
            continue
        if glosa_col not in df.columns:
            df[glosa_col] = ""
        glosas_actuales = df[glosa_col].fillna("").astype(str)
        faltan = glosas_actuales.str.strip() == ""
        codigos = df[codigo_col]
        df.loc[faltan, glosa_col] = codigos.map(iccs_glosa_map).fillna("")
    return df


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

6. CRITERIO DE CLASIFICACION - MOVIL DEL DELITO:
   El criterio principal es el MOVIL O JUSTIFICACION ORIGINAL del delito, independiente de sus consecuencias.

   EJEMPLOS DE CLASIFICACION CORRECTA:
   - "Robo con homicidio" -> Clasifica como ROBO (delito contra propiedad). El movil es apropiarse de bienes; la muerte es fortuita.
   - "Robo con violacion" -> Clasifica como ROBO si el movil inicial era robar y la violacion fue oportunista. Si el movil inicial era sexual y el robo secundario, clasifica como violacion.
   - "Secuestro extorsivo" -> Clasifica como SECUESTRO. El movil es privar de libertad; la extorsion es el objetivo del secuestro.
   - "Violacion con homicidio para eliminar testigo" -> Clasifica como VIOLACION. El movil original era sexual; matar al testigo es posterior.
   - "Homicidio para robar" -> Clasifica como HOMICIDIO si matar era necesario para robar (no fortuito).

   REGLA: Preguntate "¿Cual era la intencion INICIAL del delincuente antes de actuar?" Esa es la clasificacion correcta.

7. DELITOS SIN DESCRIPCION: Si el delito CNP no tiene descripcion o dice "sin descripcion":
   a) Intenta clasificar usando la GLOSA y FAMILIA aunque sea en terminos genericos.
   b) Solo devuelve "NINGUNO" si el delito es completamente generico (ej: "otros delitos") y no hay contexto suficiente para clasificar.

8. Si NINGUN candidato es apropiado tras aplicar todas las reglas anteriores, devuelve "NINGUNO" y explica por que.

9. RESTRICCION DE CODIGOS - PROHIBIDO INVENTAR:
   - DEBES elegir uno de los {len(cnp_data['candidatos'])} codigos ICCS listados arriba como candidatos.
   - EXCEPCION: Si en las EXCLUSIONES o INCLUSIONES de algun candidato se menciona explicitamente otro codigo ICCS que describe mejor el delito CNP, puedes elegir ese codigo aunque NO este en la lista de candidatos.
   - Si eliges un codigo que no esta en los candidatos, DEBES justificar en que exclusion/inclusion aparece mencionado.
   - NO puedes inventar codigos arbitrarios. Si no hay candidato apropiado ni codigo mencionado en exclusiones/inclusiones, devuelve "NINGUNO".

10. Tu analisis legal y criminologico es prioritario sobre el score de similitud de embeddings.

CODIGOS ICCS VALIDOS PARA ESTE DELITO CNP (elige UNO de estos o "NINGUNO"):
{', '.join([cand['iccs_codigo'] for cand in cnp_data['candidatos']])}

RESPONDE UNICAMENTE CON UN OBJETO JSON (sin markdown ni explicaciones adicionales):
{{
  "iccs_elegido": "codigo ICCS elegido (debe ser uno de la lista de arriba) o NINGUNO",
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
                            "Clasifica segun el MOVIL del delito, no la consecuencia mas grave. "
                            "Elige de los 10 candidatos presentados o de codigos mencionados en exclusiones/inclusiones. "
                            "NUNCA inventes codigos arbitrarios. "
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

        # VALIDACION: Verificar que el codigo elegido este en los candidatos o sea NINGUNO
        if iccs_elegido != "NINGUNO":
            iccs_elegido_norm = normalizar_codigo_iccs(iccs_elegido)
            codigos_validos = [normalizar_codigo_iccs(cand["iccs_codigo"]) for cand in cnp_data["candidatos"]]

            # Extraer codigos mencionados en exclusiones/inclusiones/notas de todos los candidatos
            codigos_mencionados = set()
            for cand in cnp_data["candidatos"]:
                codigos_mencionados.update(extraer_codigos_iccs_de_texto(cand.get("iccs_exclusiones", "")))
                codigos_mencionados.update(extraer_codigos_iccs_de_texto(cand.get("iccs_inclusiones", "")))
                codigos_mencionados.update(extraer_codigos_iccs_de_texto(cand.get("iccs_notas", "")))

            # Verificar si el codigo esta en candidatos o en codigos mencionados
            if iccs_elegido_norm not in codigos_validos and iccs_elegido_norm not in codigos_mencionados:
                print(f"  ALERTA: CNP {cnp_codigo} - LLM alucino codigo '{iccs_elegido}' que NO esta en candidatos ni mencionado")
                print(f"         Candidatos validos: {codigos_validos}")
                print(f"         Codigos mencionados en exclusiones/inclusiones: {codigos_mencionados}")
                errores.append({
                    "cnp_codigo": cnp_codigo,
                    "error": f"Codigo alucinado: {iccs_elegido}",
                    "candidatos_validos": codigos_validos,
                    "codigos_mencionados": list(codigos_mencionados),
                    "justificacion_llm": respuesta_llm.get("justificacion", "")
                })
                continue

            # Buscar la glosa del codigo elegido
            for cand in cnp_data["candidatos"]:
                if normalizar_codigo_iccs(cand["iccs_codigo"]) == iccs_elegido_norm:
                    iccs_glosa_elegida = cand["iccs_glosa"]
                    break

            # Si no se encontro en candidatos, buscar en el mapa general de ICCS
            if not iccs_glosa_elegida:
                iccs_glosa_elegida = iccs_glosa_map.get(iccs_elegido_norm, "")

        top_refs: dict[str, Any] = {}
        for idx in range(TOP_K):
            codigo = cnp_data["candidatos"][idx]["iccs_codigo"] if idx < len(cnp_data["candidatos"]) else ""
            glosa = cnp_data["candidatos"][idx]["iccs_glosa"] if idx < len(cnp_data["candidatos"]) else ""
            score = cnp_data["candidatos"][idx]["similarity_score"] if idx < len(cnp_data["candidatos"]) else ""
            top_refs[f"top{idx + 1}_codigo"] = codigo
            top_refs[f"top{idx + 1}_glosa"] = glosa
            top_refs[f"top{idx + 1}_score"] = score

        resultado = {
            "cnp_codigo": cnp_data["cnp_codigo"],
            "cnp_glosa": cnp_data["cnp_glosa"],
            "cnp_descripcion": cnp_data["cnp_descripcion"],
            "cnp_familia": cnp_data["cnp_familia"],
            "cnp_articulado": cnp_data.get("cnp_articulado", ""),
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
        print(f"\n{'=' * 60}")
        print(f"ERRORES Y ALUCINACIONES: {len(errores)} casos detectados")
        print(f"{'=' * 60}")

        alucinaciones = [e for e in errores if "alucinado" in e.get("error", "").lower()]
        otros_errores = [e for e in errores if "alucinado" not in e.get("error", "").lower()]

        if alucinaciones:
            print(f"\n  ALUCINACIONES DE CODIGOS: {len(alucinaciones)}")
            for error in alucinaciones[:5]:
                print(f"    - CNP {error['cnp_codigo']}: {error['error']}")
            if len(alucinaciones) > 5:
                print(f"    ... y {len(alucinaciones) - 5} mas")

        if otros_errores:
            print(f"\n  OTROS ERRORES: {len(otros_errores)}")
            for error in otros_errores[:5]:
                print(f"    - CNP {error['cnp_codigo']}: {error['error']}")
            if len(otros_errores) > 5:
                print(f"    ... y {len(otros_errores) - 5} mas")

        error_file = OUTPUT_DIR / "errores.log"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errores, f, indent=2, ensure_ascii=False)

        if alucinaciones:
            alucinaciones_file = OUTPUT_DIR / "alucinaciones_detectadas.json"
            with open(alucinaciones_file, "w", encoding="utf-8") as f:
                json.dump(alucinaciones, f, indent=2, ensure_ascii=False)
            print(f"\n  Log de alucinaciones: {alucinaciones_file}")

        print(f"  Log completo de errores: {error_file}")
        print(f"{'=' * 60}")

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


def _extraer_manual_codigos(row: pd.Series) -> list[int]:
    """Devuelve los codigos manuales desde el nivel mas granular al menos granular."""
    codigos = []
    for col in MANUAL_COLS:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            cod_int = codigo_a_int(val)
            if cod_int is not None:
                codigos.append(cod_int)
    return codigos


def _lista_a_texto(valor: Any) -> str:
    """Convierte listas a texto legible."""
    if valor is None:
        return ""
    if isinstance(valor, list):
        return "; ".join([str(v) for v in valor if str(v).strip()])
    if pd.isna(valor):
        return ""
    return str(valor)


def _glosas_para_codigos(codigos: list[int], iccs_glosa_map: dict[str, str]) -> list[str]:
    """Mapea codigos ICCS a glosas."""
    return [iccs_glosa_map.get(cod, "") for cod in codigos]


def _codigo_str(codigo: Any) -> str:
    """Normaliza codigo como string numerico sin separadores."""
    cod_int = codigo_a_int(codigo)
    return str(cod_int) if cod_int is not None else ""


def _tipo_relacion_jerarquia(codigo_llm: Any, codigo_manual: Any) -> str:
    """
    Relacion jerarquica entre codigo LLM y manual.
    Retorna: exacto | llm_mas_desagregado | manual_mas_desagregado | sin_relacion
    """
    llm_str = _codigo_str(codigo_llm)
    manual_str = _codigo_str(codigo_manual)
    if not llm_str or not manual_str:
        return "sin_relacion"
    if llm_str == manual_str:
        return "exacto"
    if llm_str.startswith(manual_str):
        return "llm_mas_desagregado"
    if manual_str.startswith(llm_str):
        return "manual_mas_desagregado"
    return "sin_relacion"


def _buscar_jerarquia_en_lista(codigo_llm: Any, codigos_manual: list[int]) -> tuple[bool, int | None, str]:
    """Busca compatibilidad jerarquica (no exacta) entre codigo LLM y lista manual."""
    for codigo_manual in codigos_manual:
        relacion = _tipo_relacion_jerarquia(codigo_llm, codigo_manual)
        if relacion in {"llm_mas_desagregado", "manual_mas_desagregado"}:
            return True, codigo_manual, relacion
    return False, None, "sin_relacion"


def _buscar_match_topk(
    row: pd.Series,
    manual_set: set[int],
    top_k: int = TOP_K,
) -> tuple[int | None, int | None, str]:
    """Busca si algun top-k coincide con los codigos manuales."""
    for idx in range(1, top_k + 1):
        codigo_col = f"top{idx}_codigo"
        codigo = codigo_a_int(row.get(codigo_col, None))
        if codigo is not None and codigo in manual_set:
            glosa = row.get(f"top{idx}_glosa", "")
            return codigo, idx, glosa
    return None, None, ""


def _buscar_match_topk_jerarquia(
    row: pd.Series,
    manual_codigos: list[int],
    top_k: int = TOP_K,
) -> tuple[int | None, int | None, str, int | None, str]:
    """Busca primer top-k con compatibilidad jerarquica (no exacta) con lista manual."""
    for idx in range(1, top_k + 1):
        codigo_col = f"top{idx}_codigo"
        codigo_topk = codigo_a_int(row.get(codigo_col, None))
        if codigo_topk is None:
            continue
        coincide, codigo_manual_match, tipo_relacion = _buscar_jerarquia_en_lista(codigo_topk, manual_codigos)
        if coincide:
            glosa = row.get(f"top{idx}_glosa", "")
            return codigo_topk, idx, glosa, codigo_manual_match, tipo_relacion
    return None, None, "", None, "sin_relacion"


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
    manual_df["cnp_codigo"] = serie_a_int(manual_df["CUM"])
    manual_df["manual_codigos"] = manual_df.apply(_extraer_manual_codigos, axis=1)
    manual_df["manual_codigo_granular"] = manual_df["manual_codigos"].apply(
        lambda xs: xs[0] if xs else pd.NA
    ).astype("Int64")
    manual_df["glosa_manual"] = manual_df["GLOSA 2024"].astype(str)
    manual_df["manual_codigos_str"] = manual_df["manual_codigos"].apply(_lista_a_texto)
    manual_df["manual_glosa_iccs"] = manual_df["manual_codigo_granular"].map(iccs_glosa_map).fillna("")
    manual_df["manual_glosas_iccs"] = manual_df["manual_codigos"].apply(
        lambda xs: _glosas_para_codigos(xs, iccs_glosa_map)
    )
    manual_df["manual_glosas_iccs_str"] = manual_df["manual_glosas_iccs"].apply(_lista_a_texto)
    manual_filtrado = manual_df[manual_df["manual_codigo_granular"].notna()].copy()

    df_resultados = df_resultados.copy()
    if "iccs_elegido" not in df_resultados.columns or "cnp_codigo" not in df_resultados.columns:
        raise ValueError(
            "La salida LLM no contiene columnas requeridas: se esperan 'cnp_codigo' e 'iccs_elegido'."
        )

    if "iccs_glosa_elegida" not in df_resultados.columns:
        df_resultados["iccs_glosa_elegida"] = ""

    df_resultados["cnp_codigo"] = serie_a_int(df_resultados["cnp_codigo"])
    df_resultados["iccs_elegido"] = df_resultados["iccs_elegido"].fillna("")
    df_resultados["iccs_elegido"] = serie_a_int(df_resultados["iccs_elegido"])
    df_resultados["iccs_elegido_norm"] = df_resultados["iccs_elegido"]

    def _glosa_por_codigo(codigo: Any) -> str:
        if codigo is None or pd.isna(codigo):
            return ""
        return iccs_glosa_map.get(int(codigo), "")

    df_resultados["iccs_glosa_elegida"] = df_resultados.apply(
        lambda r: r["iccs_glosa_elegida"] or _glosa_por_codigo(r["iccs_elegido_norm"]),
        axis=1,
    )
    if "exclusiones_aplicadas" in df_resultados.columns:
        df_resultados["exclusiones_aplicadas_str"] = df_resultados["exclusiones_aplicadas"].apply(_lista_a_texto)

    # Convertir codigos top-k a int
    for idx in range(1, TOP_K + 1):
        col = f"top{idx}_codigo"
        if col in df_resultados.columns:
            df_resultados[col] = serie_a_int(df_resultados[col])

    df_resultados = completar_glosas_topk(df_resultados, iccs_glosa_map, top_k=TOP_K)

    comparacion = manual_filtrado.merge(df_resultados, on="cnp_codigo", how="left")
    comparacion["llm_codigo"] = comparacion["iccs_elegido_norm"]
    comparacion["coincide_granular"] = comparacion.apply(
        lambda r: pd.notna(r["llm_codigo"]) and r["llm_codigo"] == r["manual_codigo_granular"],
        axis=1,
    )
    comparacion["coincide_manual"] = comparacion.apply(
        lambda r: pd.notna(r["llm_codigo"]) and r["llm_codigo"] in set(r.get("manual_codigos", [])),
        axis=1,
    )
    comparacion["coincide"] = comparacion["coincide_manual"]

    comparacion["tipo_relacion_jerarquia_granular"] = comparacion.apply(
        lambda r: _tipo_relacion_jerarquia(r["llm_codigo"], r["manual_codigo_granular"]),
        axis=1,
    )
    comparacion["coincide_jerarquia_granular"] = comparacion["tipo_relacion_jerarquia_granular"].isin(
        {"llm_mas_desagregado", "manual_mas_desagregado"}
    )

    jerarquia_manual = comparacion.apply(
        lambda r: _buscar_jerarquia_en_lista(r["llm_codigo"], r.get("manual_codigos", [])),
        axis=1,
    )
    comparacion["coincide_jerarquia_manual"] = jerarquia_manual.apply(lambda x: x[0])
    comparacion["manual_match_jerarquia"] = jerarquia_manual.apply(lambda x: x[1])
    comparacion["tipo_relacion_jerarquia_manual"] = jerarquia_manual.apply(lambda x: x[2])
    comparacion["manual_match_jerarquia"] = serie_a_int(comparacion["manual_match_jerarquia"])
    comparacion["manual_match_jerarquia_glosa"] = comparacion["manual_match_jerarquia"].map(iccs_glosa_map).fillna("")

    topk_info = comparacion.apply(
        lambda r: _buscar_match_topk(r, set(r.get("manual_codigos", []))), axis=1
    )
    comparacion["topk_match_codigo"] = topk_info.apply(lambda x: x[0])
    comparacion["topk_match_rank"] = topk_info.apply(lambda x: x[1] if x[1] else pd.NA)
    comparacion["topk_match_glosa"] = topk_info.apply(lambda x: x[2])
    comparacion["topk_match_codigo"] = serie_a_int(comparacion["topk_match_codigo"])
    comparacion["topk_match_rank"] = serie_a_int(comparacion["topk_match_rank"])
    comparacion["topk_coincide_manual"] = comparacion["topk_match_codigo"].notna()

    topk_jerarquia = comparacion.apply(
        lambda r: _buscar_match_topk_jerarquia(r, r.get("manual_codigos", [])),
        axis=1,
    )
    comparacion["topk_match_jerarquia_codigo"] = topk_jerarquia.apply(lambda x: x[0])
    comparacion["topk_match_jerarquia_rank"] = topk_jerarquia.apply(lambda x: x[1] if x[1] else pd.NA)
    comparacion["topk_match_jerarquia_glosa"] = topk_jerarquia.apply(lambda x: x[2])
    comparacion["topk_match_jerarquia_manual_codigo"] = topk_jerarquia.apply(lambda x: x[3])
    comparacion["topk_match_jerarquia_tipo"] = topk_jerarquia.apply(lambda x: x[4])
    comparacion["topk_match_jerarquia_codigo"] = serie_a_int(comparacion["topk_match_jerarquia_codigo"])
    comparacion["topk_match_jerarquia_rank"] = serie_a_int(comparacion["topk_match_jerarquia_rank"])
    comparacion["topk_match_jerarquia_manual_codigo"] = serie_a_int(
        comparacion["topk_match_jerarquia_manual_codigo"]
    )
    comparacion["topk_match_jerarquia_manual_glosa"] = comparacion["topk_match_jerarquia_manual_codigo"].map(
        iccs_glosa_map
    ).fillna("")
    comparacion["topk_coincide_jerarquia_manual"] = comparacion["topk_match_jerarquia_codigo"].notna()

    def _clasificar(row: pd.Series) -> tuple[str, str]:
        if row["coincide_granular"]:
            return "ok", "llm=granular"
        if row["coincide_manual"]:
            return "parcial", "llm=en_manual_codigos"
        if row["coincide_jerarquia_granular"]:
            if row["tipo_relacion_jerarquia_granular"] == "llm_mas_desagregado":
                detalle = f"llm_mas_desagregado_que_manual_granular (manual={row['manual_codigo_granular']})"
            else:
                detalle = f"manual_mas_desagregado_que_llm (manual={row['manual_codigo_granular']})"
            return "parcial", detalle
        if row["coincide_jerarquia_manual"]:
            if row["tipo_relacion_jerarquia_manual"] == "llm_mas_desagregado":
                detalle = (
                    f"llm_mas_desagregado_que_manual_codigos "
                    f"(manual_match={row['manual_match_jerarquia']})"
                )
            else:
                detalle = (
                    f"manual_codigos_mas_desagregado_que_llm "
                    f"(manual_match={row['manual_match_jerarquia']})"
                )
            return "parcial", detalle
        if row["topk_coincide_manual"]:
            detalle = f"topk_en_manual_codigos (top{row['topk_match_rank']})"
            return "parcial", detalle
        if row["topk_coincide_jerarquia_manual"]:
            if row["topk_match_jerarquia_tipo"] == "llm_mas_desagregado":
                detalle = (
                    f"topk_jerarquia_llm_mas_desagregado (top{row['topk_match_jerarquia_rank']}, "
                    f"manual_match={row['topk_match_jerarquia_manual_codigo']})"
                )
            else:
                detalle = (
                    f"topk_jerarquia_manual_mas_desagregado (top{row['topk_match_jerarquia_rank']}, "
                    f"manual_match={row['topk_match_jerarquia_manual_codigo']})"
                )
            return "parcial", detalle
        if pd.isna(row["llm_codigo"]):
            return "no coincide", "sin_llm"
        return "no coincide", "sin_match"

    clasif = comparacion.apply(_clasificar, axis=1)
    comparacion["COMPARACION"] = clasif.apply(lambda x: x[0])
    comparacion["detalle_comparacion"] = clasif.apply(lambda x: x[1])
    comparacion["justificacion_comparacion"] = comparacion["detalle_comparacion"]

    total_manual = len(manual_filtrado)
    con_llm = comparacion["llm_codigo"].notna().sum()
    coincidencias = (comparacion["COMPARACION"] == "ok").sum()
    parciales = (comparacion["COMPARACION"] == "parcial").sum()
    no_coincide = (comparacion["COMPARACION"] == "no coincide").sum()
    discrepancias = comparacion[
        (comparacion["manual_codigo_granular"].notna())
        & (comparacion["llm_codigo"].notna())
        & (comparacion["COMPARACION"] == "no coincide")
    ]
    sin_clas_llm = comparacion[
        (comparacion["manual_codigo_granular"].notna()) & (comparacion["llm_codigo"].isna())
    ]

    print(f"\n{'=' * 60}")
    print("EVALUACION VS CORRESPONDENCIA MANUAL:")
    print(f"  Total con etiqueta manual: {total_manual}")
    print(f"  LLM con codigo asignado: {con_llm}")
    print(f"  Coincidencias (OK): {coincidencias}")
    print(f"  Parciales: {parciales}")
    print(f"  No coincide: {no_coincide}")
    print(f"  Discrepancias (LLM distinto y sin match): {len(discrepancias)}")
    print(f"  Manual con NINGUNA respuesta LLM: {len(sin_clas_llm)}")
    if len(discrepancias) > 0:
        print("  Muestras de discrepancias (codigo_manual -> codigo_llm):")
        for _, row in discrepancias.head(5).iterrows():
            print(
                f"    CNP {row['cnp_codigo']}: {row['manual_codigo_granular']} ({row['glosa_manual']}) "
                f"vs {row['llm_codigo']} ({row['iccs_glosa_elegida']})"
            )
    print(f"{'=' * 60}")

    topk_cols: list[str] = []
    for idx in range(1, TOP_K + 1):
        topk_cols.extend([f"top{idx}_codigo", f"top{idx}_glosa", f"top{idx}_score"])

    detalle_cols = [
        "cnp_codigo",
        "cnp_glosa",
        "cnp_descripcion",
        "cnp_familia",
        "cnp_articulado",
        "glosa_manual",
        "manual_codigo_granular",
        "manual_glosa_iccs",
        "manual_codigos",
        "manual_codigos_str",
        "manual_glosas_iccs",
        "manual_glosas_iccs_str",
        "iccs_elegido",
        "iccs_elegido_norm",
        "llm_codigo",
        "iccs_glosa_elegida",
        "confianza",
        "justificacion",
        "exclusiones_aplicadas",
        "exclusiones_aplicadas_str",
        "coincide_granular",
        "coincide_manual",
        "coincide_jerarquia_granular",
        "tipo_relacion_jerarquia_granular",
        "coincide_jerarquia_manual",
        "manual_match_jerarquia",
        "manual_match_jerarquia_glosa",
        "tipo_relacion_jerarquia_manual",
        "topk_coincide_manual",
        "topk_coincide_jerarquia_manual",
        "COMPARACION",
        "detalle_comparacion",
        "justificacion_comparacion",
        "topk_match_codigo",
        "topk_match_rank",
        "topk_match_glosa",
        "topk_match_jerarquia_codigo",
        "topk_match_jerarquia_rank",
        "topk_match_jerarquia_glosa",
        "topk_match_jerarquia_manual_codigo",
        "topk_match_jerarquia_manual_glosa",
        "topk_match_jerarquia_tipo",
        *topk_cols,
    ]
    detalle = comparacion.copy()
    detalle_cols = [c for c in detalle_cols if c in detalle.columns]
    extra_cols = [c for c in detalle.columns if c not in detalle_cols]
    detalle = detalle[detalle_cols + extra_cols]
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
    parser.add_argument(
        "--modo",
        choices=["llm", "comparar"],
        help="Modo de ejecucion: llm (consulta LLM) o comparar (sin LLM)",
    )
    parser.add_argument(
        "--llm-output",
        type=str,
        help="Ruta a la salida LLM para comparar (default: outputs/clasificacion_con_justificacion.csv)",
    )
    args = parser.parse_args()

    modo = args.modo or seleccionar_modo()
    if modo == "salir":
        print("Cancelado.")
        sys.exit(0)

    if modo == "comparar":
        if args.test or args.limite:
            print("Aviso: --test y --limite se ignoran en modo comparar.")

        iccs_full_df = cargar_iccs_descripcion()
        iccs_glosa_map = build_iccs_glosa_map(iccs_full_df)

        llm_output_path = Path(args.llm_output) if args.llm_output else OUTPUT_DIR / "clasificacion_con_justificacion.csv"
        df_resultados = cargar_salida_llm(llm_output_path)
        evaluar_contra_manual(df_resultados, iccs_full_df, iccs_glosa_map, OUTPUT_DIR)
        return

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

    client = get_openai_client()

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
