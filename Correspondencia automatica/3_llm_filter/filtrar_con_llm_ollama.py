#!/usr/bin/env python3
"""Filtro LLM CNP -> ICCS con Ollama (qwen3:8b) — etapa 3.

Migración desde la API de OpenAI a un modelo local servido por Ollama.
Toma el top-k REORDENADO por el reranker (etapa 2.b) y, con razonamiento legal,
elige el mejor código ICCS por delito CNP, considerando exclusiones/notas.

NO requiere API key. Requiere Ollama corriendo localmente:
    ollama pull qwen3:8b
    ollama serve            # (o el servicio de Ollama ya activo)

Este script está PREPARADO pero pensado para ejecutarse en la máquina con GPU.
Ejemplos:
    python filtrar_con_llm_ollama.py --test          # 10 códigos
    python filtrar_con_llm_ollama.py                 # todos
    python filtrar_con_llm_ollama.py --model qwen3:8b --host http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CA_DIR = REPO_ROOT / "Correspondencia automatica"

# Insumo: top-k reordenado por el reranker (etapa 2.b, modelo qwen3 por defecto)
MATCHES_RERANK_PATH = CA_DIR / "2_embeddings" / "outputs" / "qwen3" / "matches_rerank_detallado.csv"
# Respaldo: si no hay rerank, usar el top-k de embeddings
MATCHES_EMBED_PATH = CA_DIR / "2_embeddings" / "outputs" / "qwen3" / "matches_detallado.csv"
EMBED_SCORE_PATHS = {
    "qwen3": CA_DIR / "2_embeddings" / "outputs" / "qwen3" / "matches_detallado.csv",
    "e5": CA_DIR / "2_embeddings" / "outputs" / "e5" / "matches_detallado.csv",
}
ICCS_DESCRIPCION_PATH = CA_DIR / "1_iccs" / "outputs" / "iccs_descripcion.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"
TOP_K = 10
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = (
    "Eres un experto en clasificacion de delitos penales. "
    "Respondes solo en JSON valido. No inventes datos fuera de los insumos. "
    "Clasifica segun el MOVIL del delito, no la consecuencia mas grave. "
    "Elige de los candidatos presentados o de codigos mencionados en exclusiones/inclusiones. "
    "NUNCA inventes codigos arbitrarios. "
    "Para delitos sin descripcion, clasifica con glosa y familia. "
    "Solo responde NINGUNO si el delito es completamente generico sin contexto suficiente."
)


def cargar_matches(path: Path | None) -> pd.DataFrame:
    if path is None:
        path = MATCHES_RERANK_PATH if MATCHES_RERANK_PATH.exists() else MATCHES_EMBED_PATH
    if not path.exists():
        raise SystemExit(
            f"No se encontró el insumo de candidatos: {path}\n"
            "Corre primero la etapa 2 (preparar_embeddings.py + rerank_matches.py)."
        )
    print(f"Insumo de candidatos: {path}")
    df = pd.read_csv(path, dtype={"cnp_codigo": str, "iccs_codigo": str})
    # rank de orden: usa rank_rerank si existe (rerank), si no rank (embeddings)
    df["rank_orden"] = df["rank_rerank"] if "rank_rerank" in df.columns else df["rank"]
    return df


def _fmt_score(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def cargar_contexto_embeddings(top_k: int) -> tuple[dict, dict]:
    """Carga ranks/scores top-k de cada modelo de embeddings para el prompt."""
    score_lookup: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    top_lookup: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_key, path in EMBED_SCORE_PATHS.items():
        if not path.exists():
            print(f"Advertencia: no existe {path}; se omiten scores {model_key}.")
            continue
        df = pd.read_csv(path, dtype={"cnp_codigo": str, "iccs_codigo": str})
        df = df.sort_values(["cnp_codigo", "rank"])
        score_lookup[model_key] = {}
        top_lookup[model_key] = {}
        for cnp_codigo, grp in df.groupby("cnp_codigo", sort=False):
            grp_top = grp.head(top_k)
            top_lookup[model_key][str(cnp_codigo)] = [
                {
                    "rank": int(m["rank"]),
                    "iccs_codigo": str(m["iccs_codigo"]),
                    "iccs_glosa": _clean_text(m.get("iccs_glosa", "")),
                    "iccs_seccion": _clean_text(m.get("iccs_seccion", "")),
                    "score": float(m["similarity_score"]),
                }
                for _, m in grp_top.iterrows()
            ]
            for _, m in grp.iterrows():
                score_lookup[model_key][(str(cnp_codigo), str(m["iccs_codigo"]))] = {
                    "rank": int(m["rank"]),
                    "score": float(m["similarity_score"]),
                }
    return score_lookup, top_lookup


def _score_info(score_lookup: dict, model_key: str, cnp_codigo: str, iccs_codigo: str) -> dict[str, Any]:
    return score_lookup.get(model_key, {}).get((str(cnp_codigo), str(iccs_codigo)), {})


def preparar_candidatos(
    matches: pd.DataFrame,
    iccs_full: pd.DataFrame,
    top_k: int,
    score_lookup: dict,
    top_lookup: dict,
) -> dict[str, dict]:
    """Agrupa por CNP y une con ICCS para exclusiones/notas."""
    matches = matches.copy()
    matches["cnp_codigo"] = matches["cnp_codigo"].astype(str)
    matches["iccs_codigo"] = matches["iccs_codigo"].astype(str)
    iccs_full = iccs_full.copy()
    iccs_full["codigo_iccs"] = iccs_full["codigo_iccs"].astype(str)
    iccs_idx = iccs_full.set_index("codigo_iccs")

    por_cnp: dict[str, dict] = {}
    for cnp_codigo, grp in matches.groupby("cnp_codigo", sort=False):
        grp = grp.sort_values("rank_orden").head(top_k)
        info = grp.iloc[0]
        candidatos = []
        for _, m in grp.iterrows():
            cod = str(m["iccs_codigo"])
            excl = notas = seccion = ""
            if cod in iccs_idx.index:
                fila = iccs_idx.loc[cod]
                excl = _clean_text(fila.get("exclusiones", ""))
                notas = _clean_text(fila.get("notas", ""))
                seccion = _clean_text(fila.get("seccion", ""))
            score_rerank = m.get("rerank_score")
            qwen3 = _score_info(score_lookup, "qwen3", str(cnp_codigo), cod)
            e5 = _score_info(score_lookup, "e5", str(cnp_codigo), cod)
            if not qwen3:
                qwen3 = {"rank": int(m.get("rank", 0)), "score": float(m.get("similarity_score", 0.0))}
            candidatos.append({
                "rank": int(m["rank_orden"]),
                "iccs_codigo": cod,
                "iccs_glosa": _clean_text(m.get("iccs_glosa", "")),
                "iccs_seccion": seccion or _clean_text(m.get("iccs_seccion", "")),
                "iccs_descripcion": _clean_text(m.get("iccs_descripcion", "")),
                "iccs_inclusiones": _clean_text(m.get("iccs_inclusiones", "")),
                "iccs_exclusiones": excl,
                "iccs_notas": notas,
                "rerank_score": None if pd.isna(score_rerank) else float(score_rerank),
                "qwen3_rank": qwen3.get("rank"),
                "qwen3_score": qwen3.get("score"),
                "e5_rank": e5.get("rank"),
                "e5_score": e5.get("score"),
            })
        cnp_key = str(cnp_codigo)
        por_cnp[cnp_key] = {
            "cnp_codigo": str(info["cnp_codigo"]),
            "cnp_glosa": _clean_text(info.get("cnp_glosa", "")),
            "cnp_descripcion": _clean_text(info.get("cnp_descripcion", "")),
            "cnp_familia": _clean_text(info.get("cnp_familia", "")),
            "candidatos": candidatos,
            "embedding_top10": {
                "qwen3": top_lookup.get("qwen3", {}).get(cnp_key, []),
                "e5": top_lookup.get("e5", {}).get(cnp_key, []),
            },
        }
    return por_cnp


def construir_resumen_embeddings(cnp: dict) -> str:
    lines = []
    labels = {
        "qwen3": "Qwen3-Embedding-0.6B",
        "e5": "multilingual-e5-large",
    }
    for model_key in ("qwen3", "e5"):
        top = cnp.get("embedding_top10", {}).get(model_key, [])
        if not top:
            continue
        items = []
        for item in top[:10]:
            seccion = str(item.get("iccs_seccion", "")).strip()
            sec_txt = f" | seccion: {seccion}" if seccion else ""
            items.append(
                f"{item['rank']}) {item['iccs_codigo']} "
                f"score={_fmt_score(item.get('score'))} - {item['iccs_glosa']}{sec_txt}"
            )
        lines.append(f"{labels[model_key]}: " + " ; ".join(items))
    return "\n".join(lines)


def construir_prompt(cnp: dict) -> str:
    cands = ""
    for c in cnp["candidatos"]:
        rerank_txt = _fmt_score(c.get("rerank_score"))
        qwen3_txt = (
            f"rank {c.get('qwen3_rank')} score {_fmt_score(c.get('qwen3_score'))}"
            if c.get("qwen3_rank") is not None else "NA"
        )
        e5_txt = (
            f"rank {c.get('e5_rank')} score {_fmt_score(c.get('e5_score'))}"
            if c.get("e5_rank") is not None else "NA"
        )
        cands += (
            f"\n{c['rank']}. Codigo ICCS: {c['iccs_codigo']}\n"
            f"   Glosa: {c['iccs_glosa']}\n"
            f"   Seccion ICCS: {c['iccs_seccion']}\n"
            f"   Descripcion: {c['iccs_descripcion']}\n"
            f"   Inclusiones: {c['iccs_inclusiones']}\n"
            f"   EXCLUSIONES: {c['iccs_exclusiones']}\n"
            f"   NOTAS: {c['iccs_notas']}\n"
            f"   Scores: rerank={rerank_txt}; embedding_qwen3=({qwen3_txt}); embedding_e5=({e5_txt})\n"
        )
    codigos_validos = ", ".join(c["iccs_codigo"] for c in cnp["candidatos"])
    n = len(cnp["candidatos"])
    resumen_embeddings = construir_resumen_embeddings(cnp)
    return f"""Eres un experto en clasificacion de delitos penales. Tu tarea es mapear un delito del Codigo Penal Nacional (CNP) chileno a la Clasificacion Internacional de Delitos con Fines Estadisticos (ICCS) de la ONU.

DELITO NACIONAL (CNP):
- Codigo: {cnp['cnp_codigo']}
- Glosa: {cnp['cnp_glosa']}
- Descripcion: {cnp['cnp_descripcion']}
- Familia: {cnp['cnp_familia']}

CANDIDATOS ICCS (Top {n} reordenados por relevancia):
{cands}

TOP 10 POR MODELO DE EMBEDDING (contexto comparativo; las escalas no son identicas):
{resumen_embeddings}

INSTRUCCIONES CRITICAS:
1. Elige el codigo ICCS que MEJOR se aproxime a la definicion del delito CNP.
2. NO busques el codigo mas especifico; busca el MAS PRECISO (puede ser general si es mas exacto).
3. Considera ESPECIALMENTE las EXCLUSIONES y NOTAS de cada candidato.
4. Si una exclusion descarta el delito CNP, ese candidato NO es valido.
5. Las NOTAS dan contexto sobre cuando aplicar cada codigo.
6. La SECCION ICCS situa el marco juridico/estadistico del delito. Si dos candidatos comparten palabras parecidas pero pertenecen a secciones incompatibles, prefiere el candidato cuya seccion corresponda al marco real del delito CNP.
7. CRITERIO PRINCIPAL - MOVIL DEL DELITO: clasifica segun la intencion INICIAL del delincuente, no la consecuencia mas grave.
   - "Robo con homicidio" -> ROBO (el movil es apropiarse de bienes).
   - "Secuestro extorsivo" -> SECUESTRO (el movil es privar de libertad).
   - "Violacion con homicidio para eliminar testigo" -> VIOLACION (el movil original era sexual).
8. DELITOS SIN DESCRIPCION: clasifica con GLOSA y FAMILIA; solo "NINGUNO" si es completamente generico.
9. Usa los scores Qwen3/E5 como evidencia de recuperacion semantica. Si ambos modelos ubican alto un codigo y la seccion/exclusiones son compatibles, eso aumenta la confianza; no elijas por score si el analisis legal lo descarta.
10. Si NINGUN candidato es apropiado tras aplicar las reglas, devuelve "NINGUNO" y explica por que.
11. PROHIBIDO INVENTAR: elige uno de los {n} codigos listados. EXCEPCION: si una EXCLUSION/INCLUSION menciona explicitamente otro codigo ICCS que describe mejor el delito, puedes elegirlo justificando donde aparece.
12. Tu analisis legal es prioritario sobre el score.

CODIGOS ICCS VALIDOS (elige UNO o "NINGUNO"): {codigos_validos}

RESPONDE UNICAMENTE CON UN OBJETO JSON (sin markdown):
{{
  "iccs_elegido": "codigo elegido o NINGUNO",
  "confianza": "alta|media|baja",
  "justificacion": "explicacion breve; menciona exclusiones aplicadas",
  "exclusiones_aplicadas": ["exclusiones que descartaron otros candidatos, vacio si no aplica"]
}}"""


def llamar_ollama(prompt: str, model: str, host: str, cnp_codigo: str) -> dict[str, Any] | None:
    """Llama a Ollama /api/chat (JSON, sin 'thinking') con reintentos."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "think": False,            # qwen3: desactiva el modo de razonamiento extendido
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 512},
    }
    data = json.dumps(payload).encode("utf-8")
    for intento in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            contenido = body.get("message", {}).get("content", "")
            resultado = json.loads(contenido)
            if not all(k in resultado for k in ("iccs_elegido", "confianza", "justificacion")):
                raise ValueError(f"JSON incompleto: {resultado}")
            return resultado
        except (urllib.error.URLError, json.JSONDecodeError, ValueError) as e:
            print(f"  CNP {cnp_codigo}: fallo intento {intento}/{MAX_RETRIES} ({e})")
            if intento < MAX_RETRIES:
                time.sleep(RETRY_DELAY * intento)
    return None


def guardar(resultados: list[dict], output_dir: Path, quiet: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(resultados)
    _to_csv_atomic(df, output_dir / "clasificacion_con_justificacion.csv")
    cols = ["cnp_codigo", "cnp_glosa", "iccs_elegido", "iccs_glosa_elegida", "confianza"]
    _to_csv_atomic(df[[c for c in cols if c in df.columns]], output_dir / "clasificacion_final.csv")
    if not quiet:
        print(f"\nResultados en {output_dir}")


def _to_csv_atomic(df: pd.DataFrame, path: Path, retries: int = 5) -> None:
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    last_error: Exception | None = None
    for intento in range(1, retries + 1):
        try:
            df.to_csv(tmp, index=False, encoding="utf-8-sig")
            tmp.replace(path)
            return
        except OSError as e:
            last_error = e
            time.sleep(0.5 * intento)
    raise last_error or OSError(f"No se pudo escribir {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo Ollama (default: {DEFAULT_MODEL}).")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Host Ollama (default: {DEFAULT_HOST}).")
    p.add_argument("--input", default=None, help="CSV de candidatos (default: rerank qwen3).")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--test", action="store_true", help="Procesa solo 10 códigos.")
    p.add_argument("--limite", type=int, default=None)
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--no-resume", action="store_true",
                   help="No reutiliza clasificacion_con_justificacion.csv existente.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matches = cargar_matches(Path(args.input) if args.input else None)
    iccs_full = pd.read_csv(ICCS_DESCRIPCION_PATH)
    glosa_map = dict(zip(iccs_full["codigo_iccs"].astype(str), iccs_full["glosa_iccs"].astype(str)))
    score_lookup, top_lookup = cargar_contexto_embeddings(args.top_k)

    por_cnp = preparar_candidatos(matches, iccs_full, args.top_k, score_lookup, top_lookup)
    codigos = list(por_cnp)
    if args.test:
        codigos = codigos[:10]
    elif args.limite:
        codigos = codigos[:args.limite]

    output_dir = Path(args.output_dir)
    resultados = []
    if not args.test and not args.no_resume:
        prev_path = output_dir / "clasificacion_con_justificacion.csv"
        if prev_path.exists():
            prev = pd.read_csv(prev_path, dtype={"cnp_codigo": str})
            if len(prev):
                resultados = prev.to_dict("records")
                ya_procesados = set(prev["cnp_codigo"].astype(str))
                codigos = [c for c in codigos if str(c) not in ya_procesados]
                print(f"Reanudando: {len(resultados)} códigos ya procesados; "
                      f"{len(codigos)} pendientes.", flush=True)

    total_objetivo = len(resultados) + len(codigos)
    print(f"Procesando {len(codigos)} códigos CNP pendientes con {args.model} en {args.host} ...",
          flush=True)

    for cod in codigos:
        cnp = por_cnp[cod]
        prompt = construir_prompt(cnp)
        r = llamar_ollama(prompt, args.model, args.host, cod)
        elegido = (r or {}).get("iccs_elegido", "ERROR")
        resultados.append({
            "cnp_codigo": cnp["cnp_codigo"],
            "cnp_glosa": cnp["cnp_glosa"],
            "cnp_descripcion": cnp["cnp_descripcion"],
            "cnp_familia": cnp["cnp_familia"],
            "iccs_elegido": elegido,
            "iccs_glosa_elegida": glosa_map.get(str(elegido), ""),
            "confianza": (r or {}).get("confianza", ""),
            "justificacion": (r or {}).get("justificacion", ""),
            "exclusiones_aplicadas": "; ".join((r or {}).get("exclusiones_aplicadas", []) or []),
        })
        guardar(resultados, output_dir, quiet=True)
        if len(resultados) % 10 == 0:
            print(f"  {len(resultados)}/{total_objetivo}", flush=True)

    guardar(resultados, output_dir)


if __name__ == "__main__":
    main()
