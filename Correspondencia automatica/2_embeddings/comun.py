#!/usr/bin/env python3
"""Utilidades compartidas del pipeline de embeddings + rerank CNP -> ICCS.

Centraliza: rutas, carga de insumos (CNP TC_2025 e ICCS), construcción de
textos, mapeo de sección ICCS (N1) y carga de modelos de embeddings.

Usado por: preparar_embeddings.py, rerank_matches.py, evaluar_ab.py
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CA_DIR = REPO_ROOT / "Correspondencia automatica"

# Insumo CNP vigente: tabla de correspondencia 2025 (una sola hoja)
CNP_TC2025_PATH = REPO_ROOT / "Correspondencia manual" / "2025" / "09062026_TC_2025_v1.0.xlsx"
CNP_TC2025_SHEET = "TC_2025_v1.0"

# Catálogo ICCS (metadatos + jerarquía)
ICCS_DESC_PATH = CA_DIR / "1_iccs" / "outputs" / "iccs_descripcion.csv"
ICCS_TABLA_PATH = CA_DIR / "1_iccs" / "outputs" / "iccs_tabla.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ----------------------------------------------------------------------------
# Modelos de embeddings disponibles
# ----------------------------------------------------------------------------
EMBED_MODELS = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "e5": "intfloat/multilingual-e5-large",
}
RERANKER_DEFAULT = "BAAI/bge-reranker-v2-m3"

# Instrucción para la query de Qwen3-Embedding (retrieval asimétrico)
QWEN3_TASK = (
    "Dado un delito del código penal nacional chileno, recupera la categoría "
    "de la Clasificación Internacional de Delitos (ICCS) que mejor lo clasifica."
)


# ----------------------------------------------------------------------------
# Texto
# ----------------------------------------------------------------------------
def normalize_text(parts: Iterable[str]) -> str:
    """Limpia y concatena textos con ' | ', colapsando espacios."""
    cleaned = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text and text.lower() != "nan":
            cleaned.append(text)
    return " ".join(" | ".join(cleaned).split())


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_accents_lower(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


# ----------------------------------------------------------------------------
# Carga CNP (insumo TC_2025)
# ----------------------------------------------------------------------------
def _find_col(cols: list[str], *candidates: str) -> str | None:
    norm = {_strip_accents_lower(c): c for c in cols}
    for cand in candidates:
        key = _strip_accents_lower(cand)
        if key in norm:
            return norm[key]
    return None


def load_cnp(path: Path = CNP_TC2025_PATH, sheet: str = CNP_TC2025_SHEET) -> pd.DataFrame:
    """Carga el insumo CNP 2025 y lo normaliza a columnas estándar.

    Columnas de salida:
      codigo, glosa, descripcion, familia_nombre, situacion_2025,
      iccs_2025_manual, estado, texto

    `estado` = 'no_clasificado' cuando el código no tiene glosa, descripción
    ni familia (p.ej. CUM 0); estos NO se envían a embeddings/match.
    """
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    c_cum = _find_col(cols, "CUM")
    c_glosa = _find_col(cols, "GLOSA")
    c_desc = _find_col(cols, "Descripción", "Descripcion")
    c_fam = _find_col(cols, "Familia de delito")
    c_sit = _find_col(cols, "Situación 2025", "Situacion 2025")
    c_iccs = _find_col(cols, "ICCS _2025", "ICCS_2025", "ICCS 2025")
    if c_cum is None:
        raise ValueError(f"No se encontró la columna CUM en {path.name}: {cols}")

    out = pd.DataFrame()
    out["codigo"] = df[c_cum].apply(_fmt_cum)
    out["glosa"] = df[c_glosa].fillna("") if c_glosa else ""
    out["descripcion"] = df[c_desc].fillna("") if c_desc else ""
    out["familia_nombre"] = df[c_fam].fillna("") if c_fam else ""
    out["situacion_2025"] = df[c_sit].fillna("").astype(str).str.strip() if c_sit else ""
    out["iccs_2025_manual"] = df[c_iccs].fillna("").astype(str).str.strip() if c_iccs else ""

    for col in ("glosa", "descripcion", "familia_nombre"):
        out[col] = out[col].astype(str).str.strip().replace({"nan": ""})

    # Estado: no_clasificado si no hay contenido textual para clasificar
    sin_contenido = (out["glosa"] == "") & (out["descripcion"] == "") & (out["familia_nombre"] == "")
    out["estado"] = "matchable"
    out.loc[sin_contenido, "estado"] = "no_clasificado"

    out["texto"] = out.apply(
        lambda r: normalize_text([r["glosa"], r["descripcion"], r["familia_nombre"]]), axis=1
    )
    out["texto_hash"] = out["texto"].apply(hash_text)
    return out


def _fmt_cum(value) -> str:
    """CUM como string entero ('101'), tolerando floats ('101.0') y nulos."""
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


# ----------------------------------------------------------------------------
# Carga ICCS
# ----------------------------------------------------------------------------
def load_iccs(path: Path = ICCS_DESC_PATH) -> pd.DataFrame:
    """Carga catálogo ICCS y construye el texto de pasaje.

    Columnas de salida añadidas: texto, texto_hash, seccion_n1 (int 1-11).
    """
    iccs = pd.read_csv(path)
    for col in ("glosa_iccs", "descripcion", "inclusiones", "seccion"):
        if col not in iccs.columns:
            iccs[col] = ""
        iccs[col] = iccs[col].fillna("")
    iccs["codigo_iccs"] = iccs["codigo_iccs"].astype(str)

    def _build(row: pd.Series) -> str:
        parts = [row["glosa_iccs"], row["descripcion"], row["inclusiones"]]
        if str(row["seccion"]).strip():
            parts.append(f"Sección {row['seccion']}")
        return normalize_text(parts)

    iccs["texto"] = iccs.apply(_build, axis=1)
    iccs["texto_hash"] = iccs["texto"].apply(hash_text)

    sec_map = seccion_to_n1()
    iccs["seccion_n1"] = iccs["seccion"].map(
        lambda s: sec_map.get(_strip_accents_lower(s))
    )
    return iccs


def seccion_to_n1(path: Path = ICCS_TABLA_PATH) -> dict[str, int]:
    """Mapa: texto de sección (normalizado) -> número de sección N1 (1-11)."""
    t = pd.read_csv(path)
    m: dict[str, int] = {}
    for _, row in t.dropna(subset=["seccion"]).iterrows():
        try:
            n1 = int(row["nivel_1"])
        except (ValueError, TypeError):
            continue
        m[_strip_accents_lower(row["seccion"])] = n1
    return m


def code_to_n1_map(iccs: pd.DataFrame) -> dict[str, int]:
    """Mapa: codigo_iccs -> sección N1 (1-11), vía la sección del catálogo."""
    return {
        str(r["codigo_iccs"]): int(r["seccion_n1"])
        for _, r in iccs.iterrows()
        if pd.notna(r.get("seccion_n1"))
    }


def parse_manual_n1(value: str) -> int | None:
    """Convierte la etiqueta manual ICCS_2025 a sección N1 (1-11) o None.

    Valores no numéricos ('No clasificado_NV', 'Excluido') -> None.
    """
    s = str(value).strip()
    if re.fullmatch(r"\d{1,2}", s):
        n = int(s)
        if 1 <= n <= 11:
            return n
    return None


# ----------------------------------------------------------------------------
# Construcción de inputs por modelo
# ----------------------------------------------------------------------------
def build_query_texts(texts: list[str], model_key: str) -> list[str]:
    """Aplica el prefijo/instrucción de query según el modelo."""
    if model_key == "e5":
        return [f"query: {t}" for t in texts]
    if model_key == "qwen3":
        return [f"Instruct: {QWEN3_TASK}\nQuery: {t}" for t in texts]
    raise ValueError(f"Modelo desconocido: {model_key}")


def build_doc_texts(texts: list[str], model_key: str) -> list[str]:
    """Aplica el prefijo de documento según el modelo (Qwen3 no usa prefijo)."""
    if model_key == "e5":
        return [f"passage: {t}" for t in texts]
    if model_key == "qwen3":
        return list(texts)
    raise ValueError(f"Modelo desconocido: {model_key}")
