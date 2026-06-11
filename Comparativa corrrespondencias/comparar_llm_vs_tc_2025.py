#!/usr/bin/env python3
"""Compara la clasificacion LLM contra la TC manual 2025.

La TC 2025 trae `ICCS _2025` a nivel de seccion ICCS (N1, valores 1-11).
La salida LLM puede venir en un nivel mas detallado, por lo que primero se
expande el codigo LLM a N1-N4 usando el catalogo ICCS y luego se compara N1.

Salida por defecto:
    Comparativa corrrespondencias/comparacion_llm_vs_tc_2025.xlsx
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANUAL_PATH = (
    REPO_ROOT
    / "Correspondencia manual"
    / "2025"
    / "09062026_TC_2025_v1.0.xlsx"
)
DEFAULT_MANUAL_SHEET = "TC_2025_v1.0"
DEFAULT_LLM_PATH = (
    REPO_ROOT
    / "Correspondencia automatica"
    / "3_llm_filter"
    / "outputs"
    / "clasificacion_con_justificacion.csv"
)
DEFAULT_ICCS_DESC_PATH = (
    REPO_ROOT
    / "Correspondencia automatica"
    / "1_iccs"
    / "outputs"
    / "iccs_descripcion.csv"
)
DEFAULT_ICCS_TABLA_PATH = (
    REPO_ROOT
    / "Correspondencia automatica"
    / "1_iccs"
    / "outputs"
    / "iccs_tabla.csv"
)
DEFAULT_TOP10_PATH = (
    REPO_ROOT
    / "Correspondencia automatica"
    / "2_embeddings"
    / "outputs"
    / "qwen3"
    / "matches_rerank_detallado.csv"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "comparacion_llm_vs_tc_2025.xlsx"


LEVELS = ("n1", "n2", "n3", "n4")
TC_REQUIRED_COLS = ("CUM", "GLOSA", "Descripción", "Familia de delito", "ICCS _2025")
LLM_REQUIRED_COLS = (
    "cnp_codigo",
    "cnp_glosa",
    "cnp_descripcion",
    "cnp_familia",
    "iccs_elegido",
    "confianza",
    "justificacion",
)


def clean_text(value: Any) -> str:
    """Devuelve texto limpio, dejando nulos como cadena vacia."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def strip_accents_lower(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def normalize_code(value: Any) -> str:
    """Normaliza codigos CUM/ICCS guardados como texto, entero o float."""
    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("\ufeff", "").strip()
    if re.fullmatch(r"\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def parse_manual_n1(value: Any) -> int | None:
    """Convierte `ICCS _2025` a seccion N1 numerica; no numericos quedan fuera."""
    text = normalize_code(value)
    if re.fullmatch(r"\d{1,2}", text):
        n1 = int(text)
        if 1 <= n1 <= 11:
            return n1
    return None


def require_columns(df: pd.DataFrame, cols: tuple[str, ...], source: Path) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {source}: {missing}")


def read_manual(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    df.columns = [clean_text(c) for c in df.columns]
    require_columns(df, TC_REQUIRED_COLS, path)
    df["cum"] = df["CUM"].map(normalize_code)
    df["manual_iccs_2025_raw"] = df["ICCS _2025"].map(clean_text)
    df["manual_n1_codigo"] = df["manual_iccs_2025_raw"].map(parse_manual_n1)
    df["manual_n1_codigo"] = df["manual_n1_codigo"].astype("Int64")
    return df


def read_llm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    require_columns(df, LLM_REQUIRED_COLS, path)
    df["cum"] = df["cnp_codigo"].map(normalize_code)
    df["llm_iccs_elegido_norm"] = df["iccs_elegido"].map(normalize_code)
    return df


def load_iccs_catalog(desc_path: Path, tabla_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Carga metadatos ICCS y un lookup codigo -> niveles N1-N4."""
    desc = pd.read_csv(desc_path, dtype=str).fillna("")
    tabla = pd.read_csv(tabla_path, dtype=str).fillna("")

    desc["codigo_iccs"] = desc["codigo_iccs"].map(normalize_code)
    meta = {}
    for _, row in desc.iterrows():
        code = row["codigo_iccs"]
        if not code:
            continue
        meta[code] = {
            "codigo": code,
            "glosa": clean_text(row.get("glosa_iccs", "")),
            "seccion": clean_text(row.get("seccion", "")),
            "descripcion": clean_text(row.get("descripcion", "")),
            "inclusiones": clean_text(row.get("inclusiones", "")),
            "exclusiones": clean_text(row.get("exclusiones", "")),
            "notas": clean_text(row.get("notas", "")),
        }

    n1_to_section: dict[str, str] = {}
    code_to_levels: dict[str, dict] = {}
    for _, row in tabla.iterrows():
        levels = {
            "n1": normalize_code(row.get("nivel_1", "")),
            "n2": normalize_code(row.get("nivel_2", "")),
            "n3": normalize_code(row.get("nivel_3", "")),
            "n4": normalize_code(row.get("nivel_4", "")),
        }
        section = clean_text(row.get("seccion", ""))
        if levels["n1"] and section:
            n1_to_section.setdefault(levels["n1"], section)

        deepest_code = ""
        deepest_level = ""
        for level in LEVELS:
            if levels[level]:
                deepest_code = levels[level]
                deepest_level = level
        if not deepest_code:
            continue
        code_to_levels.setdefault(
            deepest_code,
            {
                **levels,
                "nivel_codigo_elegido": deepest_level.upper(),
                "seccion": section,
            },
        )

    for n1, section in n1_to_section.items():
        meta.setdefault(
            n1,
            {
                "codigo": n1,
                "glosa": section,
                "seccion": section,
                "descripcion": "",
                "inclusiones": "",
                "exclusiones": "",
                "notas": "",
            },
        )
    return meta, code_to_levels


def expand_iccs_code(code: Any, meta: dict[str, dict], code_to_levels: dict[str, dict]) -> dict[str, Any]:
    """Expande un codigo ICCS elegido por el LLM a niveles y metadatos."""
    code_norm = normalize_code(code)
    out: dict[str, Any] = {
        "llm_iccs_elegido_norm": code_norm,
        "llm_nivel_codigo_elegido": "",
    }
    if not code_norm or code_norm.upper() == "NINGUNO":
        for level in LEVELS:
            add_level_metadata(out, level, "", meta)
        return out

    levels = code_to_levels.get(code_norm)
    if levels is None:
        levels = infer_levels_from_metadata(code_norm, meta)

    out["llm_nivel_codigo_elegido"] = levels.get("nivel_codigo_elegido", "")
    for level in LEVELS:
        add_level_metadata(out, level, levels.get(level, ""), meta)
    return out


def infer_levels_from_metadata(code: str, meta: dict[str, dict]) -> dict[str, str]:
    """Respaldo si el codigo no aparece en iccs_tabla.csv."""
    section = meta.get(code, {}).get("seccion", "")
    n1 = ""
    if section:
        section_norm = strip_accents_lower(section)
        for candidate, item in meta.items():
            if len(candidate) <= 2 and strip_accents_lower(item.get("seccion", "")) == section_norm:
                n1 = candidate
                break

    level_codes = {"n1": n1, "n2": "", "n3": "", "n4": ""}
    prefixes = sorted(
        [candidate for candidate in meta if candidate != code and code.startswith(candidate)],
        key=len,
    )
    for candidate in prefixes:
        if len(candidate) <= 3:
            level_codes["n2"] = candidate
        elif len(candidate) == 4:
            level_codes["n3"] = candidate
        else:
            level_codes["n4"] = candidate

    if len(code) <= 3:
        level_codes["n2"] = code
        deepest = "N2"
    elif len(code) == 4:
        level_codes["n3"] = code
        deepest = "N3"
    else:
        level_codes["n4"] = code
        deepest = "N4"
    return {**level_codes, "nivel_codigo_elegido": deepest}


def add_level_metadata(out: dict[str, Any], level: str, code: Any, meta: dict[str, dict]) -> None:
    code_norm = normalize_code(code)
    item = meta.get(code_norm, {})
    prefix = f"llm_{level}"
    out[f"{prefix}_codigo"] = code_norm
    out[f"{prefix}_glosa"] = item.get("glosa", "")
    out[f"{prefix}_descripcion"] = item.get("descripcion", "")
    out[f"{prefix}_seccion"] = item.get("seccion", "")
    out[f"{prefix}_inclusiones"] = item.get("inclusiones", "")
    out[f"{prefix}_exclusiones"] = item.get("exclusiones", "")
    out[f"{prefix}_notas"] = item.get("notas", "")


def build_llm_expanded(llm: pd.DataFrame, meta: dict[str, dict],
                       code_to_levels: dict[str, dict]) -> pd.DataFrame:
    expanded = pd.DataFrame(
        [expand_iccs_code(code, meta, code_to_levels) for code in llm["iccs_elegido"]]
    )
    out = pd.concat([llm.reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)
    # Evita columnas duplicadas por la normalizacion previa.
    out = out.loc[:, ~out.columns.duplicated()]
    out["llm_n1_codigo_num"] = pd.to_numeric(out["llm_n1_codigo"], errors="coerce").astype("Int64")
    return out


def add_manual_n1_metadata(manual: pd.DataFrame, meta: dict[str, dict]) -> pd.DataFrame:
    out = manual.copy()
    out["manual_n1_glosa"] = out["manual_n1_codigo"].map(
        lambda value: meta.get(normalize_code(value), {}).get("glosa", "")
    )
    return out


def comparison_status(row: pd.Series) -> str:
    if row["_merge"] == "left_only":
        return "SOLO_TC_MANUAL"
    if row["_merge"] == "right_only":
        return "SOLO_LLM"
    if pd.isna(row["manual_n1_codigo"]):
        return "MANUAL_NO_COMPARABLE"
    if pd.isna(row["llm_n1_codigo_num"]):
        return "LLM_SIN_ICCS"
    if int(row["manual_n1_codigo"]) == int(row["llm_n1_codigo_num"]):
        return "COINCIDE_N1"
    return "DIVERGE_N1"


def build_comparison(manual: pd.DataFrame, llm_expanded: pd.DataFrame) -> pd.DataFrame:
    manual_cols = [
        "cum",
        "CUM",
        "GLOSA",
        "Descripción",
        "CodFam",
        "Familia de delito",
        "Situación 2024",
        "Situación 2025",
        "N1-2024 FINAL",
        "ICCS _2025",
        "manual_iccs_2025_raw",
        "manual_n1_codigo",
        "manual_n1_glosa",
    ]
    manual_cols = [col for col in manual_cols if col in manual.columns]

    llm_cols = [
        "cum",
        "cnp_codigo",
        "cnp_glosa",
        "cnp_descripcion",
        "cnp_familia",
        "iccs_elegido",
        "llm_iccs_elegido_norm",
        "llm_nivel_codigo_elegido",
        "llm_n1_codigo",
        "llm_n1_codigo_num",
        "llm_n1_glosa",
        "llm_n1_descripcion",
        "llm_n2_codigo",
        "llm_n2_glosa",
        "llm_n2_descripcion",
        "llm_n3_codigo",
        "llm_n3_glosa",
        "llm_n3_descripcion",
        "llm_n4_codigo",
        "llm_n4_glosa",
        "llm_n4_descripcion",
        "iccs_glosa_elegida",
        "confianza",
        "justificacion",
        "exclusiones_aplicadas",
    ]
    llm_cols = [col for col in llm_cols if col in llm_expanded.columns]

    merged = manual[manual_cols].merge(
        llm_expanded[llm_cols],
        on="cum",
        how="outer",
        indicator=True,
        suffixes=("_manual", "_llm"),
    )
    merged["estado_comparacion"] = merged.apply(comparison_status, axis=1)
    merged["coincide_n1"] = merged["estado_comparacion"].eq("COINCIDE_N1")
    merged["cum_glosa_revision"] = merged["GLOSA"].fillna("").where(
        merged["GLOSA"].fillna("").ne(""), merged.get("cnp_glosa", "")
    )
    merged["cum_descripcion_revision"] = merged["Descripción"].fillna("").where(
        merged["Descripción"].fillna("").ne(""), merged.get("cnp_descripcion", "")
    )
    merged["cum_familia_revision"] = merged["Familia de delito"].fillna("").where(
        merged["Familia de delito"].fillna("").ne(""), merged.get("cnp_familia", "")
    )
    merged["observacion_revision"] = merged["estado_comparacion"].map(
        {
            "COINCIDE_N1": "Coincide a primer nivel ICCS.",
            "DIVERGE_N1": "Revisar diferencia entre N1 manual y N1 del LLM.",
            "LLM_SIN_ICCS": "La TC tiene N1 manual valido, pero el LLM no entrego ICCS comparable.",
            "MANUAL_NO_COMPARABLE": "La TC no trae N1 numerico en ICCS _2025.",
            "SOLO_TC_MANUAL": "CUM presente solo en la TC manual.",
            "SOLO_LLM": "CUM presente solo en la salida LLM.",
        }
    )

    first_cols = [
        "cum",
        "estado_comparacion",
        "coincide_n1",
        "manual_iccs_2025_raw",
        "manual_n1_codigo",
        "manual_n1_glosa",
        "llm_iccs_elegido_norm",
        "llm_nivel_codigo_elegido",
        "llm_n1_codigo",
        "llm_n1_glosa",
        "confianza",
        "cum_glosa_revision",
        "cum_descripcion_revision",
        "cum_familia_revision",
        "observacion_revision",
    ]
    remaining = [col for col in merged.columns if col not in first_cols]
    return merged[first_cols + remaining].sort_values("cum", key=lambda s: s.map(sort_key))


def sort_key(value: Any) -> tuple[int, str]:
    code = normalize_code(value)
    if code.isdigit():
        return int(code), code
    return 10**9, code


def build_summary(comparison: pd.DataFrame, manual: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    counts = comparison["estado_comparacion"].value_counts()
    comparables = comparison["estado_comparacion"].isin(
        ["COINCIDE_N1", "DIVERGE_N1", "LLM_SIN_ICCS"]
    ).sum()
    coincidencias = counts.get("COINCIDE_N1", 0)
    divergencias = counts.get("DIVERGE_N1", 0)
    llm_sin_iccs = counts.get("LLM_SIN_ICCS", 0)
    tasa = coincidencias / comparables if comparables else 0
    rows = [
        ("TC manual 2025 - registros", len(manual)),
        ("Salida LLM - registros", len(llm)),
        ("Registros en ambos insumos", int(comparison["_merge"].eq("both").sum())),
        ("Solo en TC manual", int(counts.get("SOLO_TC_MANUAL", 0))),
        ("Solo en LLM", int(counts.get("SOLO_LLM", 0))),
        ("TC con ICCS _2025 numerico N1 (1-11)", int(manual["manual_n1_codigo"].notna().sum())),
        ("Comparables N1 en ambos", int(comparables)),
        ("Coinciden N1", int(coincidencias)),
        ("Difieren N1", int(divergencias)),
        ("LLM sin ICCS comparable", int(llm_sin_iccs)),
        ("Manual no comparable", int(counts.get("MANUAL_NO_COMPARABLE", 0))),
        ("Tasa coincidencia N1 sobre comparables", f"{tasa:.2%}"),
    ]
    return pd.DataFrame(rows, columns=["metrica", "valor"])


def compact_top10(top10_path: Path) -> pd.DataFrame:
    if not top10_path.exists():
        return pd.DataFrame()
    cols = [
        "cnp_codigo",
        "cnp_glosa",
        "cnp_descripcion",
        "cnp_familia",
        "rank_rerank",
        "rank",
        "iccs_codigo",
        "iccs_glosa",
        "iccs_descripcion",
        "iccs_inclusiones",
        "iccs_seccion",
        "iccs_seccion_n1",
        "similarity_score",
        "rerank_score",
    ]
    df = pd.read_csv(top10_path, dtype=str)
    keep = [col for col in cols if col in df.columns]
    return df[keep].sort_values(
        ["cnp_codigo", "rank_rerank"],
        key=lambda s: s.map(sort_key) if s.name == "cnp_codigo" else pd.to_numeric(s, errors="coerce"),
    )


def autosize_workbook(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    """Aplica filtros, freeze panes y anchos razonables."""
    for sheet_name in sheet_names:
        ws = writer.book[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for column_cells in ws.columns:
            letter = column_cells[0].column_letter
            header = clean_text(column_cells[0].value)
            width = min(max(len(header) + 2, 12), 48)
            if header.lower().endswith(("descripcion", "justificacion", "exclusiones_aplicadas")):
                width = 48
            ws.column_dimensions[letter].width = width


def write_output(
    output_path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    llm_expanded: pd.DataFrame,
    manual: pd.DataFrame,
    top10: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    divergencias = comparison[comparison["estado_comparacion"].isin(["DIVERGE_N1", "LLM_SIN_ICCS"])]
    coincidencias = comparison[comparison["estado_comparacion"].eq("COINCIDE_N1")]
    no_comparables = comparison[
        comparison["estado_comparacion"].isin(["MANUAL_NO_COMPARABLE", "SOLO_TC_MANUAL", "SOLO_LLM"])
    ]

    sheets: dict[str, pd.DataFrame] = {
        "resumen": summary,
        "comparacion_detallada": comparison,
        "divergencias_n1": divergencias,
        "coincidencias_n1": coincidencias,
        "no_comparables": no_comparables,
        "llm_desagregado": llm_expanded.sort_values("cum", key=lambda s: s.map(sort_key)),
        "tc_manual_2025": manual.sort_values("cum", key=lambda s: s.map(sort_key)),
    }
    if not top10.empty:
        sheets["top10_rerank"] = top10

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize_workbook(writer, list(sheets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_PATH)
    parser.add_argument("--manual-sheet", default=DEFAULT_MANUAL_SHEET)
    parser.add_argument("--llm", type=Path, default=DEFAULT_LLM_PATH)
    parser.add_argument("--iccs-desc", type=Path, default=DEFAULT_ICCS_DESC_PATH)
    parser.add_argument("--iccs-tabla", type=Path, default=DEFAULT_ICCS_TABLA_PATH)
    parser.add_argument("--top10", type=Path, default=DEFAULT_TOP10_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual = read_manual(args.manual, args.manual_sheet)
    llm = read_llm(args.llm)
    meta, code_to_levels = load_iccs_catalog(args.iccs_desc, args.iccs_tabla)

    manual = add_manual_n1_metadata(manual, meta)
    llm_expanded = build_llm_expanded(llm, meta, code_to_levels)
    comparison = build_comparison(manual, llm_expanded)
    summary = build_summary(comparison, manual, llm)
    top10 = compact_top10(args.top10)

    write_output(args.output, summary, comparison, llm_expanded, manual, top10)

    print(f"Archivo generado: {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
