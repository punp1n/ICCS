#!/usr/bin/env python3
"""Genera una version enriquecida de la TC manual 2025 con la salida del LLM.

Toma la TC manual (09062026_TC_2025_v1.0.xlsx, hoja TC_2025_v1.0), conserva
TODAS sus columnas originales y agrega:
  - ICCS_LLM_especifico      : codigo ICCS elegido por el LLM (nivel mas fino)
  - ICCS_LLM_especifico_glosa: glosa del codigo especifico
  - ICCS_LLM_N1              : codigo ICCS al primer nivel (seccion 1-11)
  - ICCS_LLM_N1_glosa        : glosa del primer nivel
  - LLM_justificacion        : justificacion del LLM para la clasificacion
  - LLM_exclusiones_aplicadas: exclusiones que el LLM aplico
  - coincide_N1_manual_vs_llm: bool, comparacion manual vs automatica a N1

El libro de salida incluye ademas las hojas "top10_rerank" y "resumen",
reutilizando la logica ya validada de comparar_llm_vs_tc_2025.py.

Salida por defecto:
    Comparativa corrrespondencias/TC_2025_v1.0_con_LLM.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import comparar_llm_vs_tc_2025 as base


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "TC_2025_v1.0_con_LLM.xlsx"

# Columnas auxiliares que read_manual/add_manual_n1_metadata agregan y que no
# van en la hoja principal (se conservan solo las columnas originales de la TC).
HELPER_MANUAL_COLS = (
    "cum",
    "manual_iccs_2025_raw",
    "manual_n1_codigo",
    "manual_n1_glosa",
)

# Nuevas columnas del LLM, en orden de salida.
NEW_LLM_COLS = (
    "ICCS_LLM_especifico",
    "ICCS_LLM_especifico_glosa",
    "ICCS_LLM_N1",
    "ICCS_LLM_N1_glosa",
    "LLM_justificacion",
    "LLM_exclusiones_aplicadas",
    "coincide_N1_manual_vs_llm",
)


def build_enriched_manual(manual: pd.DataFrame, llm_expanded: pd.DataFrame) -> pd.DataFrame:
    """TC manual original + columnas del LLM, conservando todas las columnas."""
    orig_cols = [c for c in manual.columns if c not in HELPER_MANUAL_COLS]

    llm_slice = pd.DataFrame({
        "cum": llm_expanded["cum"],
        "ICCS_LLM_especifico": llm_expanded["llm_iccs_elegido_norm"],
        "ICCS_LLM_especifico_glosa": llm_expanded.get("iccs_glosa_elegida", ""),
        "ICCS_LLM_N1": llm_expanded["llm_n1_codigo"],
        "ICCS_LLM_N1_glosa": llm_expanded.get("llm_n1_glosa", ""),
        "LLM_justificacion": llm_expanded.get("justificacion", ""),
        "LLM_exclusiones_aplicadas": llm_expanded.get("exclusiones_aplicadas", ""),
        "_llm_n1_num": llm_expanded["llm_n1_codigo_num"],
    })

    merged = manual.merge(llm_slice, on="cum", how="left")

    manual_n1 = pd.to_numeric(merged["manual_n1_codigo"], errors="coerce")
    llm_n1 = pd.to_numeric(merged["_llm_n1_num"], errors="coerce")
    merged["coincide_N1_manual_vs_llm"] = (
        manual_n1.notna() & llm_n1.notna() & (manual_n1 == llm_n1)
    )

    out = merged[orig_cols + list(NEW_LLM_COLS)]
    return out.sort_values(orig_cols[0], key=lambda s: s.map(base.sort_key)) \
        if orig_cols and orig_cols[0].upper() == "CUM" \
        else out


def write_output(
    output_path: Path,
    enriched: pd.DataFrame,
    top10: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, pd.DataFrame] = {base.DEFAULT_MANUAL_SHEET: enriched}
    if not top10.empty:
        sheets["top10_rerank"] = top10
    sheets["resumen"] = summary

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        base.autosize_workbook(writer, list(sheets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, default=base.DEFAULT_MANUAL_PATH)
    parser.add_argument("--manual-sheet", default=base.DEFAULT_MANUAL_SHEET)
    parser.add_argument("--llm", type=Path, default=base.DEFAULT_LLM_PATH)
    parser.add_argument("--iccs-desc", type=Path, default=base.DEFAULT_ICCS_DESC_PATH)
    parser.add_argument("--iccs-tabla", type=Path, default=base.DEFAULT_ICCS_TABLA_PATH)
    parser.add_argument("--top10", type=Path, default=base.DEFAULT_TOP10_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual = base.read_manual(args.manual, args.manual_sheet)
    llm = base.read_llm(args.llm)
    meta, code_to_levels = base.load_iccs_catalog(args.iccs_desc, args.iccs_tabla)

    manual = base.add_manual_n1_metadata(manual, meta)
    llm_expanded = base.build_llm_expanded(llm, meta, code_to_levels)

    enriched = build_enriched_manual(manual, llm_expanded)
    comparison = base.build_comparison(manual, llm_expanded)
    summary = base.build_summary(comparison, manual, llm)
    top10 = base.compact_top10(args.top10)

    write_output(args.output, enriched, top10, summary)

    coincide = int(enriched["coincide_N1_manual_vs_llm"].sum())
    print(f"Archivo generado: {args.output}")
    print(f"Filas TC: {len(enriched)} | coincide_N1=True: {coincide}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
