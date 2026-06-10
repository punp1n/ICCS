#!/usr/bin/env python3
"""Evaluación A/B de configuraciones de candidatos vs etiqueta manual ICCS_2025.

La etiqueta manual `ICCS _2025` del insumo TC_2025 está a nivel SECCIÓN (N1,
1-11). Por eso la evaluación mide acierto a nivel sección: para cada CNP con
etiqueta numérica válida, se revisa si la sección del/los código(s) ICCS
predicho(s) coincide con la sección manual.

Configuraciones comparadas (las que existan en outputs/<model>/):
  - <model> embeddings   (matches_compacto.csv)
  - <model> + rerank     (matches_rerank_compacto.csv)
para model en {qwen3, e5}.

Métricas: top1, top3, recall@10 (acierto de sección).

Salida: outputs/ab_report.csv y outputs/ab_report.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import comun


def _pred_codes(row: pd.Series, top_n: int) -> list[str]:
    codes = []
    for r in range(1, top_n + 1):
        col = f"top{r}_codigo"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            codes.append(str(row[col]).strip())
    return codes


def evaluate(compact_path: Path, gold: dict[str, int],
             code2n1: dict[str, int]) -> dict | None:
    if not compact_path.exists():
        return None
    df = pd.read_csv(compact_path)
    df["cnp_codigo"] = df["cnp_codigo"].apply(comun._fmt_cum)
    n = top1 = top3 = rec10 = 0
    for _, row in df.iterrows():
        manual = gold.get(str(row["cnp_codigo"]))
        if manual is None:
            continue
        codes = _pred_codes(row, 10)
        secs = [code2n1.get(c) for c in codes]
        secs = [s for s in secs if s is not None]
        if not secs:
            continue
        n += 1
        if secs[0] == manual:
            top1 += 1
        if manual in secs[:3]:
            top3 += 1
        if manual in secs[:10]:
            rec10 += 1
    if n == 0:
        return None
    return {"n_eval": n,
            "top1": round(top1 / n, 4),
            "top3": round(top3 / n, 4),
            "recall@10": round(rec10 / n, 4)}


def main() -> None:
    cnp = comun.load_cnp()
    gold = {}
    for _, r in cnp.iterrows():
        n1 = comun.parse_manual_n1(r["iccs_2025_manual"])
        if n1 is not None:
            gold[str(r["codigo"])] = n1
    print(f"CNP con etiqueta manual de sección válida (1-11): {len(gold)}")

    iccs = comun.load_iccs()
    code2n1 = comun.code_to_n1_map(iccs)

    configs = []
    for model in ("qwen3", "e5"):
        d = comun.OUTPUT_DIR / model
        configs.append((f"{model} (embeddings)", d / "matches_compacto.csv"))
        configs.append((f"{model} + rerank", d / "matches_rerank_compacto.csv"))

    rows = []
    for nombre, path in configs:
        res = evaluate(path, gold, code2n1)
        if res:
            rows.append({"config": nombre, **res})
            print(f"  {nombre:24s} n={res['n_eval']:4d}  "
                  f"top1={res['top1']:.3f}  top3={res['top3']:.3f}  recall@10={res['recall@10']:.3f}")
        else:
            print(f"  {nombre:24s} (sin datos)")

    if not rows:
        print("No hay configuraciones para evaluar todavía.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(comun.OUTPUT_DIR / "ab_report.csv", index=False, encoding="utf-8-sig")

    md = ["# Evaluación A/B — acierto de sección ICCS (N1) vs etiqueta manual ICCS_2025",
          "",
          f"CNP evaluables (etiqueta 1-11): **{len(gold)}**. "
          "Métrica: ¿la sección del código ICCS predicho coincide con la manual?",
          "",
          "| Configuración | n | top-1 | top-3 | recall@10 |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['n_eval']} | {r['top1']:.3f} | "
                  f"{r['top3']:.3f} | {r['recall@10']:.3f} |")
    (comun.OUTPUT_DIR / "ab_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nReporte: {comun.OUTPUT_DIR / 'ab_report.md'}")


if __name__ == "__main__":
    main()
