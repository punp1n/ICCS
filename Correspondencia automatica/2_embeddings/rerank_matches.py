#!/usr/bin/env python3
"""Rerank del top-K de embeddings con un cross-encoder (etapa 2.b).

Toma `matches_detallado.csv` (pool de candidatos por CNP, generado por
preparar_embeddings.py) y reordena los mejores candidatos con
BAAI/bge-reranker-v2-m3, que evalúa cada par (delito CNP, candidato ICCS)
de forma conjunta. Produce el top-N final reordenado.

Ejemplos:
  python rerank_matches.py --model qwen3                 # rerank salida qwen3
  python rerank_matches.py --model e5 --rerank-pool 25
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder

import comun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(comun.EMBED_MODELS), default="qwen3",
                   help="Qué salida de embeddings reordenar (default: qwen3).")
    p.add_argument("--reranker", default=comun.RERANKER_DEFAULT,
                   help=f"Modelo cross-encoder (default: {comun.RERANKER_DEFAULT}).")
    p.add_argument("--input-dir", default=None, help="Default: outputs/<model>/")
    p.add_argument("--rerank-pool", type=int, default=25,
                   help="Candidatos por CNP a reordenar (default: 25).")
    p.add_argument("--top-final", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir) if args.input_dir else comun.OUTPUT_DIR / args.model
    detailed_path = in_dir / "matches_detallado.csv"
    if not detailed_path.exists():
        raise SystemExit(f"No existe {detailed_path}. Corre primero preparar_embeddings.py --model {args.model}")

    det = pd.read_csv(detailed_path)
    # Limitar al pool a reordenar (los mejores por embedding)
    det = det[det["rank"] <= args.rerank_pool].copy()

    print(f"Cargando reranker {args.reranker} ...")
    ce = CrossEncoder(args.reranker, max_length=args.max_length,
                      device=args.device)  # device None -> autodetecta

    pairs = list(zip(det["cnp_texto"].fillna("").astype(str),
                     det["iccs_texto"].fillna("").astype(str)))
    print(f"Reordenando {len(pairs)} pares (CNP x candidatos)...")
    scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    det["rerank_score"] = scores

    # Reordenar por rerank dentro de cada CNP y quedarse con top-N
    det = det.sort_values(["cnp_codigo", "rerank_score"], ascending=[True, False])
    det["rank_rerank"] = det.groupby("cnp_codigo").cumcount() + 1
    top = det[det["rank_rerank"] <= args.top_final].copy()

    top.to_csv(in_dir / "matches_rerank_detallado.csv", index=False, encoding="utf-8-sig")

    # Compacto top1..topN
    rows = []
    for codigo, grp in top.groupby("cnp_codigo", sort=False):
        grp = grp.sort_values("rank_rerank")
        row = {"cnp_codigo": codigo, "cnp_glosa": grp.iloc[0]["cnp_glosa"]}
        for r, (_, m) in enumerate(grp.iterrows(), start=1):
            row[f"top{r}_codigo"] = m["iccs_codigo"]
            row[f"top{r}_score"] = round(float(m["rerank_score"]), 4)
            row[f"top{r}_glosa"] = m["iccs_glosa"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(in_dir / "matches_rerank_compacto.csv",
                              index=False, encoding="utf-8-sig")

    meta = {
        "reranker": args.reranker,
        "embeddings_model": comun.EMBED_MODELS[args.model],
        "rerank_pool": args.rerank_pool,
        "top_final": args.top_final,
        "max_length": args.max_length,
        "pares_evaluados": len(pairs),
        "generado_utc": datetime.now(timezone.utc).isoformat(),
    }
    (in_dir / "metadata_rerank.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Listo. Rerank en {in_dir}/matches_rerank_*.csv")


if __name__ == "__main__":
    main()
