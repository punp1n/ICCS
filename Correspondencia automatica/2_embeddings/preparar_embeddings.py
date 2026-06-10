#!/usr/bin/env python3
"""Embeddings CNP -> ICCS y top-K candidatos (etapa 2 del pipeline).

Modelo de embeddings configurable (Qwen3-Embedding-0.6B por defecto, o E5).
Insumo CNP: tabla de correspondencia 2025 (TC_2025). Los códigos sin glosa,
descripción ni familia (p.ej. CUM 0) se marcan 'no_clasificado' y NO se envían
al modelo. Genera un pool de top-K (50 por defecto) para alimentar el rerank.

Ejemplos:
  python preparar_embeddings.py                 # qwen3, top-50, salida outputs/qwen3/
  python preparar_embeddings.py --model e5      # baseline para el A/B
  python preparar_embeddings.py --k 30 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, __version__ as st_version

import comun


def load_embedder(model_key: str, device: str) -> SentenceTransformer:
    name = comun.EMBED_MODELS[model_key]
    print(f"Cargando modelo de embeddings: {name} (device={device}) ...")
    try:
        return SentenceTransformer(name, device=device)
    except Exception:
        # Algunos modelos requieren código remoto del repo
        return SentenceTransformer(name, device=device, trust_remote_code=True)


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def top_k(cnp_emb: np.ndarray, iccs_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k por similitud coseno (embeddings ya normalizados)."""
    sims = cnp_emb @ iccs_emb.T  # (n_cnp, n_iccs)
    k = min(k, iccs_emb.shape[0])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    # ordenar cada fila por score desc
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    scores = np.take_along_axis(sims, idx, axis=1)
    return idx, scores


def build_detailed(cnp: pd.DataFrame, iccs: pd.DataFrame,
                   idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(cnp)):
        c = cnp.iloc[i]
        for rank in range(idx.shape[1]):
            j = int(idx[i, rank])
            ic = iccs.iloc[j]
            rows.append({
                "cnp_codigo": c["codigo"],
                "cnp_glosa": c["glosa"],
                "cnp_descripcion": c["descripcion"],
                "cnp_familia": c["familia_nombre"],
                "cnp_texto": c["texto"],
                "rank": rank + 1,
                "similarity_score": float(scores[i, rank]),
                "iccs_codigo": ic["codigo_iccs"],
                "iccs_glosa": ic["glosa_iccs"],
                "iccs_descripcion": ic["descripcion"],
                "iccs_inclusiones": ic["inclusiones"],
                "iccs_seccion": ic["seccion"],
                "iccs_seccion_n1": ic.get("seccion_n1"),
                "iccs_texto": ic["texto"],
            })
    return pd.DataFrame(rows)


def build_compact(detailed: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Una fila por CNP con top-N en columnas (top1..topN)."""
    rows = []
    for codigo, grp in detailed.groupby("cnp_codigo", sort=False):
        grp = grp.sort_values("rank").head(top_n)
        row = {"cnp_codigo": codigo, "cnp_glosa": grp.iloc[0]["cnp_glosa"]}
        for r, (_, m) in enumerate(grp.iterrows(), start=1):
            row[f"top{r}_codigo"] = m["iccs_codigo"]
            row[f"top{r}_score"] = round(float(m["similarity_score"]), 4)
            row[f"top{r}_glosa"] = m["iccs_glosa"]
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(comun.EMBED_MODELS), default="qwen3",
                   help="Modelo de embeddings (default: qwen3).")
    p.add_argument("--k", type=int, default=50,
                   help="Tamaño del pool de candidatos para rerank (default: 50).")
    p.add_argument("--top-final", type=int, default=10,
                   help="Top-N en el reporte compacto de embeddings (default: 10).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None, help="cpu/cuda (default: autodetecta).")
    p.add_argument("--cnp-source", default=str(comun.CNP_TC2025_PATH))
    p.add_argument("--output-dir", default=None,
                   help="Default: outputs/<model>/")
    p.add_argument("--save-embeddings", action="store_true",
                   help="Guarda los vectores en parquet (pesado).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else comun.OUTPUT_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando insumos...")
    cnp_all = comun.load_cnp(Path(args.cnp_source))
    iccs = comun.load_iccs()
    matchable = cnp_all[cnp_all["estado"] == "matchable"].reset_index(drop=True)
    no_clasif = cnp_all[cnp_all["estado"] == "no_clasificado"].reset_index(drop=True)
    print(f"  CNP total: {len(cnp_all)} | matchable: {len(matchable)} | "
          f"no_clasificado: {len(no_clasif)}")
    print(f"  ICCS: {len(iccs)} códigos")

    cnp_all.to_csv(out_dir / "cnp_preparado.csv", index=False, encoding="utf-8-sig")
    iccs.drop(columns=["texto_hash"]).to_csv(out_dir / "iccs_preparado.csv",
                                             index=False, encoding="utf-8-sig")
    if len(no_clasif):
        no_clasif.to_csv(out_dir / "cnp_no_clasificado.csv", index=False, encoding="utf-8-sig")

    model = load_embedder(args.model, device)
    print("Embeddings CNP (queries)...")
    q_texts = comun.build_query_texts(matchable["texto"].tolist(), args.model)
    cnp_emb = encode(model, q_texts, args.batch_size)
    print("Embeddings ICCS (passages)...")
    d_texts = comun.build_doc_texts(iccs["texto"].tolist(), args.model)
    iccs_emb = encode(model, d_texts, args.batch_size)

    if args.save_embeddings:
        np.save(out_dir / "cnp_embeddings.npy", cnp_emb)
        np.save(out_dir / "iccs_embeddings.npy", iccs_emb)

    print(f"Top-{args.k} por coseno...")
    idx, scores = top_k(cnp_emb, iccs_emb, args.k)
    detailed = build_detailed(matchable, iccs, idx, scores)
    detailed.to_csv(out_dir / "matches_detallado.csv", index=False, encoding="utf-8-sig")
    compact = build_compact(detailed, args.top_final)
    compact.to_csv(out_dir / "matches_compacto.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "modelo_embeddings": comun.EMBED_MODELS[args.model],
        "model_key": args.model,
        "dispositivo": device,
        "k_pool": args.k,
        "top_final": args.top_final,
        "batch_size": args.batch_size,
        "embeddings_dim": int(cnp_emb.shape[1]),
        "sentence_transformers": st_version,
        "torch": torch.__version__,
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "filas": {"cnp_total": len(cnp_all), "cnp_matchable": len(matchable),
                  "cnp_no_clasificado": len(no_clasif), "iccs": len(iccs)},
        "insumo_cnp": str(args.cnp_source),
    }
    (out_dir / "metadata_embeddings.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nListo ({args.model}). Salidas en {out_dir}")
    print(f"  similitud top-1: media={scores[:, 0].mean():.4f} "
          f"min={scores[:, 0].min():.4f} max={scores[:, 0].max():.4f}")


if __name__ == "__main__":
    main()
