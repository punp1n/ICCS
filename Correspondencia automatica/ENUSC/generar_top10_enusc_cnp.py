#!/usr/bin/env python3
"""Genera top-k CUM para glosas ENUSC usando embeddings E5."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

MODEL_NAME = "intfloat/multilingual-e5-large"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 10

REPO_ROOT = Path(__file__).resolve().parents[2]
CNP_PATH = REPO_ROOT / "CNP" / "consolidado_CNP_2025_2021.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

ENUSC_GLOSAS = [
    "Rayados, marcas o pinturas en la propiedad pública o privada sin autorización",
    "Presencia de personas que habitan y/o duermen en la calle",
    "Presencia de comercio ilegal",
    "Consumo de alcohol o droga en la vía pública",
    "Venta clandestina de alcohol",
    "Sitios eriazos descuidados o acumulación de basura",
    "Prostitución o comercio sexual",
    "Lanzamiento de fuegos artificiales",
    "Vandalismo o daño a la propiedad pública o privada, excluyendo rayados o marcas",
    "Amenazas o peleas entre vecinos",
    "Presencia de pandillas violentas",
    "Peleas callejeras con armas blancas o de fuego",
    "Peleas callejeras sin armas",
    "Robos o asaltos en la vía pública",
    "Balaceras o disparos",
]


def normalize_text(parts: Iterable[str]) -> str:
    """Concatena campos de texto removiendo blancos redundantes."""
    cleaned = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text and text.lower() != "nan":
            cleaned.append(text)
    return " ".join(" | ".join(cleaned).split())


def hash_series(series: pd.Series) -> str:
    """Retorna un hash reproducible del contenido de una serie."""
    digest = hashlib.sha256()
    for value in series.astype(str):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def resolve_device(explicit_device: str | None) -> str:
    """Selecciona GPU si está disponible, salvo que se indique otro dispositivo."""
    if explicit_device:
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_enusc() -> pd.DataFrame:
    """Construye las consultas ENUSC con prefijo requerido por E5."""
    rows = []
    for index, glosa in enumerate(ENUSC_GLOSAS, start=1):
        rows.append(
            {
                "enusc_id": index,
                "enusc_glosa": glosa,
                "texto_embedding": f"query: {normalize_text([glosa])}",
            }
        )
    return pd.DataFrame(rows)


def prepare_cnp(cnp_raw: pd.DataFrame) -> pd.DataFrame:
    """Construye pasajes CNP/CUM usando glosa y descripción."""
    cnp = cnp_raw.copy()
    for col in ("codigo", "glosa", "descripcion", "familia_nombre", "ultimo_vigente"):
        if col not in cnp.columns:
            cnp[col] = ""
        cnp[col] = cnp[col].fillna("")

    cnp["codigo_cum"] = cnp["codigo"].astype(str)
    cnp["cum_glosa"] = cnp["glosa"].astype(str).str.strip()
    cnp["cum_descripcion"] = cnp["descripcion"].astype(str).str.strip()
    cnp["texto_embedding"] = cnp.apply(
        lambda row: f"passage: {normalize_text([row['cum_glosa'], row['cum_descripcion']])}",
        axis=1,
    )
    return cnp


class E5TransformerBackend:
    """Backend mínimo para embeddings E5 usando transformers."""

    def __init__(self, model_name: str, device: str) -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Calcula embeddings normalizados por lotes."""
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._average_pool(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu().numpy())
                print(f"  Procesados {min(start + batch_size, len(texts))}/{len(texts)} textos")
        return np.vstack(embeddings)


def encode_texts(model: E5TransformerBackend, texts: list[str], batch_size: int) -> np.ndarray:
    """Calcula embeddings normalizados."""
    return model.encode(texts, batch_size=batch_size)


def compute_top_k(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula top-k por similitud coseno."""
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    passage_norm = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
    similarity_matrix = query_norm @ passage_norm.T
    top_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
    top_scores = np.take_along_axis(similarity_matrix, top_indices, axis=1)
    return top_indices, top_scores


def build_outputs(
    enusc_df: pd.DataFrame,
    cnp_df: pd.DataFrame,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea salidas detallada y compacta para revisión."""
    detailed_rows = []
    compact_rows = []
    k = top_indices.shape[1]

    for enusc_idx, enusc_row in enusc_df.iterrows():
        compact = {
            "enusc_id": enusc_row["enusc_id"],
            "enusc_glosa": enusc_row["enusc_glosa"],
        }

        for rank in range(k):
            cnp_idx = int(top_indices[enusc_idx, rank])
            cnp_row = cnp_df.iloc[cnp_idx]
            score = float(top_scores[enusc_idx, rank])

            detailed_rows.append(
                {
                    "enusc_id": enusc_row["enusc_id"],
                    "enusc_glosa": enusc_row["enusc_glosa"],
                    "rank": rank + 1,
                    "similarity_score": score,
                    "codigo_cum": cnp_row["codigo_cum"],
                    "cum_glosa": cnp_row["cum_glosa"],
                    "cum_descripcion": cnp_row["cum_descripcion"],
                    "cum_familia": cnp_row.get("familia_nombre", ""),
                    "cum_ultimo_vigente": cnp_row.get("ultimo_vigente", ""),
                }
            )

            prefix = f"top{rank + 1}"
            compact[f"{prefix}_codigo_cum"] = cnp_row["codigo_cum"]
            compact[f"{prefix}_score"] = round(score, 4)
            compact[f"{prefix}_glosa_cum"] = cnp_row["cum_glosa"]

        compact_rows.append(compact)

    return pd.DataFrame(detailed_rows), pd.DataFrame(compact_rows)


def save_outputs(
    detailed_df: pd.DataFrame,
    compact_df: pd.DataFrame,
    enusc_df: pd.DataFrame,
    cnp_df: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> Path:
    """Guarda Excel, CSV y metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = output_dir / "top10_enusc_cnp.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        detailed_df.to_excel(writer, sheet_name="top10_detallado", index=False)
        compact_df.to_excel(writer, sheet_name="top10_compacto", index=False)
        enusc_df[["enusc_id", "enusc_glosa", "texto_embedding"]].to_excel(
            writer,
            sheet_name="glosas_enusc",
            index=False,
        )
        cnp_df[
            [
                "codigo_cum",
                "cum_glosa",
                "cum_descripcion",
                "familia_nombre",
                "ultimo_vigente",
                "texto_embedding",
            ]
        ].to_excel(writer, sheet_name="cnp_preparado", index=False)

    detailed_df.to_csv(output_dir / "top10_enusc_cnp_detallado.csv", index=False, encoding="utf-8-sig")
    compact_df.to_csv(output_dir / "top10_enusc_cnp_compacto.csv", index=False, encoding="utf-8-sig")
    (output_dir / "metadata_top10_enusc_cnp.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return xlsx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k CUM para glosas ENUSC mediante embeddings.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Cantidad de candidatos CUM por glosa.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Tamaño de batch.")
    parser.add_argument("--device", type=str, default=None, help="Forzar dispositivo: cuda o cpu.")
    parser.add_argument("--cnp-path", type=Path, default=CNP_PATH, help="Ruta al consolidado CNP parquet/xlsx/csv.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Carpeta de salidas.")
    return parser.parse_args()


def read_cnp(path: Path) -> pd.DataFrame:
    """Lee CNP desde parquet, xlsx o csv."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo CNP: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato CNP no soportado: {path.suffix}")


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k debe ser mayor o igual a 1")

    print(f"Cargando CNP desde {args.cnp_path}...")
    cnp_raw = read_cnp(args.cnp_path)
    enusc_df = prepare_enusc()
    cnp_df = prepare_cnp(cnp_raw)

    print(f"Glosas ENUSC: {len(enusc_df)}")
    print(f"Códigos CUM/CNP: {len(cnp_df)}")

    device = resolve_device(args.device)
    print(f"Cargando modelo {MODEL_NAME} en {device}...")
    model = E5TransformerBackend(MODEL_NAME, device=device)

    print("Calculando embeddings ENUSC...")
    enusc_embeddings = encode_texts(model, enusc_df["texto_embedding"].tolist(), args.batch_size)
    print("Calculando embeddings CNP/CUM...")
    cnp_embeddings = encode_texts(model, cnp_df["texto_embedding"].tolist(), args.batch_size)

    print(f"Calculando top-{args.k} CUM por glosa ENUSC...")
    top_indices, top_scores = compute_top_k(enusc_embeddings, cnp_embeddings, args.k)
    detailed_df, compact_df = build_outputs(enusc_df, cnp_df, top_indices, top_scores)

    metadata = {
        "modelo": MODEL_NAME,
        "backend": "transformers",
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "dispositivo": device,
        "batch_size": args.batch_size,
        "k": args.k,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "origen": {"cnp_path": str(args.cnp_path)},
        "filas": {"enusc": len(enusc_df), "cnp": len(cnp_df), "matches": len(detailed_df)},
        "hashes": {
            "enusc_texto_embedding": hash_series(enusc_df["texto_embedding"]),
            "cnp_texto_embedding": hash_series(cnp_df["texto_embedding"]),
        },
        "texto_embedding": {
            "enusc": "query: glosa_enusc",
            "cnp": "passage: glosa_cum | descripcion_cum",
        },
        "sin_llm": True,
    }
    xlsx_path = save_outputs(detailed_df, compact_df, enusc_df, cnp_df, args.output_dir, metadata)

    print("Salida generada:")
    print(f"  Excel: {xlsx_path}")
    print(f"  CSV detallado: {args.output_dir / 'top10_enusc_cnp_detallado.csv'}")
    print(f"  CSV compacto: {args.output_dir / 'top10_enusc_cnp_compacto.csv'}")
    print(f"Score top-1 promedio: {top_scores[:, 0].mean():.4f}")


if __name__ == "__main__":
    main()
