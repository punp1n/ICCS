#!/usr/bin/env python3
"""Pipeline de preparación y embeddings para CNP -> ICCS."""

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
from sentence_transformers import SentenceTransformer, __version__ as st_version

MODEL_NAME = "intfloat/multilingual-e5-large"
DEFAULT_BATCH_SIZE = 16
NORMALIZE_EMBEDDINGS = True

# Pesos por defecto para embeddings multi-campo
# CNP: Glosa (50%), Descripción (35%), Familia (15%)
DEFAULT_CNP_WEIGHTS = [0.50, 0.35, 0.15]
# ICCS: Glosa (40%), Descripción (35%), Inclusiones (20%), Sección (5%)
DEFAULT_ICCS_WEIGHTS = [0.40, 0.35, 0.20, 0.05]

REPO_ROOT = Path(__file__).resolve().parents[2]
CNP_PATH = REPO_ROOT / "CNP" / "consolidado_CNP_2025_2021.xlsx"
ICCS_PATH = REPO_ROOT / "Correspondencia automatica" / "outputs" / "iccs_descripcion.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


def normalize_text(parts: Iterable[str]) -> str:
    """Limpia y concatena textos removiendo espacios extra."""
    cleaned_parts = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            cleaned_parts.append(text)
    joined = " | ".join(cleaned_parts)
    return " ".join(joined.split())


def hash_text(value: str) -> str:
    """Hash SHA256 para rastrear versiones de textos."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_series(series: pd.Series) -> str:
    """Hash SHA256 acumulado de una serie para reproducibilidad."""
    digest = hashlib.sha256()
    for value in series.astype(str):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def prepare_cnp(df: pd.DataFrame, multi_field: bool = False) -> pd.DataFrame:
    """Construye el texto de consulta con prefijo query: para E5.

    Args:
        df: DataFrame con datos CNP
        multi_field: Si True, genera campos separados para embeddings multi-campo
    """
    cnp = df.copy()
    for col in ("glosa", "descripcion", "familia_nombre", "articulado"):
        if col not in cnp.columns:
            cnp[col] = ""
        cnp[col] = cnp[col].fillna("")
    cnp["codigo"] = cnp["codigo"].astype(str)
    cnp["articulado"] = cnp["articulado"].astype(str)

    if multi_field:
        # Campos separados para embeddings con pesos
        cnp["campo_1_glosa"] = cnp["glosa"].apply(lambda x: f"query: {x.strip()}" if x.strip() else "query: ")
        cnp["campo_2_descripcion"] = cnp["descripcion"].apply(lambda x: f"query: {x.strip()}" if x.strip() else "query: ")
        cnp["campo_3_familia"] = cnp["familia_nombre"].apply(lambda x: f"query: {x.strip()}" if x.strip() else "query: ")

        # Texto concatenado para backward compatibility y hash
        def _build_text(row: pd.Series) -> str:
            return f"query: {normalize_text([row['glosa'], row['descripcion'], row['familia_nombre']])}"
        cnp["texto_embedding"] = cnp.apply(_build_text, axis=1)
    else:
        # Método original: todo concatenado
        def _build_text(row: pd.Series) -> str:
            return f"query: {normalize_text([row['glosa'], row['descripcion'], row['familia_nombre']])}"
        cnp["texto_embedding"] = cnp.apply(_build_text, axis=1)

    cnp["texto_hash"] = cnp["texto_embedding"].apply(hash_text)
    return cnp


def prepare_iccs(df: pd.DataFrame, multi_field: bool = False) -> pd.DataFrame:
    """Construye el texto de pasaje con prefijo passage: para E5.

    Args:
        df: DataFrame con datos ICCS
        multi_field: Si True, genera campos separados para embeddings multi-campo
    """
    iccs = df.copy()
    for col in ("glosa_iccs", "descripcion", "inclusiones", "seccion"):
        if col not in iccs.columns:
            iccs[col] = ""
        iccs[col] = iccs[col].fillna("")
    iccs["codigo_iccs"] = iccs["codigo_iccs"].astype(str)

    if multi_field:
        # Campos separados para embeddings con pesos
        iccs["campo_1_glosa"] = iccs["glosa_iccs"].apply(lambda x: f"passage: {x.strip()}" if x.strip() else "passage: ")
        iccs["campo_2_descripcion"] = iccs["descripcion"].apply(lambda x: f"passage: {x.strip()}" if x.strip() else "passage: ")
        iccs["campo_3_inclusiones"] = iccs["inclusiones"].apply(lambda x: f"passage: {x.strip()}" if x.strip() else "passage: ")
        iccs["campo_4_seccion"] = iccs["seccion"].apply(
            lambda x: f"passage: Seccion {x.strip()}" if x.strip() else "passage: "
        )

        # Texto concatenado para backward compatibility y hash
        def _build_text(row: pd.Series) -> str:
            parts = [row["glosa_iccs"], row["descripcion"], row["inclusiones"]]
            if row["seccion"]:
                parts.append(f"Seccion {row['seccion']}")
            return f"passage: {normalize_text(parts)}"
        iccs["texto_embedding"] = iccs.apply(_build_text, axis=1)
    else:
        # Método original: todo concatenado
        def _build_text(row: pd.Series) -> str:
            parts = [row["glosa_iccs"], row["descripcion"], row["inclusiones"]]
            if row["seccion"]:
                parts.append(f"Seccion {row['seccion']}")
            return f"passage: {normalize_text(parts)}"
        iccs["texto_embedding"] = iccs.apply(_build_text, axis=1)

    iccs["texto_hash"] = iccs["texto_embedding"].apply(hash_text)
    return iccs


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> list[list[float]]:
    """Calcula embeddings con batching y normalización opcional."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def encode_multi_field(
    model: SentenceTransformer,
    df: pd.DataFrame,
    field_columns: list[str],
    weights: list[float],
    batch_size: int,
) -> np.ndarray:
    """
    Calcula embeddings multi-campo con pesos.

    Args:
        model: Modelo de sentence transformers
        df: DataFrame con las columnas de campos
        field_columns: Lista de nombres de columnas con los textos por campo
        weights: Lista de pesos para cada campo (deben sumar 1.0)
        batch_size: Tamaño de batch para encoding

    Returns:
        Array (n_samples, embedding_dim) con embeddings combinados y normalizados
    """
    # Validar pesos
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Los pesos deben sumar 1.0, suma actual: {weights.sum()}")

    if len(field_columns) != len(weights):
        raise ValueError(f"Número de campos ({len(field_columns)}) no coincide con número de pesos ({len(weights)})")

    # Calcular embedding por cada campo
    field_embeddings = []
    for col in field_columns:
        print(f"  Calculando embeddings para {col}...")
        texts = df[col].tolist()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # Normalizar cada campo individualmente
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        field_embeddings.append(embeddings)

    # Combinar con pesos
    combined = np.zeros_like(field_embeddings[0])
    for emb, weight in zip(field_embeddings, weights):
        combined += weight * emb

    # Normalizar el resultado final
    combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

    return combined


def save_table(df: pd.DataFrame, name: str, output_dir: Path, save_csv: bool = True) -> dict[str, Path]:
    """Guarda un DataFrame en parquet y opcionalmente en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)
    paths = {"parquet": parquet_path}
    if save_csv:
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path
    return paths


def build_metadata(
    cnp_df: pd.DataFrame,
    iccs_df: pd.DataFrame,
    device: str,
    batch_size: int,
    embeddings_dim: int | None,
    multi_field: bool = False,
    cnp_weights: list[float] | None = None,
    iccs_weights: list[float] | None = None,
) -> dict:
    """Metadata de trazabilidad (modelo, hashes, fechas)."""
    metadata = {
        "modelo": MODEL_NAME,
        "sentence_transformers_version": st_version,
        "torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "dispositivo": device,
        "batch_size": batch_size,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "embeddings_dim": embeddings_dim,
        "multi_field": multi_field,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {
            "cnp_texto_embedding": hash_series(cnp_df["texto_embedding"]),
            "iccs_texto_embedding": hash_series(iccs_df["texto_embedding"]),
        },
        "filas": {
            "cnp": len(cnp_df),
            "iccs": len(iccs_df),
        },
        "origen": {
            "cnp_path": str(CNP_PATH),
            "iccs_path": str(ICCS_PATH),
        },
    }

    if multi_field:
        metadata["pesos"] = {
            "cnp": {
                "glosa": cnp_weights[0] if cnp_weights else DEFAULT_CNP_WEIGHTS[0],
                "descripcion": cnp_weights[1] if cnp_weights else DEFAULT_CNP_WEIGHTS[1],
                "familia": cnp_weights[2] if cnp_weights else DEFAULT_CNP_WEIGHTS[2],
            },
            "iccs": {
                "glosa": iccs_weights[0] if iccs_weights else DEFAULT_ICCS_WEIGHTS[0],
                "descripcion": iccs_weights[1] if iccs_weights else DEFAULT_ICCS_WEIGHTS[1],
                "inclusiones": iccs_weights[2] if iccs_weights else DEFAULT_ICCS_WEIGHTS[2],
                "seccion": iccs_weights[3] if iccs_weights else DEFAULT_ICCS_WEIGHTS[3],
            },
        }

    return metadata


def resolve_device(explicit_device: str | None) -> str:
    """Selecciona dispositivo priorizando GPU disponible."""
    if explicit_device:
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_top_k_matches(
    cnp_embeddings: np.ndarray,
    iccs_embeddings: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula los k vecinos más cercanos para cada embedding CNP contra ICCS.

    Returns:
        indices: array (n_cnp, k) con índices de los top-k ICCS
        similarities: array (n_cnp, k) con las similitudes coseno correspondientes
    """
    # Normalizar embeddings para usar producto punto como similitud coseno
    cnp_norm = cnp_embeddings / np.linalg.norm(cnp_embeddings, axis=1, keepdims=True)
    iccs_norm = iccs_embeddings / np.linalg.norm(iccs_embeddings, axis=1, keepdims=True)

    # Calcular matriz de similitudes (producto punto = coseno si está normalizado)
    similarities_matrix = cnp_norm @ iccs_norm.T  # (n_cnp, n_iccs)

    # Obtener top-k índices y scores
    top_k_indices = np.argsort(-similarities_matrix, axis=1)[:, :k]
    top_k_scores = np.take_along_axis(similarities_matrix, top_k_indices, axis=1)

    return top_k_indices, top_k_scores


def generate_matches_report(
    cnp_df: pd.DataFrame,
    iccs_df: pd.DataFrame,
    top_k_indices: np.ndarray,
    top_k_scores: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Genera reportes detallados y compactos de los k vecinos más cercanos.

    Returns:
        Tupla con paths a (reporte_detallado, reporte_compacto)
    """
    k = top_k_indices.shape[1]

    # Reporte detallado: una fila por cada match (CNP x k filas)
    detailed_rows = []
    for cnp_idx in range(len(cnp_df)):
        cnp_row = cnp_df.iloc[cnp_idx]
        for rank in range(k):
            iccs_idx = top_k_indices[cnp_idx, rank]
            iccs_row = iccs_df.iloc[iccs_idx]
            similarity = top_k_scores[cnp_idx, rank]

            detailed_rows.append({
                "cnp_codigo": cnp_row["codigo"],
                "cnp_glosa": cnp_row.get("glosa", ""),
                "cnp_descripcion": cnp_row.get("descripcion", ""),
                "cnp_familia": cnp_row.get("familia_nombre", ""),
                "cnp_articulado": cnp_row.get("articulado", ""),
                "rank": rank + 1,
                "similarity_score": float(similarity),
                "iccs_codigo": iccs_row["codigo_iccs"],
                "iccs_glosa": iccs_row.get("glosa_iccs", ""),
                "iccs_descripcion": iccs_row.get("descripcion", ""),
                "iccs_inclusiones": iccs_row.get("inclusiones", ""),
                "iccs_seccion": iccs_row.get("seccion", ""),
            })

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_path = output_dir / "matches_detallado.csv"
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    # Reporte compacto: una fila por CNP con top-k en columnas
    compact_rows = []
    for cnp_idx in range(len(cnp_df)):
        cnp_row = cnp_df.iloc[cnp_idx]
        row_data = {
            "cnp_codigo": cnp_row["codigo"],
            "cnp_glosa": cnp_row.get("glosa", ""),
            "cnp_descripcion": cnp_row.get("descripcion", "")[:100] + "..." if len(str(cnp_row.get("descripcion", ""))) > 100 else cnp_row.get("descripcion", ""),
        }

        for rank in range(k):
            iccs_idx = top_k_indices[cnp_idx, rank]
            iccs_row = iccs_df.iloc[iccs_idx]
            similarity = top_k_scores[cnp_idx, rank]

            row_data[f"top{rank+1}_codigo"] = iccs_row["codigo_iccs"]
            row_data[f"top{rank+1}_score"] = round(float(similarity), 4)
            row_data[f"top{rank+1}_glosa"] = iccs_row.get("glosa_iccs", "")

        compact_rows.append(row_data)

    compact_df = pd.DataFrame(compact_rows)
    compact_path = output_dir / "matches_compacto.csv"
    compact_df.to_csv(compact_path, index=False, encoding="utf-8-sig")

    # Generar reporte de texto legible
    text_report_path = output_dir / "matches_reporte.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE DE MATCHES CNP -> ICCS (Top 10 vecinos más cercanos)\n")
        f.write("=" * 100 + "\n\n")

        for cnp_idx in range(min(len(cnp_df), 20)):  # Primeros 20 para no hacer el archivo muy grande
            cnp_row = cnp_df.iloc[cnp_idx]
            f.write(f"\n{'─' * 100}\n")
            f.write(f"CNP CÓDIGO: {cnp_row['codigo']}\n")
            f.write(f"GLOSA: {cnp_row.get('glosa', 'N/A')}\n")
            f.write(f"DESCRIPCIÓN: {str(cnp_row.get('descripcion', 'N/A'))[:200]}...\n")
            f.write(f"\nTop {k} matches ICCS:\n")
            f.write(f"{'─' * 100}\n")

            for rank in range(k):
                iccs_idx = top_k_indices[cnp_idx, rank]
                iccs_row = iccs_df.iloc[iccs_idx]
                similarity = top_k_scores[cnp_idx, rank]

                f.write(f"\n  [{rank+1}] Score: {similarity:.4f} | Código ICCS: {iccs_row['codigo_iccs']}\n")
                f.write(f"      Glosa: {iccs_row.get('glosa_iccs', 'N/A')}\n")
                f.write(f"      Descripción: {str(iccs_row.get('descripcion', 'N/A'))[:100]}...\n")

        if len(cnp_df) > 20:
            f.write(f"\n\n... (mostrando solo los primeros 20 de {len(cnp_df)} códigos CNP)\n")
            f.write(f"Ver archivos CSV para el reporte completo.\n")

    print(f"\n{'='*80}")
    print(f"REPORTES GENERADOS:")
    print(f"  - Detallado: {detailed_path}")
    print(f"  - Compacto: {compact_path}")
    print(f"  - Texto (primeros 20): {text_report_path}")
    print(f"{'='*80}\n")

    return detailed_path, compact_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara textos y embeddings E5.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Solo genera tablas preprocesadas.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Tamaño de lote para embeddings.")
    parser.add_argument("--device", type=str, default=None, help="Forzar dispositivo (cuda/cpu).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directorio donde se guardan los artefactos.",
    )
    parser.add_argument("--k", type=int, default=10, help="Número de vecinos más cercanos a calcular.")
    parser.add_argument(
        "--multi-field",
        action="store_true",
        help="Usar embeddings multi-campo con pesos (mejora precisión).",
    )
    parser.add_argument(
        "--cnp-weights",
        type=float,
        nargs=3,
        default=DEFAULT_CNP_WEIGHTS,
        metavar=("GLOSA", "DESC", "FAMILIA"),
        help=f"Pesos para CNP: glosa, descripción, familia (default: {DEFAULT_CNP_WEIGHTS})",
    )
    parser.add_argument(
        "--iccs-weights",
        type=float,
        nargs=4,
        default=DEFAULT_ICCS_WEIGHTS,
        metavar=("GLOSA", "DESC", "INCL", "SECC"),
        help=f"Pesos para ICCS: glosa, descripción, inclusiones, sección (default: {DEFAULT_ICCS_WEIGHTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando datos de CNP e ICCS...")
    cnp_raw = pd.read_excel(CNP_PATH)
    iccs_raw = pd.read_csv(ICCS_PATH)

    # Validar pesos si está en modo multi-field
    if args.multi_field:
        cnp_weights_sum = sum(args.cnp_weights)
        iccs_weights_sum = sum(args.iccs_weights)
        if not np.isclose(cnp_weights_sum, 1.0):
            print(f"ERROR: Los pesos CNP deben sumar 1.0, suma actual: {cnp_weights_sum}")
            return
        if not np.isclose(iccs_weights_sum, 1.0):
            print(f"ERROR: Los pesos ICCS deben sumar 1.0, suma actual: {iccs_weights_sum}")
            return

        print(f"\nModo MULTI-FIELD activado:")
        print(f"  Pesos CNP: Glosa={args.cnp_weights[0]:.2f}, Desc={args.cnp_weights[1]:.2f}, Familia={args.cnp_weights[2]:.2f}")
        print(f"  Pesos ICCS: Glosa={args.iccs_weights[0]:.2f}, Desc={args.iccs_weights[1]:.2f}, Incl={args.iccs_weights[2]:.2f}, Secc={args.iccs_weights[3]:.2f}\n")

    print("Normalizando textos (query/passage)...")
    cnp_prepared = prepare_cnp(cnp_raw, multi_field=args.multi_field)
    iccs_prepared = prepare_iccs(iccs_raw, multi_field=args.multi_field)

    print("Guardando tablas preprocesadas (parquet + csv)...")
    save_table(cnp_prepared, "cnp_preparado", output_dir, save_csv=True)
    save_table(iccs_prepared, "iccs_preparado", output_dir, save_csv=True)

    embeddings_dim = None
    device = resolve_device(args.device)
    if args.skip_embeddings:
        print("Embeddings omitidos por bandera --skip-embeddings.")
    else:
        print(f"Cargando modelo {MODEL_NAME} en {device}...")
        model = SentenceTransformer(MODEL_NAME, device=device)

        if args.multi_field:
            # Modo multi-campo con pesos
            print("Calculando embeddings CNP multi-campo...")
            cnp_field_cols = ["campo_1_glosa", "campo_2_descripcion", "campo_3_familia"]
            cnp_embeddings_array = encode_multi_field(
                model, cnp_prepared, cnp_field_cols, args.cnp_weights, args.batch_size
            )

            print("Calculando embeddings ICCS multi-campo...")
            iccs_field_cols = ["campo_1_glosa", "campo_2_descripcion", "campo_3_inclusiones", "campo_4_seccion"]
            iccs_embeddings_array = encode_multi_field(
                model, iccs_prepared, iccs_field_cols, args.iccs_weights, args.batch_size
            )

            embeddings_dim = cnp_embeddings_array.shape[1]
            cnp_embeddings = cnp_embeddings_array.tolist()
            iccs_embeddings = iccs_embeddings_array.tolist()
        else:
            # Modo original: concatenación simple
            print("Calculando embeddings CNP...")
            cnp_embeddings = encode_texts(model, cnp_prepared["texto_embedding"].tolist(), args.batch_size)
            print("Calculando embeddings ICCS...")
            iccs_embeddings = encode_texts(model, iccs_prepared["texto_embedding"].tolist(), args.batch_size)

            embeddings_dim = len(cnp_embeddings[0]) if cnp_embeddings else None
            cnp_embeddings_array = np.array(cnp_embeddings)
            iccs_embeddings_array = np.array(iccs_embeddings)

        cnp_with_embeddings = cnp_prepared.copy()
        iccs_with_embeddings = iccs_prepared.copy()
        cnp_with_embeddings["embedding"] = cnp_embeddings
        iccs_with_embeddings["embedding"] = iccs_embeddings

        print("Guardando embeddings en parquet (se evita CSV por tamaño).")
        save_table(cnp_with_embeddings, "cnp_embeddings", output_dir, save_csv=False)
        save_table(iccs_with_embeddings, "iccs_embeddings", output_dir, save_csv=False)
        print("Embeddings listos.")

        # Calcular y reportar top-k matches
        print(f"\nCalculando top-{args.k} matches CNP -> ICCS...")

        top_k_indices, top_k_scores = compute_top_k_matches(
            cnp_embeddings_array,
            iccs_embeddings_array,
            k=args.k,
        )

        print(f"Generando reportes de matches...")
        generate_matches_report(
            cnp_prepared,
            iccs_prepared,
            top_k_indices,
            top_k_scores,
            output_dir,
        )

        # Mostrar estadísticas de similitud
        print(f"\n{'='*80}")
        print(f"ESTADÍSTICAS DE SIMILITUD:")
        print(f"  - Similitud promedio top-1: {top_k_scores[:, 0].mean():.4f}")
        print(f"  - Similitud mínima top-1: {top_k_scores[:, 0].min():.4f}")
        print(f"  - Similitud máxima top-1: {top_k_scores[:, 0].max():.4f}")
        print(f"  - Similitud promedio top-{args.k}: {top_k_scores[:, -1].mean():.4f}")
        print(f"{'='*80}\n")

    metadata = build_metadata(
        cnp_prepared,
        iccs_prepared,
        device,
        args.batch_size,
        embeddings_dim,
        multi_field=args.multi_field,
        cnp_weights=args.cnp_weights if args.multi_field else None,
        iccs_weights=args.iccs_weights if args.multi_field else None,
    )
    metadata_path = output_dir / "metadata_embeddings.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata guardada en {metadata_path}")


if __name__ == "__main__":
    main()
