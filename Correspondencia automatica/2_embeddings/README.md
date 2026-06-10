# Etapa 2 — Embeddings + Rerank (CNP → ICCS)

Genera, para cada delito del Código Penal Nacional (CNP/CUM), los mejores
candidatos ICCS mediante **búsqueda semántica (embeddings)** y un **rerank**
con cross-encoder. La salida (top-10 reordenado) alimenta la etapa 3 (LLM).

> Actualizado al 10/06/2026: nuevo modelo de embeddings (Qwen3), paso de rerank
> (bge-reranker-v2-m3), insumo CNP 2025 y evaluación A/B. Antes se usaba
> `intfloat/multilingual-e5-large` con embeddings multi-campo ponderados.

## Modelos

| Rol | Modelo | Notas |
|---|---|---|
| Embeddings (default) | `Qwen/Qwen3-Embedding-0.6B` | SOTA multilingüe para su tamaño; corre en CPU |
| Embeddings (baseline A/B) | `intfloat/multilingual-e5-large` | modelo anterior, para comparar |
| Reranker (GPU, recomendado) | `BAAI/bge-reranker-v2-m3` | cross-encoder multilingüe, default de producción |
| Reranker (CPU, usado aquí) | `BAAI/bge-reranker-base` | misma familia, 278M, ~2× más rápido en CPU |

> En CPU, `bge-reranker-v2-m3` (568M) resultó demasiado lento (~3 h para ~13k
> pares). La corrida del 10/06/2026 usó `bge-reranker-base` con `--rerank-pool 8`
> y `--max-length 256` (`--reranker BAAI/bge-reranker-base`). En la máquina con GPU
> conviene volver a `bge-reranker-v2-m3` y un pool mayor.

## Insumos

- **CNP**: `Correspondencia manual/2025/09062026_TC_2025_v1.0.xlsx` (hoja `TC_2025_v1.0`, 668 códigos).
  Columnas usadas: `CUM`, `GLOSA`, `Descripción`, `Familia de delito`.
  Además se conservan `Situación 2025` e `ICCS _2025` (etiqueta manual, usada en el A/B).
  El **CUM 0** (sin glosa/descripción/familia) se marca `no_clasificado` y no se envía al modelo.
- **ICCS**: `1_iccs/outputs/iccs_descripcion.csv` (309 códigos) + `iccs_tabla.csv` (mapeo de sección N1).

## Flujo

```
preparar_embeddings.py  →  top-50 candidatos por CNP (coseno)
        │
        ▼
rerank_matches.py       →  reordena top-25 con cross-encoder → top-10 final
        │
        ▼
evaluar_ab.py           →  métricas de acierto de sección vs ICCS_2025
```

## Uso (WSL, entorno `~/.venvs/iccs`)

> El `.venv/` versionado quedó obsoleto (symlinks rotos por OneDrive). El entorno
> de ejecución vive en el home de WSL (ext4). Para recrearlo:
> ```bash
> python3.12 -m venv ~/.venvs/iccs
> ~/.venvs/iccs/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
> ~/.venvs/iccs/bin/pip install -r requirements.txt
> ```

```bash
PY=~/.venvs/iccs/bin/python

# 1) Embeddings + top-50 (modelo por defecto: qwen3)
$PY preparar_embeddings.py --model qwen3
$PY preparar_embeddings.py --model e5        # baseline para el A/B

# 2) Rerank top-25 → top-10
$PY rerank_matches.py --model qwen3 --rerank-pool 25
$PY rerank_matches.py --model e5 --rerank-pool 25

# 3) Evaluación A/B (acierto de sección vs ICCS_2025)
$PY evaluar_ab.py
```

Opciones útiles: `--k` (tamaño del pool, default 50), `--batch-size`,
`--device cpu|cuda`, `--rerank-pool` (candidatos a reordenar), `--top-final`.

## Salidas (`outputs/<model>/`)

- `cnp_preparado.csv`, `iccs_preparado.csv` — textos normalizados.
- `cnp_no_clasificado.csv` — CUM sin contenido (p.ej. CUM 0).
- `matches_detallado.csv` — pool top-K (una fila por candidato).
- `matches_compacto.csv` — top-10 por embeddings (top1..top10 en columnas).
- `matches_rerank_detallado.csv` / `matches_rerank_compacto.csv` — **top-10 final reordenado**.
- `metadata_embeddings.json`, `metadata_rerank.json` — trazabilidad.

A nivel de carpeta `outputs/`:
- `ab_report.csv` / `ab_report.md` — resultados de la evaluación A/B.

## Resultados A/B (10/06/2026)

Acierto de **sección ICCS (N1)** contra la etiqueta manual `ICCS_2025` (n=590 con
etiqueta 1–11). Reranker usado: `bge-reranker-base` (CPU), `pool=8`, `len=256`.

| Configuración | top-1 | top-3 | recall@10 |
|---|---:|---:|---:|
| **qwen3 (embeddings)** | **0.597** | 0.788 | **0.922** |
| qwen3 + rerank | 0.578 | **0.814** | 0.903 |
| e5 (embeddings) | 0.573 | 0.773 | 0.912 |
| e5 + rerank | 0.571 | 0.783 | 0.883 |

**Lectura:**
- **Qwen3 > e5** en todas las métricas → el cambio de modelo está justificado.
- El rerank (modelo liviano, CPU) **mejora el top-3** pero baja levemente top-1 y
  recall@10; la caída de recall@10 es en parte artefacto de `pool=8` (solo se
  reordenan 8 candidatos). La etiqueta es **a nivel sección** (coarse): el aporte
  del rerank al nivel granular no se mide aquí.
- Pendiente en la máquina GPU: repetir con `bge-reranker-v2-m3` y `pool` ≥ 15 para
  evaluar la ganancia real del rerank.

El insumo para la etapa 3 (LLM) es `outputs/qwen3/matches_rerank_detallado.csv`.

## Archivos

- `comun.py` — utilidades compartidas (carga de insumos, mapeo de sección, modelos).
- `preparar_embeddings.py` — embeddings + top-K.
- `rerank_matches.py` — rerank con cross-encoder.
- `evaluar_ab.py` — evaluación A/B.
- `analizar_*.py`, `comparar_resultados.py` — diagnósticos puntuales (heredados).
