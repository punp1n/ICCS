# Repository Guidelines

## Project Structure & Module Organization
The repository centers on two automation tracks. `CNP/` stores raw Codificación Penal DOCX/XLSX per period plus `procesar_consolidado.py`, which consolidates the newest code definition into parquet/xlsx under the same folder. `Correspondencia automatica/` contains `iccs_tabla.py`, `parse_defs/*.csv`, and the generated `iccs_parse_final.*` outputs derived from `ICSS_PDF/*.pdf`. Manual adjudications live under `Correspondencia manual/<año>/`, while `ICCS_UNODC/` and `ICSS_PDF/` host reference material. Avoid editing generated `.xlsx/.parquet/.csv` directly; regenerate them via the scripts.

## Build, Test, and Development Commands
Use Python 3.11+. Typical bootstrapping:
```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas pdfplumber openpyxl pyarrow
```
Run the consolidator with `python CNP/procesar_consolidado.py`. Ensure `BASE_DIR` keeps pointing to the `CNP` folder before executing. Rebuild the ICCS lookup by running `python "Correspondencia automatica/iccs_tabla.py"`; confirm `ICCS_SPANISH_2016_web.pdf` is present and `parse_defs/*.csv` are up to date. Commands print progress counts—treat warnings about missing tables or CSVs as blockers.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive `snake_case` variables, and docstrings for helpers that implement parsing logic. Use `pathlib.Path` for filesystem work and guard script entry points with `if __name__ == "__main__":`. Keep constants (paths, month maps, regex patterns) grouped near the top and document any locale-specific assumptions.

## Testing Guidelines
Formal tests are not yet in place, so verification is manual: spot-check a few DOCX rows per intake, confirm resulting `consolidado_CNP_2025_2021.*` retains the last valid period per código, and compare article/description merges for duplicates. For ICCS parsing, diff the regenerated CSV/JSON line counts against previous versions and inspect mismatched `glosa_match` rows to catch OCR issues. When feasible, add lightweight `pytest` functions under `tests/` to exercise regex helpers before shipping major refactors.

## Commit & Pull Request Guidelines
Commits should be short, Spanish sentences that describe the action (see `git log`: “procesamiento para consolidar…”). Reference data vintages or sections touched, and avoid bundling unrelated files. Pull requests must explain the intent, list regenerated artifacts, mention manual validation performed, and link to the request ticket; attach screenshots or sample rows when output formats change.

## Contexto de mapeo ICCS (27 de noviembre de 2025)
- Rol: Data Scientist / Data Analyst.
- Objetivo: Mapeo automatizado de códigos penales nacionales (~600) a la Clasificación Internacional de Delitos con Fines Estadísticos (ICCS ~300).

### Infraestructura y recursos
- Hardware: PC local (Ryzen 5 5600, 16GB RAM).
- Aceleración: NVIDIA GeForce RTX 4060 Ti (8GB VRAM) dedicada a embeddings y búsqueda vectorial.
- Software: Windows 11 + WSL (entorno Python).
- Motor de inferencia (lógica): API externa (OpenAI GPT-4o / Claude 3.5 Sonnet / Gemini Pro) para precisión máxima en razonamiento legal.

### Metodología: RAG híbrido
- Enfoque híbrido: GPU local para filtrado semántico y API solo para la decisión final.
- Flujo de trabajo:
  - Preprocesamiento asimétrico: cadenas optimizadas separando información positiva (glosas/descripciones) de la negativa (exclusiones).
  - Retrieval (local/GPU): modelo `intfloat/multilingual-e5-large`, vectorización de ~600 delitos nacionales y ~300 ICCS, búsqueda de top 10 candidatos por similitud coseno.
  - Reasoning (nube/API): LLM recibe el delito nacional y los 10 candidatos preseleccionados; usa exclusiones y notas como filtros lógicos negativos.

### Decisiones técnicas clave
- Jerarquía dual (específica + conservadora): se intenta llegar al nivel 4 (ej. 010321); si hay ambigüedad se reporta también el nivel 2 (ej. 0103) como respaldo.
- Cardinalidad 1:1 estricta: cada delito nacional tiene un único código ICCS principal (“Primary Offense”).
- Motor de embeddings: local (E5-Large) por costo y velocidad; SOTA en recuperación multilingüe.
- Manejo de IDs: fila única; ante códigos nacionales repetidos se procesa por índice de fila o combinación código + articulado.

### Estrategia de datos (feature engineering)
- **Embeddings multi-campo con pesos** (implementado 30-nov-2025):
  - Se generan embeddings separados por campo y se combinan con pesos configurables
  - Permite control granular sobre qué información es más relevante para el matching

- **Input nacional (CNP query)**:
  - Campo 1 - Glosa (peso: 50%): término jurídico principal (ej: "HOMICIDIO.")
  - Campo 2 - Descripción (peso: 35%): definición legal detallada
  - Campo 3 - Familia (peso: 15%): categoría amplia del delito
  - Prefijo E5: `query: ` para optimizar retrieval asimétrico

- **Input ICCS (passage)**:
  - Campo 1 - Glosa (peso: 40%): nombre internacional del delito
  - Campo 2 - Descripción (peso: 35%): definición estándar ICCS
  - Campo 3 - Inclusiones (peso: 20%): ejemplos y casos específicos
  - Campo 4 - Sección (peso: 5%): categoría jerárquica superior
  - Prefijo E5: `passage: ` para optimizar retrieval asimétrico

- **Excluido del embedding**: exclusiones y notas no se vectorizan; se pasan al LLM en la etapa final para filtrar falsos positivos.

- **Similitud combinada**:
  - Embedding final = suma ponderada de embeddings individuales (normalizada)
  - Similitud = coseno entre embedding CNP y embedding ICCS
  - Pesos configurables vía argumentos CLI para experimentación
