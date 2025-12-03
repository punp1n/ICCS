# Embeddings CNP -> ICCS

Pipeline para preparar textos, generar embeddings y calcular matches CNP -> ICCS con `intfloat/multilingual-e5-large`.

## Uso rápido (Windows)

### Desde CMD (recomendado):
1. Abrir CMD y navegar a la carpeta del proyecto:
   ```cmd
   cd "C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\embeddings"
   ```

2. Activar el entorno virtual:
   ```cmd
   .venv\Scripts\activate.bat
   ```

3. Ejecutar el script:
   ```cmd
   python preparar_embeddings.py
   ```

### Desde PowerShell:
1. `cd "Correspondencia automatica\embeddings"`
2. Activar el entorno virtual: `.\.venv\Scripts\Activate.ps1`
3. `python preparar_embeddings.py` para generar embeddings y reportes de matches.

### Opciones de ejecución
```bash
python preparar_embeddings.py [opciones]
```

**Opciones disponibles:**
- `--k 10` número de vecinos más cercanos (default: 10)
- `--skip-embeddings` solo prepara los textos sin generar embeddings
- `--batch-size 16` tamaño de lote para embeddings (default: 16, reducir si falta VRAM)
- `--device cuda` fuerza GPU (default: auto-detecta cuda si está disponible)
- `--device cpu` fuerza CPU
- `--output-dir "ruta"` directorio de salida (default: `artifacts/`)

**Ejemplos:**
```bash
# Ejecución estándar con GPU (genera top-10 matches)
python preparar_embeddings.py

# Ajustar k para obtener más candidatos
python preparar_embeddings.py --k 20

# Reducir batch si se queda sin VRAM
python preparar_embeddings.py --batch-size 8

# Forzar CPU
python preparar_embeddings.py --device cpu

# Solo preprocesar textos
python preparar_embeddings.py --skip-embeddings
```

## Cómo se construyen los textos
- **CNP (query)**: `glosa + descripcion + familia_nombre`, prefijo `query:`.
- **ICCS (passage)**: `glosa_iccs + descripcion + inclusiones (+ seccion)`, prefijo `passage:`.
- Espacios normalizados y hash por fila para trazabilidad.
- **Nota**: Se utilizan TODOS los códigos ICCS independiente de su nivel jerárquico (nivel 1, 2, 3 o 4). El algoritmo k-NN seleccionará los mejores matches basándose en similitud semántica.

## Salidas (carpeta artifacts/)

### Tablas preprocesadas
- `cnp_preparado.(parquet|csv)` textos CNP limpios con hash
- `iccs_preparado.(parquet|csv)` textos ICCS limpios con hash

### Embeddings
- `cnp_embeddings.parquet` embeddings CNP (columna `embedding`)
- `iccs_embeddings.parquet` embeddings ICCS (columna `embedding`)

### Reportes de matches (NUEVO)
- **`matches_compacto.csv`**: Una fila por código CNP con los top-k candidatos en columnas
  - Columnas: `cnp_codigo`, `cnp_glosa`, `top1_codigo`, `top1_score`, `top1_glosa`, ..., `top10_codigo`, `top10_score`, `top10_glosa`
  - **Este es el archivo más útil para revisar matches rápidamente**

- **`matches_detallado.csv`**: Una fila por cada match (CNP × k filas totales)
  - Incluye toda la información de CNP e ICCS para cada match
  - Columnas: `cnp_codigo`, `cnp_glosa`, `cnp_descripcion`, `cnp_familia`, `cnp_articulado`, `rank`, `similarity_score`, `iccs_codigo`, `iccs_glosa`, `iccs_descripcion`, `iccs_inclusiones`, `iccs_seccion`
  - Útil para análisis detallado o filtrado por similitud
  - **Nota**: Todos los códigos CNP tendrán exactamente k matches asignados

- **`matches_reporte.txt`**: Reporte en texto plano de los primeros 20 códigos CNP
  - Formato legible para revisión rápida
  - Muestra código, glosa y top-k matches con scores

### Metadata
- `metadata_embeddings.json`: modelo, dispositivo, batch, hashes de textos y filas procesadas

## Estadísticas generadas
El script imprime automáticamente:
- Similitud promedio del mejor match (top-1)
- Similitud mínima y máxima del top-1
- Similitud promedio del candidato k-ésimo
