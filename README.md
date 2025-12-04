# Clasificación Automatizada CNP-ICCS

https://github.com/punp1n/ICCS

## Sistema de Mapeo Inteligente entre Códigos Penales Nacionales (CNP) y la Clasificación Internacional de Delitos con Fines Estadísticos (ICCS)

---

## Resumen Ejecutivo

Este proyecto implementa un pipeline automatizado para clasificar delitos del Código Penal Nacional (CNP) de Chile según la Clasificación Internacional de Delitos con Fines Estadísticos (ICCS) de las Naciones Unidas. El sistema combina técnicas de procesamiento de lenguaje natural (NLP), búsqueda vectorial mediante embeddings y razonamiento legal asistido por modelos de lenguaje de gran escala (LLM).

**Motivación**: Tradicionalmente, la correspondencia entre CNP e ICCS se realizaba de forma manual por analistas INE con asesorías UNODC y la Corporación Administrativa del Poder Judicial (CAPJ), proceso que requería semanas de trabajo. Este sistema reduce el tiempo de clasificación a minutos, manteniendo alta precisión mediante validación experta.

**Resultados**: El sistema procesa ~600 códigos CNP y los clasea contra ~300 códigos ICCS, generando correspondencias de alta confianza validadas contra etiquetas manuales existentes.

---

## Tabla de Contenidos

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Requisitos Técnicos](#requisitos-técnicos)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Pipeline de Procesamiento](#pipeline-de-procesamiento)
   - [Fase 1: Preparación de Datos CNP](#fase-1-preparación-de-datos-cnp)
   - [Fase 2: Preparación de Datos ICCS](#fase-2-preparación-de-datos-iccs)
   - [Fase 3: Generación de Embeddings y Búsqueda Vectorial](#fase-3-generación-de-embeddings-y-búsqueda-vectorial)
   - [Fase 4: Clasificación con LLM](#fase-4-clasificación-con-llm)
5. [Estructura de Archivos](#estructura-de-archivos)
6. [Salidas del Sistema](#salidas-del-sistema)
7. [Evaluación y Métricas](#evaluación-y-métricas)
8. [Consideraciones Metodológicas](#consideraciones-metodológicas)
9. [Solución de Problemas](#solución-de-problemas)
10. [Referencias](#referencias)

---

## Arquitectura del Sistema

El sistema utiliza una arquitectura híbrida que combina:

1. **Procesamiento Local (GPU)**: Generación de embeddings semánticos con modelo `intfloat/multilingual-e5-large`
2. **Razonamiento en Nube (API)**: Clasificación final con modelo `gpt-4o-mini` de OpenAI
3. **Validación Experta**: Comparación contra correspondencia manual para evaluar precisión

### Diagrama de Flujo

```
[Datos Crudos CNP (.docx)]  +  [PDF ICCS]  +  [Parse Defs CSV]
            |                        |                 |
            v                        v                 v
    ┌───────────────┐        ┌──────────────────────────┐
    │ Fase 1: CNP   │        │ Fase 2: ICCS             │
    │ Consolidación │        │ Descripción Completa     │
    └───────┬───────┘        └──────────┬───────────────┘
            |                           |
            v                           v
    [consolidado_CNP.xlsx]      [iccs_descripcion.csv]
            |                           |
            └──────────┬────────────────┘
                       v
            ┌──────────────────────┐
            │ Fase 3: Embeddings   │
            │ + Búsqueda Vectorial │
            └──────────┬───────────┘
                       v
            [matches_detallado.csv]
                       |
                       v
            ┌──────────────────────┐
            │ Fase 4: LLM Filter   │
            │ (Razonamiento Legal) │
            └──────────┬───────────┘
                       v
        [clasificacion_final.csv]
```

---

## Requisitos Técnicos

### Hardware Recomendado

- **CPU**: AMD Ryzen 5 5600 o equivalente
- **RAM**: 16 GB mínimo
- **GPU**: NVIDIA GeForce RTX 4060 Ti (8GB VRAM) o superior
  - *Nota*: La GPU es opcional pero altamente recomendada para acelerar la generación de embeddings (~100x más rápido)
- **Almacenamiento**: 5 GB libres

### Software

- **Sistema Operativo**: Windows 10/11 con WSL2 (opcional) o Linux nativo
- **Python**: 3.11 o superior
- **CUDA**: 11.8+ (si se usa GPU NVIDIA)

### Dependencias Python

El proyecto utiliza múltiples entornos virtuales especializados:

#### Para procesamiento CNP e ICCS:
```
pandas>=2.0.0
openpyxl>=3.1.0
pyarrow>=22.0.0
pdfplumber>=0.10.0
```

#### Para generación de embeddings:
```
pandas==2.3.3
pyarrow==22.0.0
openpyxl==3.1.5
sentence-transformers==5.1.2
torch==2.9.1
scikit-learn==1.7.2
```

#### Para clasificación con LLM:
```
openai>=1.0.0
pandas>=2.0.0
tqdm>=4.65.0
```

---

## Instalación y Configuración

### 1. Clonar o descargar el repositorio

```bash
git clone <url-del-repositorio>
cd ICCS
```

### 2. Configurar entornos virtuales

**Opción A: Windows CMD (recomendado)**

```cmd
REM Entorno para embeddings
cd "Correspondencia automatica\embeddings"
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
deactivate

REM Entorno para LLM
cd ..\llm_filter
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
deactivate
```

**Opción B: WSL/Linux**

```bash
# Entorno para embeddings
cd "Correspondencia automatica/embeddings"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate

# Entorno para LLM
cd ../llm_filter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```


---

## Pipeline de Procesamiento

### Fase 1: Preparación de Datos CNP

**Objetivo**: Consolidar archivos `.docx` del Código Penal Nacional de múltiples periodos en un archivo único con la versión más reciente de cada código.

**Script**: `CNP/procesar_consolidado.py`

**Entrada**:
- Carpetas por periodo: `CNP/2025_julio/`, `CNP/2025_enero/`, ..., `CNP/2021_enero/`
- Cada carpeta contiene archivos `.docx` con tablas estructuradas

**Proceso**:
1. **Extracción XML**: Lee las tablas de los `.docx` usando `zipfile` y `xml.etree`
2. **Parsing Estructurado**:
   - Detecta códigos penales (regex `^\d+`)
   - Extrae familia del delito (limpiando referencias legales)
   - Separa articulado de descripciones
3. **Limpieza de Texto**:
   - Elimina saltos de línea y caracteres de control
   - Normaliza espacios múltiples
   - Limpia nombres de familia (elimina "Libro X Título Y", referencias a leyes)
4. **Consolidación Temporal**:
   - Aplica estrategia "último vigente": ante códigos duplicados, conserva el del periodo más reciente
   - Ranking temporal: año × 100 + mes (ej: julio 2025 = 202507)
5. **Deduplicación**: Agrupa artículos y descripciones por código

**Salida**:
- `CNP/consolidado_CNP_2025_2021.xlsx`
- `CNP/consolidado_CNP_2025_2021.parquet`

**Columnas**:
- `codigo`: Código penal (ej: "01001")
- `familia_nombre`: Categoría amplia del delito
- `glosa`: Nombre breve del delito
- `articulado`: Artículos del código penal separados por " ; "
- `descripcion`: Descripción legal completa
- `ultimo_vigente`: Periodo de vigencia (ej: "2025_julio")

**Ejecución**:
```cmd
cd CNP
python procesar_consolidado.py
```

**Tiempo estimado**: 2-5 minutos (~600 códigos procesados)

---

### Fase 2: Preparación de Datos ICCS

**Objetivo**: Generar tabla completa de códigos ICCS con descripciones, inclusiones, exclusiones y notas desde archivos CSV preprocesados con IA.

**Script**: `Correspondencia automatica/scripts/generar_iccs_descripcion.py`

**Entrada**:
- **PDF ICCS**: `ICSS_PDF/ICCS_SPANISH_2016_web.pdf` (solo para extracción de secciones)
- **Parse Defs CSVs**: `Correspondencia automatica/parse_defs/parse_defs_secc_*.csv`
  - `parse_defs_secc_01_03.csv` (Secciones 1-3)
  - `parse_defs_secc_04_08.csv` (Secciones 4-8)
  - `parse_defs_secc_09_11.csv` (Secciones 9-11)

**Proceso**:
1. **Extracción de Secciones desde PDF**:
   - Usa `pdfplumber` para leer páginas 26-34 del PDF ICCS
   - Detecta encabezados de sección (regex: "Sección \d+ ...")
   - Asocia cada código ICCS con su sección jerárquica
2. **Carga de Parse Defs**:
   - Los archivos CSV fueron generados previamente con **Google Gemini 3.0** mediante prompts especializados para extraer:
     - `codigo_iccs`: Código numérico
     - `glosa_iccs`: Nombre del delito
     - `descripcion`: Definición legal
     - `inclusiones`: Ejemplos y casos específicos
     - `exclusiones`: Delitos que NO aplican
     - `notas`: Contexto adicional
3. **Merge de Datos**:
   - Une mapeo de secciones (del PDF) con datos estructurados (de CSVs)
   - Aplica ajustes manuales para códigos con discrepancias (ej: código 1042, 908)
4. **Validación**:
   - Reporta códigos sin sección asignada

**Salida**:
- `Correspondencia automatica/outputs/iccs_descripcion.csv`
- `Correspondencia automatica/outputs/iccs_descripcion.xlsx`
- `Correspondencia automatica/outputs/iccs_descripcion.json`

**Columnas**:
- `codigo_iccs`: Código ICCS (ej: "0101")
- `glosa_iccs`: Nombre del delito internacional
- `seccion`: Categoría jerárquica (ej: "Actos que causan la muerte...")
- `descripcion`: Definición estándar ICCS
- `inclusiones`: Casos que aplican
- `exclusiones`: Casos que NO aplican (crítico para LLM)
- `notas`: Aclaraciones legales

**Ejecución**:
```cmd
cd "Correspondencia automatica\scripts"
python generar_iccs_descripcion.py
```

**Tiempo estimado**: 10-20 segundos (~300 códigos procesados)

---

### Fase 3: Generación de Embeddings y Búsqueda Vectorial

**Objetivo**: Generar representaciones vectoriales semánticas de CNP e ICCS y encontrar los top-k candidatos más similares para cada código CNP.

**Script**: `Correspondencia automatica/embeddings/preparar_embeddings.py`

**Entrada**:
- `CNP/consolidado_CNP_2025_2021.xlsx`
- `Correspondencia automatica/outputs/iccs_descripcion.csv`

**Proceso**:

#### 3.1. Preprocesamiento de Textos

**CNP (Query)**:
- Concatena campos con pesos semánticos:
  - Glosa (50%): Término jurídico principal
  - Descripción (35%): Definición legal
  - Familia (15%): Categoría amplia
- Añade prefijo E5: `"query: "` (optimización para retrieval asimétrico)
- Normaliza espacios y genera hash SHA-256 para trazabilidad

**ICCS (Passage)**:
- Concatena campos con pesos semánticos:
  - Glosa (40%): Nombre internacional
  - Descripción (35%): Definición estándar
  - Inclusiones (20%): Ejemplos específicos
  - Sección (5%): Contexto jerárquico
- Añade prefijo E5: `"passage: "`
- **Nota**: Exclusiones y notas NO se vectorizan; se reservan para la etapa de razonamiento LLM

#### 3.2. Generación de Embeddings

**Modelo**: `intfloat/multilingual-e5-large`
- **Dimensión**: 1024
- **Idioma**: Multilingüe (optimizado para español)
- **Arquitectura**: Transformer encoder-only
- **Estrategia**: Mean pooling sobre tokens
- **Dispositivo**: CUDA (GPU) si está disponible, CPU caso contrario

**Configuración**:
- Batch size: 16 (ajustar a 8 si hay problemas de VRAM)
- Normalización L2: Activa (para similitud coseno)

#### 3.3. Búsqueda de Vecinos Más Cercanos (k-NN)

**Algoritmo**: Fuerza bruta con similitud coseno
- **k**: 10 (top-10 candidatos por defecto)
- **Métrica**: Similitud coseno (dot product tras normalización L2)
- **Implementación**: `scikit-learn.neighbors.NearestNeighbors`

**Proceso**:
1. Ajusta índice k-NN con embeddings ICCS
2. Para cada embedding CNP, encuentra los k códigos ICCS más similares
3. Ordena por score de similitud (rango: 0-1, donde 1 = identidad)

#### 3.4. Generación de Reportes

**Salidas**:

1. **Embeddings**:
   - `artifacts/cnp_embeddings.parquet`
   - `artifacts/iccs_embeddings.parquet`

2. **Matches Detallado** (`artifacts/matches_detallado.csv`):
   - Una fila por cada match (CNP × k filas)
   - Columnas: `cnp_codigo`, `cnp_glosa`, `cnp_descripcion`, `cnp_familia`, `rank`, `similarity_score`, `iccs_codigo`, `iccs_glosa`, `iccs_descripcion`, `iccs_inclusiones`
   - **Uso**: Análisis granular, filtrado por umbral de similitud

3. **Matches Compacto** (`artifacts/matches_compacto.csv`):
   - Una fila por código CNP con candidatos en columnas
   - Columnas: `cnp_codigo`, `cnp_glosa`, `top1_codigo`, `top1_score`, `top1_glosa`, ..., `top10_codigo`, `top10_score`, `top10_glosa`
   - **Uso**: Revisión rápida de candidatos

4. **Reporte Texto** (`artifacts/matches_reporte.txt`):
   - Primeros 20 códigos en formato legible

5. **Metadata** (`artifacts/metadata_embeddings.json`):
   - Modelo usado, dispositivo, hashes de datos

**Ejecución**:

```cmd
cd "Correspondencia automatica\embeddings"
.venv\Scripts\activate.bat

REM Ejecución estándar (GPU, top-10)
python preparar_embeddings.py

REM Opciones avanzadas
python preparar_embeddings.py --k 20           # Top-20 candidatos
python preparar_embeddings.py --batch-size 8   # Reducir uso de VRAM
python preparar_embeddings.py --device cpu     # Forzar CPU
```

**Tiempo estimado**:
- GPU (RTX 4060 Ti): 2-3 minutos
- CPU: 30-40 minutos

**Estadísticas generadas**:
- Similitud promedio del top-1
- Similitud mínima/máxima del top-1
- Similitud promedio del candidato k-ésimo

---

### Fase 4: Clasificación con LLM

**Objetivo**: Aplicar razonamiento legal experto para elegir el mejor código ICCS entre los candidatos top-k, considerando exclusiones, notas y gravedad del delito.

**Script**: `Correspondencia automatica/llm_filter/filtrar_con_llm.py`

**Entrada**:
- `Correspondencia automatica/embeddings/artifacts/matches_detallado.csv`
- `Correspondencia automatica/outputs/iccs_descripcion.csv`

**Proceso**:

#### 4.1. Preparación de Candidatos

1. **Agrupación**: Agrupa matches por `cnp_codigo`
2. **Selección top-k**: Toma los 10 candidatos con mejor score (configurable)
3. **Join con ICCS completo**: Obtiene `exclusiones` y `notas` desde `iccs_descripcion.csv`

#### 4.2. Construcción del Prompt

**Estructura del Prompt**:

```
DELITO NACIONAL (CNP):
- Código: [cnp_codigo]
- Glosa: [cnp_glosa]
- Descripción: [cnp_descripcion]
- Familia: [cnp_familia]

CANDIDATOS ICCS (Top 10 por similitud semántica):
1. Código ICCS: [iccs_codigo]
   Glosa: [iccs_glosa]
   Descripción: [iccs_descripcion]
   Inclusiones: [iccs_inclusiones]
   EXCLUSIONES: [iccs_exclusiones]
   NOTAS: [iccs_notas]
   Score similitud embeddings: [similarity_score]
[... candidatos 2-10 ...]

INSTRUCCIONES CRÍTICAS:
1. Elige el código ICCS que MEJOR se aproxime a la definición del delito CNP.
2. NO busques el código más específico; busca el MÁS PRECISO.
3. Considera ESPECIALMENTE las EXCLUSIONES y NOTAS de cada candidato.
4. Si una exclusión descarta el delito CNP, ese candidato NO es válido.
5. Las NOTAS dan contexto sobre cuándo aplicar cada código.
6. DELITOS MÁS GRAVOSOS: Si dos códigos se excluyen mutuamente (ej: robo vs hurto),
   elige el delito MÁS GRAVOSO. Ejemplos:
   - Robo > Hurto
   - Homicidio > Lesiones
   - Violación > Abuso sexual
7. DELITOS SIN DESCRIPCIÓN: Si el delito CNP no tiene descripción:
   a) Intenta clasificar usando GLOSA y FAMILIA aunque sea en términos genéricos.
   b) Solo devuelve "NINGUNO" si el delito es completamente genérico (ej: "otros delitos").
8. Si NINGÚN candidato es apropiado, devuelve "NINGUNO" y explica por qué.
9. Tu análisis legal es prioritario sobre el score de similitud.
10. Si una exclusión/inclusión menciona un código mejor, elige ese código aunque no esté en la lista.
```

**Formato de Respuesta** (JSON):
```json
{
  "iccs_elegido": "0101",
  "confianza": "alta",
  "justificacion": "El delito CNP de HOMICIDIO coincide exactamente con ICCS 0101...",
  "exclusiones_aplicadas": ["Descartado 0102 por exclusión X"]
}
```

#### 4.3. Llamada al LLM

**Modelo**: `gpt-4o-mini`
- **Temperatura**: 0.1 (implícito en código)
- **Formato**: JSON estructurado (`response_format={"type": "json_object"}`)
- **Reintentos**: 3 intentos con backoff exponencial (2s, 4s, 6s)
- **Validación**: Verifica campos obligatorios (`iccs_elegido`, `confianza`, `justificacion`)

**Mensaje del Sistema**:
```
Eres un experto en clasificación de delitos penales. Respondes solo en JSON válido.
Cuando hay exclusión mutua, prefiere el delito más gravoso.
Para delitos sin descripción, clasifica con glosa y familia.
Solo responde NINGUNO si el delito es completamente genérico sin contexto suficiente.
```

#### 4.4. Procesamiento en Lote

- **Checkpoint**: Guarda progreso cada 10 códigos en `checkpoint.json`
- **Barra de progreso**: Usa `tqdm` para visualización
- **Manejo de errores**: Log detallado en `outputs/errores.log`

#### 4.5. Evaluación vs Correspondencia Manual

**Archivo de referencia**: `Correspondencia manual/2024/07102025_TC_Final_2023-2024_v1.2.xlsx`

**Proceso**:
1. Lee etiquetas manuales (columnas `N4-2024 UNODC`, `N3-2024 UNODC`, etc.)
2. Normaliza códigos (elimina ceros iniciales)
3. Compara predicciones LLM vs etiquetas manuales
4. Calcula métricas:
   - Coincidencias exactas
   - Discrepancias (diferentes códigos asignados)
   - Casos sin clasificación LLM
5. Genera reporte detallado en `outputs/comparacion_llm_vs_manual.xlsx`

**Salidas**:

1. **Clasificación Completa** (`outputs/clasificacion_con_justificacion.csv`):
   - Columnas: `cnp_codigo`, `cnp_glosa`, `cnp_descripcion`, `cnp_familia`, `iccs_elegido`, `iccs_glosa_elegida`, `confianza`, `justificacion`, `exclusiones_aplicadas`, `top1_codigo`, `top1_score`, ..., `top10_codigo`, `top10_score`

2. **Clasificación Compacta** (`outputs/clasificacion_final.csv`):
   - Columnas esenciales: `cnp_codigo`, `cnp_glosa`, `iccs_elegido`, `iccs_glosa_elegida`, `confianza`, `top1_codigo`, `top1_score`, `top2_codigo`, `top2_score`

3. **Comparación Manual** (`outputs/comparacion_llm_vs_manual.xlsx`):
   - Columnas: `cnp_codigo`, `glosa_manual`, `manual_codigo_granular`, `llm_codigo`, `iccs_glosa_elegida`, `confianza`, `justificacion`, coincide (TRUE/FALSE)

4. **Log de Errores** (`outputs/errores.log`):
   - JSON con códigos que fallaron y razones

**Ejecución**:

```cmd
cd "Correspondencia automatica\llm_filter"
.venv\Scripts\activate.bat

REM Modo test (solo 10 códigos)
python filtrar_con_llm.py --test

REM Procesamiento completo
python filtrar_con_llm.py

REM Procesar solo primeros 50 códigos
python filtrar_con_llm.py --limite 50
```

**Confirmación de Ejecución**:
El script muestra estimación de costo y solicita confirmación:
```
ESTIMACION DE PROCESAMIENTO:
  Códigos CNP a procesar: 565
  Tokens estimados: ~1,243,000
  Costo estimado: $0.24 USD
  Tiempo estimado: ~18.8 minutos
  Modelo: gpt-4o-mini

Proceder con el procesamiento? (s/n):
```

**Tiempo estimado**:
- Test (10 códigos): 20-30 segundos
- Completo (~565 códigos): 15-20 minutos

**Estadísticas generadas**:
```
ESTADISTICAS:
  Total procesados: 565
  NINGUNO asignado: 12
  Confianza alta: 453
  Confianza media: 89
  Confianza baja: 11
  Coincide con top-1 embedding: 487 (86.2%)

EVALUACION VS CORRESPONDENCIA MANUAL:
  Total con etiqueta manual: 543
  LLM con código asignado: 553
  Coincidencias: 478
  Discrepancias: 65
  Manual con NINGUNA respuesta LLM: 0
```

---

## Estructura de Archivos

Para facilitar la reproducibilidad y mantenimiento, se recomienda la siguiente estructura de carpetas para el repositorio limpio:

```
ICCS/
│
├── README.md                          # Este archivo (documentación principal)
├── .gitignore                         # Exclusiones para control de versiones
│
├── CNP/                               # Procesamiento de Código Penal Nacional
│   ├── procesar_consolidado.py       # Script de consolidación
│   ├── 2025_julio/                    # Carpetas por periodo (solo en local)
│   ├── 2025_enero/
│   ├── ...
│   ├── 2021_enero/
│   ├── consolidado_CNP_2025_2021.xlsx # Salida consolidada (NO subir)
│   └── consolidado_CNP_2025_2021.parquet
│
├── ICSS_PDF/                          # PDF de referencia ICCS
│   └── ICCS_SPANISH_2016_web.pdf      # PDF oficial UNODC (NO subir por tamaño)
│
├── Correspondencia automatica/
│   │
│   ├── scripts/
│   │   └── generar_iccs_descripcion.py  # Generación de tabla ICCS
│   │
│   ├── parse_defs/                      # CSVs generados con Gemini 3.0
│   │   ├── parse_defs_secc_01_03.csv    # Secciones 1-3 (SUBIR)
│   │   ├── parse_defs_secc_04_08.csv    # Secciones 4-8 (SUBIR)
│   │   └── parse_defs_secc_09_11.csv    # Secciones 9-11 (SUBIR)
│   │
│   ├── outputs/                         # Salidas ICCS
│   │   ├── iccs_descripcion.csv         # Tabla ICCS completa (NO subir)
│   │   ├── iccs_descripcion.xlsx
│   │   └── iccs_descripcion.json
│   │
│   ├── embeddings/
│   │   ├── preparar_embeddings.py       # Script de embeddings
│   │   ├── requirements.txt             # Dependencias (SUBIR)
│   │   ├── .venv/                       # Entorno virtual (NO subir)
│   │   └── artifacts/                   # Salidas embeddings (NO subir)
│   │       ├── cnp_embeddings.parquet
│   │       ├── iccs_embeddings.parquet
│   │       ├── matches_detallado.csv
│   │       ├── matches_compacto.csv
│   │       ├── matches_reporte.txt
│   │       └── metadata_embeddings.json
│   │
│   └── llm_filter/
│       ├── filtrar_con_llm.py           # Script LLM
│       ├── requirements.txt             # Dependencias (SUBIR)
│       ├── .venv/                       # Entorno virtual (NO subir)
│       └── outputs/                     # Salidas LLM (NO subir)
│           ├── clasificacion_final.csv
│           ├── clasificacion_con_justificacion.csv
│           ├── comparacion_llm_vs_manual.xlsx
│           ├── errores.log
│           └── checkpoint.json
│
└── Correspondencia manual/              # Etiquetas manuales para validación
    └── 2024/
        └── 07102025_TC_Final_2023-2024_v1.2.xlsx  # (NO subir)
```

### Archivos a Subir al Repositorio

**✅ INCLUIR (Scripts y Configuración)**:
- `README.md`
- `.gitignore`
- `CNP/procesar_consolidado.py`
- `Correspondencia automatica/scripts/generar_iccs_descripcion.py`
- `Correspondencia automatica/parse_defs/*.csv` (3 archivos)
- `Correspondencia automatica/embeddings/preparar_embeddings.py`
- `Correspondencia automatica/embeddings/requirements.txt`
- `Correspondencia automatica/llm_filter/filtrar_con_llm.py`
- `Correspondencia automatica/llm_filter/requirements.txt`

**❌ EXCLUIR (Datos, Entornos, Salidas)**:
- `.venv/` (todos los entornos virtuales)
- `*.xlsx`, `*.parquet`, `*.csv` (excepto `parse_defs/*.csv`)
- `*.pdf`
- `artifacts/`
- `outputs/`
- `CNP/2025_julio/`, `CNP/2025_enero/`, etc. (carpetas de datos crudos)
- `checkpoint.json`
- `*.log`

**Justificación**: Se suben solo scripts reproducibles y archivos de configuración pequeños. Los datos crudos y salidas se regeneran localmente.

---

## Salidas del Sistema

### Salida Final Principal

**Archivo**: `Correspondencia automatica/llm_filter/outputs/clasificacion_final.csv`

**Descripción**: Tabla compacta con la clasificación CNP→ICCS de cada delito.

**Columnas**:
- `cnp_codigo`: Código penal nacional
- `cnp_glosa`: Nombre del delito CNP
- `iccs_elegido`: Código ICCS asignado por el LLM
- `iccs_glosa_elegida`: Nombre del delito ICCS
- `confianza`: Nivel de confianza (`alta`, `media`, `baja`)
- `top1_codigo`, `top1_score`: Mejor candidato por embeddings
- `top2_codigo`, `top2_score`: Segundo mejor candidato

**Uso**: Importar a sistemas de estadísticas criminales, generar reportes oficiales.

### Salidas Intermedias

1. **Consolidado CNP**: `CNP/consolidado_CNP_2025_2021.xlsx`
2. **Descripción ICCS**: `Correspondencia automatica/outputs/iccs_descripcion.csv`
3. **Matches Detallado**: `Correspondencia automatica/embeddings/artifacts/matches_detallado.csv`
4. **Clasificación con Justificación**: `Correspondencia automatica/llm_filter/outputs/clasificacion_con_justificacion.csv`

---

## Evaluación y Métricas

### Métricas de Similitud (Embeddings)

- **Similitud Coseno Promedio (Top-1)**: ~0.75 (varía según configuración de pesos)
- **Rango**: 0.42 - 0.98
- **Interpretación**: Valores >0.7 indican alta similitud semántica

### Métricas de Clasificación (LLM)

**Basadas en correspondencia manual 2024**:
- **Exactitud**: ~88% (coincidencias exactas con etiquetas manuales)
- **Coincidencia con Top-1 Embedding**: ~86%
- **Confianza Alta**: ~80% de casos
- **Tasa de No Clasificación (NINGUNO)**: <3%

**Nota**: Las discrepancias (~12%) no necesariamente son errores. Muchas se deben a:
- Ambigüedad legal legítima
- Múltiples interpretaciones válidas según nivel de granularidad
- Actualizaciones en criterios entre versiones manuales

---

## Consideraciones Metodológicas

### 1. Cardinalidad 1:1 Estricta

El sistema asigna **un único código ICCS principal** por delito CNP, siguiendo el concepto de "Primary Offense" de ICCS. Aunque legalmente algunos delitos podrían clasificarse en múltiples categorías, se prioriza la categoría más representativa.

### 2. Jerarquía Dual (Específica + Conservadora)

El prompt del LLM enfatiza precisión sobre especificidad. Es preferible asignar un código ICCS de nivel 2 (genérico) con alta confianza que forzar un código de nivel 4 (específico) con ambigüedad.

### 3. Exclusiones como Filtros Negativos

Las exclusiones ICCS son críticas para el razonamiento:
- **No se vectorizan** (no contaminan la similitud semántica)
- **Se pasan explícitamente al LLM** para filtrado lógico
- **Ejemplo**: Si ICCS 0103 (Homicidio no intencional) excluye "actos intencionales", el LLM descartará este código para un CNP de homicidio doloso.

### 4. Delitos Más Gravosos

Ante exclusión mutua (ej: robo 0501 vs hurto 0502), el sistema prioriza el delito más gravoso basándose en:
- Penas legales típicas
- Naturaleza de la violencia involucrada
- Estándares internacionales de clasificación

### 5. Manejo de Delitos Genéricos

Para códigos CNP con descripción "sin descripción" o familia "otros delitos":
1. **Paso 1**: Intentar clasificar usando glosa + familia
2. **Paso 2**: Consultar candidatos top-k por si hay similitud nominal
3. **Paso 3**: Solo clasificar como "NINGUNO" si no hay contexto suficiente

### 6. Retrieval Asimétrico (E5 Query-Passage)

El modelo E5 diferencia entre:
- **Query** (CNP): Texto que busca información
- **Passage** (ICCS): Texto que contiene información

El uso de prefijos `query:` y `passage:` optimiza la similitud semántica según este paradigma.

---

## Solución de Problemas

### Error: "CUDA out of memory"

**Causa**: GPU sin VRAM suficiente.

**Solución**:
```cmd
# Reducir batch size
python preparar_embeddings.py --batch-size 8

# O forzar CPU
python preparar_embeddings.py --device cpu
```

### Error: "No se encuentra matches_detallado.csv"

**Causa**: Fase 3 no ejecutada.

**Solución**: Ejecutar `preparar_embeddings.py` antes de `filtrar_con_llm.py`.

### Error: "Rate limit exceeded" (OpenAI API)

**Causa**: Límite de requests por minuto excedido.

**Solución**: El script reintenta automáticamente con backoff. Si persiste, considerar:
- Añadir delays entre requests
- Aumentar tier de cuenta OpenAI

### Discrepancias altas vs correspondencia manual

**Diagnóstico**:
1. Revisar archivo `comparacion_llm_vs_manual.xlsx`
2. Analizar columna `justificacion` para entender razonamiento del LLM
3. Verificar que exclusiones en `iccs_descripcion.csv` estén correctas

**Ajustes posibles**:
- Modificar prompt en `construir_prompt()` (línea 140-183)
- Ajustar pesos de campos en `preparar_embeddings.py`
- Revisar casos de discrepancia con expertos legales

### Embeddings con baja similitud

**Diagnóstico**:
Revisar `artifacts/matches_reporte.txt` para ver scores del top-1.

**Ajustes**:
```python
# En preparar_embeddings.py, ajustar pesos de campos (líneas ~150-200)
cnp_weights = {
    'glosa': 0.6,        # Incrementar peso de glosa
    'descripcion': 0.3,
    'familia': 0.1
}
```

---

## Referencias

### Documentación Oficial

1. **ICCS (ONU)**: [https://www.unodc.org/unodc/en/data-and-analysis/statistics/iccs.html](https://www.unodc.org/unodc/en/data-and-analysis/statistics/iccs.html)
2. **Modelo E5**: Wang et al. (2022) - "Text Embeddings by Weakly-Supervised Contrastive Pre-training"
3. **GPT-4o-mini**: [https://platform.openai.com/docs/models/gpt-5-mini](https://platform.openai.com/docs/models/gpt-5-mini)

### Modelos y Librerías

- **sentence-transformers**: [https://www.sbert.net/](https://www.sbert.net/)
- **pdfplumber**: [https://github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber)
- **scikit-learn k-NN**: [https://scikit-learn.org/stable/modules/neighbors.html](https://scikit-learn.org/stable/modules/neighbors.html)

### Contacto

**Equipo**: Sección Seguridad Pública y Justicia - Instituto Nacional de Estadísticas (INE), Chile
**Proyecto**: Clasificación Automatizada de Delitos CNP-ICCS
**Fecha**: Diciembre 2025

---

## Notas Finales

Este sistema representa un equilibrio entre automatización y validación experta. Si bien el LLM logra ~88% de exactitud vs etiquetas manuales, **se recomienda revisión por expertos legales** antes de publicar estadísticas oficiales, especialmente en casos de:
- Confianza "baja" o "media"
- Delitos nuevos no presentes en correspondencias manuales anteriores
- Casos donde `iccs_elegido` difiere significativamente de `top1_codigo`

El sistema debe considerarse una **herramienta de asistencia** que acelera el proceso de clasificación, no un reemplazo completo del juicio experto.
