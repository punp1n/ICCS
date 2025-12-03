# Plan de Trabajo

Ahora entraremos a una fase de asignar un código internacional a la codificación nacional penal.

**Fuentes de datos:**

*   **Clasificación Nacional:** `C:\Users\asvm2\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\CNP\consolidado_CNP_2025_2021.xlsx`
*   **Clasificación Internacional (ICCS):** `C:\Users\asvm2\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\iccs_parse_final.csv`

**Directorio de trabajo:**

*   `C:\Users\asvm2\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\embeddings`

**Objetivo:**

Generar embeddings para ambas clasificaciones utilizando el modelo `intfloat/multilingual-e5-large`.

**Pasos:**

1.  Preparar los dataframes de ambas clasificaciones.
2.  Agrupar/concatenar los campos de texto relevantes (glosa, definición, inclusiones, etc.) en una sola columna por cada clasificación.
3.  Utilizar el texto resultante para generar los embeddings con el modelo seleccionado.
