# Filtro LLM - CNP a ICCS

Pipeline de filtrado inteligente que utiliza GPT-4o-mini para elegir el mejor código ICCS entre los top-5 candidatos generados por embeddings.

## Instalación

```bash
cd "C:\Users\asvm2\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\llm_filter"

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Modo Test (recomendado para primera ejecución)

Procesa solo 10 códigos CNP para verificar que todo funciona:

```bash
python filtrar_con_llm.py --test
```

### Procesamiento Completo

Procesa todos los códigos CNP (~565):

```bash
python filtrar_con_llm.py
```

### Opciones Avanzadas

```bash
# Procesar solo los primeros 50 códigos
python filtrar_con_llm.py --limite 50

# Cambiar el número de candidatos (default: 5)
python filtrar_con_llm.py --top-k 3
```

## Flujo de Procesamiento

1. **Carga de datos**:
   - `matches_detallado.csv`: Top-10 candidatos por código CNP
   - `iccs_descripcion.csv`: Información completa de ICCS (incluyendo exclusiones y notas)

2. **Preparación**:
   - Agrupa matches por código CNP
   - Selecciona top-5 por defecto
   - Hace JOIN con ICCS para obtener exclusiones y notas críticas

3. **Procesamiento LLM**:
   - Construye prompt estructurado con toda la información
   - Llama a GPT-4o-mini con temperatura baja (0.1) para consistencia
   - Parsea respuesta JSON con validación

4. **Criterios de Decisión del LLM**:
   - ✅ Precisión sobre especificidad (puede elegir código general si es más exacto)
   - ✅ Considera EXCLUSIONES como filtros críticos
   - ✅ Usa NOTAS para contexto legal
   - ✅ Puede devolver "NINGUNO" si ningún candidato aplica
   - ✅ Score de similitud es orientativo, no determinante

5. **Salidas**:
   - `clasificacion_final.csv`: Compacto (solo columnas esenciales)
   - `clasificacion_con_justificacion.csv`: Completo con razonamiento del LLM
   - `errores.log`: Códigos que fallaron (si los hay)
   - `checkpoint.json`: Checkpoint automático cada 10 códigos (se elimina al terminar)

## Formato de Respuesta del LLM

```json
{
  "iccs_elegido": "101",
  "confianza": "alta",
  "justificacion": "El delito CNP de HOMICIDIO coincide exactamente con la definición ICCS 101 (Homicidio intencional). Las exclusiones no aplican.",
  "exclusiones_aplicadas": []
}
```

## Características

✅ **Checkpoint automático**: Se guarda progreso cada 10 códigos
✅ **Reintentos**: 3 intentos por código con backoff exponencial
✅ **Estimación de costo**: Calcula costo antes de ejecutar
✅ **Validación JSON**: Verifica estructura de respuestas
✅ **Join automático**: Obtiene exclusiones/notas de iccs_descripcion.csv
✅ **Barra de progreso**: Con tqdm para seguimiento visual

## Costos Estimados (GPT-4o-mini)

- **Test (10 códigos)**: ~$0.004 USD
- **Completo (~565 códigos)**: ~$0.24 USD

Costos muy bajos gracias a GPT-4o-mini.

## Estadísticas Generadas

Al finalizar, el script muestra:
- Total procesados
- Cantidad de "NINGUNO" asignados
- Distribución de confianza (alta/media/baja)
- % de coincidencia con top-1 de embeddings

## Troubleshooting

**Error: "No se encuentra matches_detallado.csv"**
→ Ejecuta primero `preparar_embeddings.py` en la carpeta embeddings

**Error de API Key**
→ Verifica que la API key de OpenAI sea válida y tenga créditos

**Rate limit error**
→ El script reintenta automáticamente con backoff exponencial
