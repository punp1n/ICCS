# Scripts de Generación de Tablas ICCS

Este directorio contiene los scripts para generar las tablas de clasificación ICCS (International Classification of Crime for Statistical Purposes) en español.

## Archivos

### `generar_iccs_tabla.py`
Genera la tabla ICCS desde el parseo directo del PDF oficial.

**Salida:** `outputs/iccs_tabla.{csv,xlsx,json}`

**Columnas:**
- `nivel_1`: Código nivel 1 (2 dígitos)
- `nivel_2`: Código nivel 2 (4 dígitos)
- `nivel_3`: Código nivel 3 (5 dígitos, opcional)
- `nivel_4`: Código nivel 4 (6 dígitos, opcional)
- `delito_iccs`: Descripción corta del delito desde el PDF
- `seccion`: Nombre de la sección ICCS

**Total de registros:** 309

---

### `generar_iccs_descripcion.py`
Genera la tabla ICCS con metadata completa desde los archivos CSV parseados (`parse_defs/`), enriquecida con la columna `seccion` desde el PDF.

**Salida:** `outputs/iccs_descripcion.{csv,xlsx,json}`

**Columnas:**
- `codigo_iccs`: Código ICCS completo
- `glosa_iccs`: Glosa oficial del delito
- `seccion`: Nombre de la sección ICCS (desde PDF)
- `descripcion`: Descripción detallada
- `inclusiones`: Tipos de delitos incluidos
- `exclusiones`: Tipos de delitos excluidos
- `notas`: Notas adicionales

**Total de registros:** 309
**Nota:** 2 códigos tienen sección asignada manualmente porque difieren entre PDF y CSV:
- Código `1042`: "Otros actos de suicidio asistido o incitación al suicidio" (en PDF aparece como `1049`)
  - Sección asignada: "Actos que causan la muerte o que tienen la intencion de causar la muerte"
- Código `908`: "Otros actos contra la seguridad pública y la seguridad del Estado" (en PDF aparece como `909`)
  - Sección asignada: "Actos contra la seguridad publica y la seguridad del Estado"

---

## Por qué dos archivos separados

Los códigos ICCS en el PDF oficial difieren ligeramente de los códigos en los archivos CSV parseados. Ejemplos:
- PDF tiene código `1049`, CSV tiene `1042`
- PDF tiene código `909`, CSV tiene `908`

Por esta razón, mantenemos ambos archivos separados:
- **iccs_tabla**: refleja exactamente el PDF oficial
- **iccs_descripcion**: contiene la metadata completa desde los CSVs

## Uso

Para generar ambos archivos:

```bash
# Ejecutar con Python de Spyder
cd scripts
python generar_iccs_tabla.py
python generar_iccs_descripcion.py
```

Los archivos de salida se generarán en la carpeta `outputs/`.

## Archivos de respaldo

Los archivos anteriores se encuentran en la carpeta `archive/`.
