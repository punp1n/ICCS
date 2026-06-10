# Carga CNP (CUM) a SQL Server

## Archivos principales
- `modelo_tablas_cum_cod.sql`: crea/ajusta esquema `cum` y tablas base.
- `cargar_cnp_sqlserver.py`: arma staging desde DOCX CAPJ + INE + historico y carga a SQL Server.
- `cargar_agrupador_delito_sqlserver.py`: agrega/actualiza `cum.cnp_periodo.agrupador_delito` desde XLSX de codificacion 2022-2025.
- `modelo_tablas_iccs_cod.sql`: crea/ajusta tablas `cods` para catalogo ICCS y correspondencias CUM-ICCS.
- `cargar_iccs_sqlserver.py`: carga catalogo ICCS + correspondencia manual/automatica + resolucion final por periodo.
- `.env`: credenciales de conexion (ignorado por git).

## Requisitos
```bash
python3 -m pip install pandas openpyxl pyodbc
```

Además, debe estar instalado el driver ODBC de SQL Server configurado en `.env`:
- `ODBC Driver 18 for SQL Server`
- En Linux/WSL, tambien se requiere runtime ODBC del sistema (`libodbc`, paquete `unixodbc`).

## Flujo recomendado
1. Validar y generar staging (sin tocar BD):
```bash
python3 CNP/BD/cargar_cnp_sqlserver.py
```
2. Crear/actualizar modelo y cargar datos:
```bash
python3 CNP/BD/cargar_cnp_sqlserver.py --load-sql
```

3. Cargar agrupador de delitos:
```bash
python3 CNP/BD/cargar_agrupador_delito_sqlserver.py --load-sql
```

4. Cargar ICCS y correspondencias:
```bash
python3 CNP/BD/cargar_iccs_sqlserver.py --load-sql
```

## Salidas de staging
Se generan en `CNP/BD/staging/`:
- `staging_periodo.csv`
- `staging_familia_delito.csv`
- `staging_cnp_hist.csv`
- `staging_cnp_periodo.csv`
- `staging_cnp_no_vigente.csv`
- `staging_cum_sin_familia.csv` (solo si hay CUM sin mapping)
- `staging_agrupador_delito.csv`

## Reglas de mapeo de familia_id
Prioridad usada por el script:
1. CUM en plantilla INE (`PLANTILLA UNICA 2023-2025_CUM.xlsx`)
2. CUM en historico interno (`cum_2025_enero_interno_AV.xlsx`, hoja `adaptado_2025_1s`)
3. Match textual de familia CAPJ vs familia INE (`Familia_2`)

Notas:
- Antes de insertar, el script verifica y actualiza esquema ejecutando `modelo_tablas_cum_cod.sql`.
- Si existe Excel del periodo con hoja `Códigos Vigentes`, se filtran automaticamente los no vigentes.
- `cum.cnp_periodo` se carga exclusivamente desde la extraccion original CAPJ vigente por periodo; no se agregan CUM residuales ni filas de correspondencia manual.
- `cum.cnp_no_vigente` se calcula desde `CUM y Reg/CUM 2025/cum_2025_julio.xlsx`, hoja `cum_2025_julio_adaptado`, contrastando contra el ultimo periodo vigente cargado en `cum.cnp_periodo`.
- `cum.cnp_hist` es el catalogo maestro de CUM conocidos: vigentes, no vigentes y codigos historicos/externos referenciados por otras tablas. Las glosas deben consultarse desde `cum.cnp_periodo`, `cum.cnp_no_vigente` o `cum.vw_cnp_catalogo`.
- `cum.vw_cnp_catalogo` entrega una fila por CUM de `cum.cnp_hist`, clasificando `vigente`, `no_vigente` o `sin_detalle`.

Si queda algun CUM sin mapping, el script detiene la carga y deja el detalle en `staging_cum_sin_familia.csv`.
