# Modelo de Datos — CNP/CUM e ICCS en SQL Server

Documento de referencia del modelo de datos relacional que consolida el Código
Penal Nacional (CNP/CUM) y la Clasificación Internacional de Delitos con Fines
Estadísticos (ICCS), junto a las correspondencias manual y automática entre ambos.

El modelo vive en SQL Server bajo dos esquemas:

- **`cum`**: catálogo nacional (CUM), periodos, familias y vigencia.
- **`cods`**: catálogo ICCS y correspondencias CUM ↔ ICCS.

Scripts y DDL en [CNP/BD/](CNP/BD/). Para el detalle operativo de carga ver
[CNP/BD/README.md](CNP/BD/README.md).

---

## 1. Conceptos clave

| Término | Significado |
|---|---|
| **CUM** | Código Único de Materia: identificador nacional del delito (Código Penal). Entero, "reservado para siempre". |
| **Periodo** | Publicación semestral CAPJ: `(anio, ciclo)` con `ciclo` 1=Enero, 2=Julio. |
| **Familia** | Agrupación amplia del delito (código + glosa), estable entre periodos. |
| **ICCS** | Clasificación internacional UNODC, jerárquica en 4 niveles (2/4/5/6 dígitos). |
| **Correspondencia** | Asignación de un código ICCS a un CUM. Puede ser **manual** (por año) o **automática** (pipeline embeddings + LLM). |
| **Vigencia** | Un CUM está *vigente* en un periodo si aparece en `cum.cnp_periodo` para ese periodo. |

---

## 2. Esquema `cum` (catálogo nacional)

### Tablas

- **`cum.periodo`** — un periodo = `(anio, ciclo)`. PK `periodo_id` (IDENTITY), UNIQUE `(anio, ciclo)`, `ciclo ∈ {1,2}`.
- **`cum.familia_delito`** — familias estables. PK `familia_id` (código), `glosa_familia`.
- **`cum.cnp_hist`** — catálogo *maestro/eterno* de CUM. Solo el código `cum` (PK). Toda otra tabla que referencia un CUM apunta aquí. **No guarda glosa**: las glosas se consultan desde `cnp_periodo` / `cnp_no_vigente` / `vw_cnp_catalogo`.
- **`cum.cnp_periodo`** — CUM **vigentes por periodo** con sus textos. PK compuesta `(periodo_id, cum)`. Contiene `glosa_cum` (CAPJ), `descripcion_delito`, `glosa_ine` (publicación), `familia_id`, y trazabilidad CAPJ↔INE (`glosa_familia_capj`, `fuente_familia`, `agrupador_delito`).
- **`cum.cnp_no_vigente`** — detalle de CUM conocidos **no vigentes** en el último periodo cargado, con motivo y trazabilidad de fuente.

### Vista

- **`cum.vw_cnp_catalogo`** — una fila por CUM de `cnp_hist`, clasificando `estado_vigencia ∈ {vigente, no_vigente, sin_detalle}` y resolviendo glosa/familia/último periodo vigente desde la fuente correspondiente. **Es el punto de entrada recomendado** para consultar el catálogo CUM.

### Relaciones (FK)

```
cum.periodo ──┐
              ├──< cum.cnp_periodo >── cum.familia_delito
cum.cnp_hist ─┤            │
              └──< cum.cnp_no_vigente >── cum.familia_delito
                           └── (ultimo_periodo_id) ── cum.periodo
```

`cnp_periodo.cum`, `cnp_no_vigente.cum` → `cnp_hist.cum`.

---

## 3. Esquema `cods` (ICCS y correspondencias)

### Tablas

- **`cods.iccs_codigo`** — catálogo ICCS **autorreferenciado** (jerarquía). PK `iccs_codigo` (varchar), `nivel ∈ {1,2,3,4}`, `parent_iccs_codigo` → `iccs_codigo` (FK a sí misma), columnas `nivel_1..nivel_4`, metadatos (`glosa_iccs`, `seccion`, `descripcion`, `inclusiones`, `exclusiones`, `notas`, `tiene_metadata`).
- **`cods.iccs_codigo_alias`** — aliases de trazabilidad PDF↔CSV (ej. `1049→1042`, `909→908`). `source_codigo` (PK) → `iccs_codigo`.
- **`cods.cum_iccs_manual`** — correspondencia **manual por año**. PK `(anio, cum)`. Guarda los códigos crudos (`*_raw`), el `estado ∈ {asignado, excluido, sin_dato}` y los códigos normalizados (`iccs_codigo`, `iccs_n1..n4`). FK `cum → cnp_hist`, `iccs_codigo → iccs_codigo`.
- **`cods.cum_iccs_automatica`** — correspondencia **automática global** (salida del pipeline LLM). PK `cum`. Guarda `iccs_codigo`, `confianza`, y los dos mejores candidatos por embeddings (`top1_*`, `top2_*`). `estado ∈ {asignado, sin_match}`. FK a `cnp_hist` y a `iccs_codigo` (incl. top1/top2).
- **`cods.cum_iccs_periodo`** — **resolución final por periodo**. PK `(periodo_id, cum)`. Combina manual + automática y resuelve `fuente_final ∈ {manual, automatica, sin_fuente}` y `estado_final ∈ {asignado, excluido, sin_match, sin_fuente}`. FK compuesta `(periodo_id, cum) → cum.cnp_periodo`.

### Relaciones (FK)

```
cods.iccs_codigo (self-ref por parent_iccs_codigo)
      ▲     ▲     ▲
      │     │     └──< cods.cum_iccs_automatica >── cum.cnp_hist
      │     └────────< cods.cum_iccs_manual     >── cum.cnp_hist
      └──────────────< cods.cum_iccs_periodo    >── cum.cnp_periodo (periodo_id, cum)
```

La tabla `cum_iccs_periodo` es el **puente** entre el modelo nacional (`cum`) y el
internacional (`cods`): cada CUM vigente en un periodo recibe su ICCS final.

---

## 4. Flujo de datos (raw → SQL)

```
DOCX CAPJ por periodo ─┐
Plantilla INE          ├─► cargar_cnp_sqlserver.py ─► staging/ ─► cum.periodo / familia_delito / cnp_hist / cnp_periodo / cnp_no_vigente
Histórico interno      ┘                                            │
XLSX codificación      ──► cargar_agrupador_delito_sqlserver.py ───► cum.cnp_periodo.agrupador_delito

iccs_tabla.csv + iccs_descripcion.csv ─┐
Correspondencia manual (TC_2023/2024)  ├─► cargar_iccs_sqlserver.py ─► staging_iccs/ ─► cods.iccs_codigo (+alias) / cum_iccs_manual / cum_iccs_automatica / cum_iccs_periodo
clasificacion_final.csv (pipeline LLM) ┘
```

Patrón común de los loaders:
1. Validan insumos y generan **staging CSV** (sin tocar la BD) cuando se ejecutan sin `--load-sql`.
2. Con `--load-sql` ejecutan el DDL (`modelo_tablas_*.sql`, idempotente) y cargan.
3. Los loaders detienen la carga si encuentran CUM sin mapeo de familia (`staging_cum_sin_familia.csv`).

### Prioridad de mapeo `familia_id` (en `cargar_cnp_sqlserver.py`)
1. CUM en plantilla INE.
2. CUM en histórico interno.
3. Match textual familia CAPJ vs familia INE.
4. Overrides manuales declarados en el script.

---

## 5. Conexión

Credenciales vía `CNP/BD/.env` (ignorado por git), claves requeridas:
`SQLSERVER_USER`, `SQLSERVER_PASSWORD`, `SQLSERVER_HOST`, `SQLSERVER_INSTANCE`,
`SQLSERVER_DATABASE`, `SQLSERVER_DRIVER` (ej. `ODBC Driver 18 for SQL Server`).

Dependencias: `pandas`, `openpyxl`, `pyodbc` (+ runtime ODBC del sistema).

---

## 6. Consultas de referencia

```sql
-- Catálogo CUM con vigencia resuelta
SELECT * FROM cum.vw_cnp_catalogo ORDER BY cum;

-- ICCS final por CUM en el último periodo cargado
SELECT cp.cum, cp.glosa_ine, ci.estado_final, ci.fuente_final, ci.iccs_codigo
FROM cods.cum_iccs_periodo ci
JOIN cum.cnp_periodo cp ON cp.periodo_id = ci.periodo_id AND cp.cum = ci.cum
WHERE ci.periodo_id = (SELECT MAX(periodo_id) FROM cum.periodo);

-- Discrepancias manual vs automática
SELECT periodo_id, cum, manual_iccs_codigo, auto_iccs_codigo
FROM cods.cum_iccs_periodo
WHERE manual_iccs_codigo IS NOT NULL
  AND auto_iccs_codigo IS NOT NULL
  AND manual_iccs_codigo <> auto_iccs_codigo;
```
