/* ===========================================
   CREACIÓN BASE: esquemas + Periodo + Familia + CUM
   SQL Server
   =========================================== */

SET NOCOUNT ON;

------------------------------------------------
-- 1) SCHEMAS
------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'cum')
  EXEC('CREATE SCHEMA cum');

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'cods')
  EXEC('CREATE SCHEMA cods');

------------------------------------------------
-- 2) PERIODO (enero/julio)
--    Un periodo = (anio, ciclo)
--    ciclo: 1=Enero, 2=Julio
------------------------------------------------
IF OBJECT_ID('cum.periodo', 'U') IS NULL
BEGIN
  CREATE TABLE cum.periodo (
    periodo_id        int IDENTITY(1,1) NOT NULL CONSTRAINT PK_cum_periodo PRIMARY KEY,
    anio              smallint NOT NULL,
    ciclo             tinyint NOT NULL,  -- 1=Enero, 2=Julio
    fecha_publicacion date NULL,
    CONSTRAINT UQ_cum_periodo UNIQUE (anio, ciclo),
    CONSTRAINT CK_cum_periodo_ciclo CHECK (ciclo IN (1,2))
  );
END;

------------------------------------------------
-- 3) FAMILIA (código + glosa) - estable
------------------------------------------------
IF OBJECT_ID('cum.familia_delito', 'U') IS NULL
BEGIN
  CREATE TABLE cum.familia_delito (
    familia_id      int NOT NULL CONSTRAINT PK_cum_familia_delito PRIMARY KEY, -- código familia
    glosa_familia   nvarchar(255) NOT NULL,                                    -- glosa familia
    -- opcional: trazabilidad
    created_at      datetime2(0) NOT NULL CONSTRAINT DF_familia_created DEFAULT (sysdatetime())
  );

  CREATE INDEX IX_familia_glosa ON cum.familia_delito (glosa_familia);
END;

------------------------------------------------
-- 4) CUM “catálogo eterno” (solo el código reservado)
------------------------------------------------
IF OBJECT_ID('cum.cnp_hist', 'U') IS NULL
BEGIN
  CREATE TABLE cum.cnp_hist (
    cum           int NOT NULL CONSTRAINT PK_cum_cnp_hist PRIMARY KEY,  -- código CUM reservado para siempre
    created_at    datetime2(0) NOT NULL CONSTRAINT DF_cnp_hist_created DEFAULT (sysdatetime())
  );
END;

------------------------------------------------
-- 5) CUM por periodo (vigentes + textos + familia)
--    Si un CUM aparece en esta tabla para un periodo, lo consideras vigente.
------------------------------------------------
IF OBJECT_ID('cum.cnp_periodo', 'U') IS NULL
BEGIN
  CREATE TABLE cum.cnp_periodo (
    periodo_id          int NOT NULL,
    cum                int NOT NULL,
    glosa_cum           nvarchar(max) NOT NULL,    -- glosa CAPJ (puede ser extensa)
    descripcion_delito  nvarchar(max) NULL,        -- descripción delito
    glosa_ine           nvarchar(255) NULL,        -- glosa para publicación INE
    familia_id          int NOT NULL,              -- código familia
    created_at          datetime2(0) NOT NULL CONSTRAINT DF_cnp_periodo_created DEFAULT (sysdatetime()),
    CONSTRAINT PK_cnp_periodo PRIMARY KEY (periodo_id, cum),
    CONSTRAINT FK_cnp_periodo_periodo FOREIGN KEY (periodo_id) REFERENCES cum.periodo(periodo_id),
    CONSTRAINT FK_cnp_periodo_cum     FOREIGN KEY (cum)       REFERENCES cum.cnp_hist(cum),
    CONSTRAINT FK_cnp_periodo_familia FOREIGN KEY (familia_id) REFERENCES cum.familia_delito(familia_id)
  );

  CREATE INDEX IX_cnp_periodo_cum ON cum.cnp_periodo (cum);
  CREATE INDEX IX_cnp_periodo_familia ON cum.cnp_periodo (familia_id);
END;

------------------------------------------------
-- 6) TRAZABILIDAD EXTRA (CAPJ vs INE)
--    Columnas opcionales para conservar la familia CAPJ
--    y la fuente usada para mapear familia_id.
------------------------------------------------
IF COL_LENGTH('cum.cnp_periodo', 'glosa_familia_capj') IS NULL
BEGIN
  ALTER TABLE cum.cnp_periodo
    ADD glosa_familia_capj nvarchar(255) NULL;
END;

IF COL_LENGTH('cum.cnp_periodo', 'fuente_familia') IS NULL
BEGIN
  ALTER TABLE cum.cnp_periodo
    ADD fuente_familia nvarchar(40) NULL;
END;

IF COL_LENGTH('cum.cnp_periodo', 'agrupador_delito') IS NULL
BEGIN
  ALTER TABLE cum.cnp_periodo
    ADD agrupador_delito nvarchar(500) NULL;
END;

-- Algunas glosas CAPJ superan 255 caracteres.
-- Asegura capacidad suficiente en bases ya creadas.
IF OBJECT_ID('cum.cnp_periodo', 'U') IS NOT NULL
   AND COL_LENGTH('cum.cnp_periodo', 'glosa_cum') IS NOT NULL
   AND COL_LENGTH('cum.cnp_periodo', 'glosa_cum') < 2000
BEGIN
  ALTER TABLE cum.cnp_periodo
    ALTER COLUMN glosa_cum nvarchar(max) NOT NULL;
END;

-- Algunas familias CAPJ completas tambien superan 255 caracteres.
IF OBJECT_ID('cum.cnp_periodo', 'U') IS NOT NULL
   AND COL_LENGTH('cum.cnp_periodo', 'glosa_familia_capj') IS NOT NULL
   AND COL_LENGTH('cum.cnp_periodo', 'glosa_familia_capj') < 2000
BEGIN
  ALTER TABLE cum.cnp_periodo
    ALTER COLUMN glosa_familia_capj nvarchar(max) NULL;
END;

------------------------------------------------
-- 7) CUM no vigentes
--    Detalle para CUM conocidos que no estan vigentes
--    en el ultimo periodo cargado en cum.cnp_periodo.
------------------------------------------------
IF OBJECT_ID('cum.cnp_no_vigente', 'U') IS NULL
BEGIN
  CREATE TABLE cum.cnp_no_vigente (
    cum                             int            NOT NULL CONSTRAINT PK_cnp_no_vigente PRIMARY KEY,
    glosa_ine                       nvarchar(1000) NULL,
    familia_id                      int            NULL,
    glosa_familia                   nvarchar(1000) NULL,
    razon_agregado                  nvarchar(500)  NULL,
    motivo_no_vigente               nvarchar(60)   NOT NULL,
    presente_fuente_2025_julio      bit            NOT NULL,
    presente_historico_cnp_periodo  bit            NOT NULL,
    ultimo_periodo_id               int            NULL,
    anio_ultimo_vigente             smallint       NULL,
    ciclo_ultimo_vigente            tinyint        NULL,
    archivo_fuente                  nvarchar(500)  NOT NULL,
    hoja_fuente                     nvarchar(128)  NOT NULL,
    loaded_at                       datetime2(0)   NOT NULL CONSTRAINT DF_cnp_no_vigente_loaded DEFAULT (sysdatetime()),
    updated_at                      datetime2(0)   NOT NULL CONSTRAINT DF_cnp_no_vigente_updated DEFAULT (sysdatetime()),
    CONSTRAINT FK_cnp_no_vigente_cum FOREIGN KEY (cum) REFERENCES cum.cnp_hist(cum),
    CONSTRAINT FK_cnp_no_vigente_familia FOREIGN KEY (familia_id) REFERENCES cum.familia_delito(familia_id),
    CONSTRAINT FK_cnp_no_vigente_ultimo_periodo FOREIGN KEY (ultimo_periodo_id) REFERENCES cum.periodo(periodo_id),
    CONSTRAINT CK_cnp_no_vigente_motivo CHECK (
      motivo_no_vigente IN (
        'fuente_2025_no_vigente_capj',
        'historico_no_vigente_actual',
        'fuente_2025_e_historico_no_vigente'
      )
    )
  );
END;

IF OBJECT_ID('cum.cnp_no_vigente', 'U') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1 FROM sys.indexes
     WHERE name = 'IX_cnp_no_vigente_motivo'
       AND object_id = OBJECT_ID('cum.cnp_no_vigente')
   )
BEGIN
  CREATE INDEX IX_cnp_no_vigente_motivo ON cum.cnp_no_vigente (motivo_no_vigente);
END;

IF OBJECT_ID('cum.cnp_no_vigente', 'U') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1 FROM sys.indexes
     WHERE name = 'IX_cnp_no_vigente_familia'
       AND object_id = OBJECT_ID('cum.cnp_no_vigente')
   )
BEGIN
  CREATE INDEX IX_cnp_no_vigente_familia ON cum.cnp_no_vigente (familia_id);
END;

IF OBJECT_ID('cum.cnp_no_vigente', 'U') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1 FROM sys.indexes
     WHERE name = 'IX_cnp_no_vigente_ultimo_periodo'
       AND object_id = OBJECT_ID('cum.cnp_no_vigente')
   )
BEGIN
  CREATE INDEX IX_cnp_no_vigente_ultimo_periodo ON cum.cnp_no_vigente (ultimo_periodo_id);
END;

PRINT 'Listo: esquemas cum/cods + tablas periodo, familia_delito, cnp_hist, cnp_periodo, cnp_no_vigente y vista catalogo.';

GO

CREATE OR ALTER VIEW cum.vw_cnp_catalogo AS
WITH ultimo_periodo AS (
  SELECT TOP (1) periodo_id, anio, ciclo
  FROM cum.periodo
  ORDER BY anio DESC, ciclo DESC
),
vigente_actual AS (
  SELECT cp.cum, cp.glosa_cum, cp.glosa_ine, cp.familia_id, fd.glosa_familia,
         p.anio, p.ciclo
  FROM cum.cnp_periodo cp
  INNER JOIN ultimo_periodo up ON up.periodo_id = cp.periodo_id
  INNER JOIN cum.periodo p ON p.periodo_id = cp.periodo_id
  LEFT JOIN cum.familia_delito fd ON fd.familia_id = cp.familia_id
),
ultimo_vigente AS (
  SELECT cp.cum, p.anio, p.ciclo,
         ROW_NUMBER() OVER (PARTITION BY cp.cum ORDER BY p.anio DESC, p.ciclo DESC) AS rn
  FROM cum.cnp_periodo cp
  INNER JOIN cum.periodo p ON p.periodo_id = cp.periodo_id
)
SELECT
  h.cum,
  CASE
    WHEN va.cum IS NOT NULL THEN 'vigente'
    WHEN nv.cum IS NOT NULL THEN 'no_vigente'
    ELSE 'sin_detalle'
  END AS estado_vigencia,
  CASE
    WHEN va.cum IS NOT NULL THEN COALESCE(NULLIF(va.glosa_ine, ''), va.glosa_cum)
    WHEN nv.cum IS NOT NULL THEN nv.glosa_ine
    ELSE NULL
  END AS glosa,
  CASE
    WHEN va.cum IS NOT NULL THEN va.familia_id
    WHEN nv.cum IS NOT NULL THEN nv.familia_id
    ELSE NULL
  END AS familia_id,
  CASE
    WHEN va.cum IS NOT NULL THEN va.glosa_familia
    WHEN nv.cum IS NOT NULL THEN nv.glosa_familia
    ELSE NULL
  END AS glosa_familia,
  CASE
    WHEN uv.cum IS NOT NULL THEN CONCAT(uv.anio, '_', CASE WHEN uv.ciclo = 1 THEN 'enero' ELSE 'julio' END)
    ELSE NULL
  END AS ultimo_periodo_vigente,
  CASE
    WHEN va.cum IS NOT NULL THEN 'cnp_periodo_actual'
    WHEN nv.cum IS NOT NULL THEN 'cnp_no_vigente'
    ELSE 'sin_detalle'
  END AS fuente_detalle
FROM cum.cnp_hist h
LEFT JOIN vigente_actual va ON va.cum = h.cum
LEFT JOIN cum.cnp_no_vigente nv ON nv.cum = h.cum
LEFT JOIN ultimo_vigente uv ON uv.cum = h.cum AND uv.rn = 1;
