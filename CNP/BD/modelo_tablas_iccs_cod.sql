/* ===========================================
   MODELO ICCS + CORRESPONDENCIAS CUM-ICCS
   SQL Server
   =========================================== */

SET NOCOUNT ON;

------------------------------------------------
-- 1) SCHEMA
------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'cods')
  EXEC('CREATE SCHEMA cods');

------------------------------------------------
-- 2) CATALOGO ICCS (todos los niveles)
------------------------------------------------
IF OBJECT_ID('cods.iccs_codigo', 'U') IS NULL
BEGIN
  CREATE TABLE cods.iccs_codigo (
    iccs_codigo        varchar(10)    NOT NULL CONSTRAINT PK_cods_iccs_codigo PRIMARY KEY,
    nivel              tinyint        NOT NULL,
    parent_iccs_codigo varchar(10)    NULL,
    nivel_1            varchar(10)    NOT NULL,
    nivel_2            varchar(10)    NULL,
    nivel_3            varchar(10)    NULL,
    nivel_4            varchar(10)    NULL,
    glosa_iccs         nvarchar(500)  NULL,
    seccion            nvarchar(500)  NULL,
    descripcion        nvarchar(max)  NULL,
    inclusiones        nvarchar(max)  NULL,
    exclusiones        nvarchar(max)  NULL,
    notas              nvarchar(max)  NULL,
    tiene_metadata     bit            NOT NULL CONSTRAINT DF_iccs_tiene_metadata DEFAULT (0),
    created_at         datetime2(0)   NOT NULL CONSTRAINT DF_iccs_created DEFAULT (sysdatetime()),
    CONSTRAINT CK_iccs_nivel CHECK (nivel IN (1,2,3,4))
  );

  CREATE INDEX IX_iccs_parent ON cods.iccs_codigo (parent_iccs_codigo);
  CREATE INDEX IX_iccs_nivel ON cods.iccs_codigo (nivel);
  CREATE INDEX IX_iccs_n1 ON cods.iccs_codigo (nivel_1);
END;

IF COL_LENGTH('cods.iccs_codigo', 'parent_iccs_codigo') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1
     FROM sys.foreign_keys
     WHERE name = 'FK_iccs_parent'
       AND parent_object_id = OBJECT_ID('cods.iccs_codigo')
   )
BEGIN
  ALTER TABLE cods.iccs_codigo
    ADD CONSTRAINT FK_iccs_parent
    FOREIGN KEY (parent_iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo);
END;

------------------------------------------------
-- 3) ALIASES DE CODIGO ICCS (trazabilidad)
------------------------------------------------
IF OBJECT_ID('cods.iccs_codigo_alias', 'U') IS NULL
BEGIN
  CREATE TABLE cods.iccs_codigo_alias (
    source_codigo  varchar(10)   NOT NULL CONSTRAINT PK_cods_iccs_alias PRIMARY KEY,
    iccs_codigo    varchar(10)   NOT NULL,
    alias_tipo     nvarchar(50)  NOT NULL,
    comentario     nvarchar(255) NULL,
    created_at     datetime2(0)  NOT NULL CONSTRAINT DF_iccs_alias_created DEFAULT (sysdatetime()),
    CONSTRAINT FK_iccs_alias_codigo FOREIGN KEY (iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo)
  );
END;

------------------------------------------------
-- 4) CORRESPONDENCIA MANUAL (por año)
------------------------------------------------
IF OBJECT_ID('cods.cum_iccs_manual', 'U') IS NULL
BEGIN
  CREATE TABLE cods.cum_iccs_manual (
    anio             smallint       NOT NULL,
    cum              int            NOT NULL,
    hoja             nvarchar(20)   NOT NULL,
    iccs_n1_raw      nvarchar(100)  NULL,
    iccs_n2_raw      nvarchar(100)  NULL,
    iccs_n3_raw      nvarchar(100)  NULL,
    iccs_n4_raw      nvarchar(100)  NULL,
    iccs_codigo_raw  nvarchar(100)  NULL,
    estado           nvarchar(20)   NOT NULL,
    iccs_codigo      varchar(10)    NULL,
    iccs_n1          varchar(10)    NULL,
    iccs_n2          varchar(10)    NULL,
    iccs_n3          varchar(10)    NULL,
    iccs_n4          varchar(10)    NULL,
    fuente_archivo   nvarchar(260)  NOT NULL,
    loaded_at        datetime2(0)   NOT NULL CONSTRAINT DF_cum_iccs_manual_loaded DEFAULT (sysdatetime()),
    CONSTRAINT PK_cum_iccs_manual PRIMARY KEY (anio, cum),
    CONSTRAINT CK_cum_iccs_manual_estado CHECK (estado IN ('asignado', 'excluido', 'sin_dato')),
    CONSTRAINT FK_cum_iccs_manual_codigo FOREIGN KEY (iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo)
  );

  CREATE INDEX IX_cum_iccs_manual_codigo ON cods.cum_iccs_manual (iccs_codigo);
  CREATE INDEX IX_cum_iccs_manual_estado ON cods.cum_iccs_manual (estado);
END;

IF OBJECT_ID('cods.cum_iccs_manual', 'U') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1
     FROM sys.foreign_keys
     WHERE name = 'FK_cum_iccs_manual_cum'
       AND parent_object_id = OBJECT_ID('cods.cum_iccs_manual')
   )
BEGIN
  ALTER TABLE cods.cum_iccs_manual
    ADD CONSTRAINT FK_cum_iccs_manual_cum
    FOREIGN KEY (cum) REFERENCES cum.cnp_hist(cum);
END;

------------------------------------------------
-- 5) CORRESPONDENCIA AUTOMATICA (global)
------------------------------------------------
IF OBJECT_ID('cods.cum_iccs_automatica', 'U') IS NULL
BEGIN
  CREATE TABLE cods.cum_iccs_automatica (
    cum               int            NOT NULL CONSTRAINT PK_cum_iccs_automatica PRIMARY KEY,
    cnp_glosa         nvarchar(1000) NULL,
    iccs_codigo_raw   nvarchar(100)  NOT NULL,
    estado            nvarchar(20)   NOT NULL,
    iccs_codigo       varchar(10)    NULL,
    iccs_n1           varchar(10)    NULL,
    iccs_n2           varchar(10)    NULL,
    iccs_n3           varchar(10)    NULL,
    iccs_n4           varchar(10)    NULL,
    iccs_glosa_elegida nvarchar(1000) NULL,
    confianza         nvarchar(20)   NULL,
    top1_codigo_raw   nvarchar(100)  NULL,
    top1_codigo       varchar(10)    NULL,
    top1_score        float          NULL,
    top2_codigo_raw   nvarchar(100)  NULL,
    top2_codigo       varchar(10)    NULL,
    top2_score        float          NULL,
    fuente_archivo    nvarchar(260)  NOT NULL,
    loaded_at         datetime2(0)   NOT NULL CONSTRAINT DF_cum_iccs_auto_loaded DEFAULT (sysdatetime()),
    CONSTRAINT CK_cum_iccs_auto_estado CHECK (estado IN ('asignado', 'sin_match')),
    CONSTRAINT FK_cum_iccs_auto_codigo FOREIGN KEY (iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo),
    CONSTRAINT FK_cum_iccs_auto_top1 FOREIGN KEY (top1_codigo) REFERENCES cods.iccs_codigo(iccs_codigo),
    CONSTRAINT FK_cum_iccs_auto_top2 FOREIGN KEY (top2_codigo) REFERENCES cods.iccs_codigo(iccs_codigo)
  );

  CREATE INDEX IX_cum_iccs_auto_codigo ON cods.cum_iccs_automatica (iccs_codigo);
  CREATE INDEX IX_cum_iccs_auto_estado ON cods.cum_iccs_automatica (estado);
END;

IF OBJECT_ID('cods.cum_iccs_automatica', 'U') IS NOT NULL
   AND NOT EXISTS (
     SELECT 1
     FROM sys.foreign_keys
     WHERE name = 'FK_cum_iccs_auto_cum'
       AND parent_object_id = OBJECT_ID('cods.cum_iccs_automatica')
   )
BEGIN
  ALTER TABLE cods.cum_iccs_automatica
    ADD CONSTRAINT FK_cum_iccs_auto_cum
    FOREIGN KEY (cum) REFERENCES cum.cnp_hist(cum);
END;

------------------------------------------------
-- 6) CORRESPONDENCIA FINAL POR PERIODO CUM
------------------------------------------------
IF OBJECT_ID('cods.cum_iccs_periodo', 'U') IS NULL
BEGIN
  CREATE TABLE cods.cum_iccs_periodo (
    periodo_id         int           NOT NULL,
    cum                int           NOT NULL,
    manual_anio        smallint      NULL,
    manual_estado      nvarchar(20)  NULL,
    manual_iccs_codigo varchar(10)   NULL,
    auto_estado        nvarchar(20)  NULL,
    auto_iccs_codigo   varchar(10)   NULL,
    fuente_final       nvarchar(20)  NOT NULL,
    estado_final       nvarchar(20)  NOT NULL,
    iccs_codigo        varchar(10)   NULL,
    iccs_n1            varchar(10)   NULL,
    iccs_n2            varchar(10)   NULL,
    iccs_n3            varchar(10)   NULL,
    iccs_n4            varchar(10)   NULL,
    loaded_at          datetime2(0)  NOT NULL CONSTRAINT DF_cum_iccs_periodo_loaded DEFAULT (sysdatetime()),
    CONSTRAINT PK_cum_iccs_periodo PRIMARY KEY (periodo_id, cum),
    CONSTRAINT CK_cum_iccs_periodo_fuente CHECK (fuente_final IN ('manual', 'automatica', 'sin_fuente')),
    CONSTRAINT CK_cum_iccs_periodo_estado CHECK (estado_final IN ('asignado', 'excluido', 'sin_match', 'sin_fuente')),
    CONSTRAINT FK_cum_iccs_periodo_cnp FOREIGN KEY (periodo_id, cum) REFERENCES cum.cnp_periodo(periodo_id, cum),
    CONSTRAINT FK_cum_iccs_periodo_codigo FOREIGN KEY (iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo),
    CONSTRAINT FK_cum_iccs_periodo_manual_codigo FOREIGN KEY (manual_iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo),
    CONSTRAINT FK_cum_iccs_periodo_auto_codigo FOREIGN KEY (auto_iccs_codigo) REFERENCES cods.iccs_codigo(iccs_codigo)
  );

  CREATE INDEX IX_cum_iccs_periodo_codigo ON cods.cum_iccs_periodo (iccs_codigo);
  CREATE INDEX IX_cum_iccs_periodo_estado ON cods.cum_iccs_periodo (estado_final);
END;

PRINT 'Listo: modelo ICCS/correspondencias en schema cods.';
