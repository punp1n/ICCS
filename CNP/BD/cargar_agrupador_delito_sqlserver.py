#!/usr/bin/env python3
"""
Carga "agrupador_delito" en cum.cnp_periodo desde los XLSX CNP 2022-2025.

Reglas:
- Lee por periodo el archivo de codificacion dentro de CNP/<anio>_<enero|julio>/
- Usa hoja "Codigos Vigentes" (busqueda flexible) con skiprows=2
- Toma columnas "Codigo Materia" (cum) y "Agrupador de Delitos"
- Crea columna cum.cnp_periodo.agrupador_delito si no existe
- Actualiza por llave (anio, ciclo, cum) via cum.periodo + cum.cnp_periodo
"""

from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from unicodedata import combining, normalize as uni_normalize

import pandas as pd

WINDOWS_BASE_DIR = Path(
    r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\CNP"
)
LOCAL_BASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = WINDOWS_BASE_DIR if WINDOWS_BASE_DIR.exists() else LOCAL_BASE_DIR
BD_DIR = BASE_DIR / "BD"
ENV_FILE = BD_DIR / ".env"
STAGING_DIR = BD_DIR / "staging"

PERIOD_PATTERN = re.compile(r"^(?P<anio>202[2-5])_(?P<mes>enero|julio)$", re.IGNORECASE)
CYCLE_MAP = {"enero": 1, "julio": 2}
TARGET_SHEET = "Codigos Vigentes"

REQUIRED_ENV_KEYS = [
    "SQLSERVER_USER",
    "SQLSERVER_PASSWORD",
    "SQLSERVER_HOST",
    "SQLSERVER_INSTANCE",
    "SQLSERVER_DATABASE",
    "SQLSERVER_DRIVER",
]

TARGET_SCHEMA = "cum"
TARGET_TABLE = "cnp_periodo"
TARGET_COLUMN = "agrupador_delito"
TARGET_COLUMN_SQL_TYPE = "nvarchar(500)"


@dataclass(frozen=True)
class PeriodSource:
    anio: int
    ciclo: int
    folder_name: str
    xlsx_path: Path


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    text = normalize_whitespace(text)
    decomposed = uni_normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not combining(ch))
    return stripped.lower()


def safe_text(value: object) -> str:
    if value is None:
        return ""
    text = normalize_whitespace(str(value))
    if text.lower() == "nan":
        return ""
    return text


def console_text(value: object) -> str:
    # En Windows/cmd, algunos nombres de archivo con tildes en forma combinada
    # fallan al imprimirse con cp1252. Normalizar a NFC evita ese error.
    return uni_normalize("NFC", safe_text(value))


def parse_int_or_none(value: object) -> Optional[int]:
    text = safe_text(value)
    if not text:
        return None
    text = text.replace(" ", "")
    text = re.sub(r"\.0+$", "", text)
    if not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def normalize_code(value: object) -> Optional[int]:
    parsed = parse_int_or_none(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {label}: {path}")


def load_dotenv_file(env_path: Path) -> Dict[str, str]:
    ensure_exists(env_path, "archivo .env")
    values: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, raw = row.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        if raw:
            parsed = shlex.split(raw)
            values[key] = parsed[0] if parsed else ""
        else:
            values[key] = ""

    missing = [k for k in REQUIRED_ENV_KEYS if not values.get(k)]
    if missing:
        raise ValueError(f"Faltan claves en .env: {', '.join(missing)}")
    return values


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str:
    norm_to_col = {normalize_key(col): col for col in df.columns}
    for alias in aliases:
        col = norm_to_col.get(normalize_key(alias))
        if col:
            return col
    raise KeyError(f"No se encontro ninguna columna esperada: {list(aliases)}")


def choose_codificacion_xlsx(period_folder: Path) -> Optional[Path]:
    xlsx_files = sorted([p for p in period_folder.glob("*.xlsx") if not p.name.startswith("~$")])
    if not xlsx_files:
        return None

    non_compare = [p for p in xlsx_files if "compar" not in normalize_key(p.name)]
    if not non_compare:
        return None

    codif_candidates = [p for p in non_compare if "codific" in normalize_key(p.name)]
    chosen = codif_candidates if codif_candidates else non_compare
    chosen = sorted(chosen, key=lambda p: (len(p.name), p.name))
    return chosen[0]


def discover_period_sources() -> Tuple[List[PeriodSource], List[str]]:
    sources: List[PeriodSource] = []
    missing_xlsx: List[str] = []

    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        m = PERIOD_PATTERN.match(folder.name)
        if not m:
            continue
        anio = int(m.group("anio"))
        mes = m.group("mes").lower()
        ciclo = CYCLE_MAP[mes]

        xlsx_path = choose_codificacion_xlsx(folder)
        if xlsx_path is None:
            missing_xlsx.append(folder.name)
            continue

        sources.append(
            PeriodSource(
                anio=anio,
                ciclo=ciclo,
                folder_name=folder.name,
                xlsx_path=xlsx_path,
            )
        )

    sources.sort(key=lambda x: (x.anio, x.ciclo))
    missing_xlsx.sort()
    return sources, missing_xlsx


def read_vigentes_sheet(xlsx_path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, sheet_name=TARGET_SHEET, skiprows=2, dtype=str)
    except ValueError:
        workbook = pd.ExcelFile(xlsx_path)
        candidates = [
            sheet
            for sheet in workbook.sheet_names
            if "codigo" in normalize_key(sheet) and "vigente" in normalize_key(sheet)
        ]
        if not candidates:
            raise RuntimeError(f"No se encontro hoja tipo 'Codigos Vigentes' en {xlsx_path}")
        return pd.read_excel(xlsx_path, sheet_name=candidates[0], skiprows=2, dtype=str)


def extract_agrupador_periodo(source: PeriodSource) -> pd.DataFrame:
    df = read_vigentes_sheet(source.xlsx_path)

    cum_col = find_column(df, ["Codigo Materia", "Cod. Materia", "Cum"])
    agrup_col = find_column(df, ["Agrupador de Delitos", "Agrupador Delitos", "Agrupador"])

    keep = df[[cum_col, agrup_col]].copy()
    keep.columns = ["cum", "agrupador_delito"]
    keep["cum"] = keep["cum"].map(normalize_code)
    keep["agrupador_delito"] = keep["agrupador_delito"].map(safe_text)

    keep = keep[keep["cum"].notna()].copy()
    keep["cum"] = keep["cum"].astype(int)
    keep["agrupador_delito"] = keep["agrupador_delito"].replace("", None)
    keep["anio"] = source.anio
    keep["ciclo"] = source.ciclo
    keep["archivo_origen"] = source.xlsx_path.name

    keep = keep[["anio", "ciclo", "cum", "agrupador_delito", "archivo_origen"]]
    keep = keep.drop_duplicates(subset=["anio", "ciclo", "cum"], keep="last")
    return keep


def build_staging_dataframe(sources: List[PeriodSource]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source in sources:
        period_df = extract_agrupador_periodo(source)
        frames.append(period_df)

    if not frames:
        raise RuntimeError("No se pudo construir staging: no hay periodos/filas para cargar.")

    stage_df = pd.concat(frames, ignore_index=True)
    stage_df = stage_df.drop_duplicates(subset=["anio", "ciclo", "cum"], keep="last")
    stage_df = stage_df.sort_values(["anio", "ciclo", "cum"]).reset_index(drop=True)
    return stage_df


def save_staging(stage_df: pd.DataFrame) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(STAGING_DIR / "staging_agrupador_delito.csv", index=False, encoding="utf-8-sig")


def ensure_pyodbc():
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        if "libodbc" in str(exc).lower():
            raise ImportError(
                "pyodbc esta instalado pero falta runtime ODBC del sistema (libodbc/unixodbc)."
            ) from exc
        raise ImportError("pyodbc no esta instalado. Instala con: pip install pyodbc") from exc
    return pyodbc


def choose_sqlserver_driver(pyodbc, requested_driver: str) -> str:
    available = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if requested_driver in available:
        return requested_driver

    def driver_rank(name: str) -> int:
        m = re.search(r"ODBC Driver\s+(\d+)\s+for SQL Server", name, flags=re.IGNORECASE)
        return int(m.group(1)) if m else -1

    odbc_drivers = sorted([d for d in available if driver_rank(d) >= 0], key=driver_rank, reverse=True)
    if odbc_drivers:
        return odbc_drivers[0]
    if available:
        return available[0]
    raise RuntimeError("No hay drivers ODBC de SQL Server instalados en el sistema.")


def connect_sql_server(env_values: Dict[str, str]):
    pyodbc = ensure_pyodbc()
    requested_driver = env_values["SQLSERVER_DRIVER"]
    selected_driver = choose_sqlserver_driver(pyodbc, requested_driver)
    if selected_driver != requested_driver:
        print(
            f"Aviso: driver solicitado '{requested_driver}' no esta instalado. "
            f"Se usara '{selected_driver}'."
        )

    server = env_values["SQLSERVER_HOST"]
    instance = env_values.get("SQLSERVER_INSTANCE", "").strip()
    if instance:
        server = f"{server}\\{instance}"

    conn_str = (
        f"DRIVER={{{selected_driver}}};"
        f"SERVER={server};"
        f"DATABASE={env_values['SQLSERVER_DATABASE']};"
        f"UID={env_values['SQLSERVER_USER']};"
        f"PWD={env_values['SQLSERVER_PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str, autocommit=False)


def has_table(cursor, schema_name: str, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        schema_name,
        table_name,
    )
    return cursor.fetchone() is not None


def has_column(cursor, schema_name: str, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = ?
        """,
        schema_name,
        table_name,
        column_name,
    )
    return cursor.fetchone() is not None


def ensure_target_column(cursor) -> None:
    if not has_table(cursor, TARGET_SCHEMA, TARGET_TABLE):
        raise RuntimeError(f"No existe tabla {TARGET_SCHEMA}.{TARGET_TABLE}.")

    if not has_column(cursor, TARGET_SCHEMA, TARGET_TABLE, TARGET_COLUMN):
        cursor.execute(
            f"""
            ALTER TABLE {TARGET_SCHEMA}.{TARGET_TABLE}
              ADD {TARGET_COLUMN} {TARGET_COLUMN_SQL_TYPE} NULL;
            """
        )
        return

    cursor.execute(
        """
        SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = ?
        """,
        TARGET_SCHEMA,
        TARGET_TABLE,
        TARGET_COLUMN,
    )
    row = cursor.fetchone()
    if row is None:
        return

    data_type = safe_text(getattr(row, "DATA_TYPE", ""))
    char_len = getattr(row, "CHARACTER_MAXIMUM_LENGTH", None)

    if normalize_key(data_type) != "nvarchar":
        cursor.execute(
            f"""
            ALTER TABLE {TARGET_SCHEMA}.{TARGET_TABLE}
              ALTER COLUMN {TARGET_COLUMN} {TARGET_COLUMN_SQL_TYPE} NULL;
            """
        )
        return

    if isinstance(char_len, int) and char_len > 0 and char_len < 500:
        cursor.execute(
            f"""
            ALTER TABLE {TARGET_SCHEMA}.{TARGET_TABLE}
              ALTER COLUMN {TARGET_COLUMN} {TARGET_COLUMN_SQL_TYPE} NULL;
            """
        )


def load_to_sql_server(env_values: Dict[str, str], stage_df: pd.DataFrame) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    conn = connect_sql_server(env_values)
    try:
        cur = conn.cursor()
        ensure_target_column(cur)

        cur.execute(
            """
            IF OBJECT_ID('tempdb..#agrupador_stage') IS NOT NULL
              DROP TABLE #agrupador_stage;
            CREATE TABLE #agrupador_stage (
              anio smallint NOT NULL,
              ciclo tinyint NOT NULL,
              cum int NOT NULL,
              agrupador_delito nvarchar(500) NULL,
              CONSTRAINT PK_agrupador_stage PRIMARY KEY (anio, ciclo, cum)
            );
            """
        )

        insert_values = [
            (
                int(row.anio),
                int(row.ciclo),
                int(row.cum),
                safe_text(row.agrupador_delito) or None,
            )
            for row in stage_df.itertuples(index=False)
        ]
        cur.fast_executemany = True
        cur.executemany(
            """
            INSERT INTO #agrupador_stage (anio, ciclo, cum, agrupador_delito)
            VALUES (?, ?, ?, ?);
            """,
            insert_values,
        )

        cur.execute(
            f"""
            UPDATE cp
               SET cp.{TARGET_COLUMN} = s.agrupador_delito
            FROM cum.cnp_periodo cp
            INNER JOIN cum.periodo p
               ON p.periodo_id = cp.periodo_id
            INNER JOIN #agrupador_stage s
               ON s.anio = p.anio
              AND s.ciclo = p.ciclo
              AND s.cum = cp.cum;
            """
        )
        updated_rows = int(cur.rowcount if cur.rowcount is not None else 0)

        cur.execute(
            """
            SELECT s.anio, s.ciclo, s.cum, s.agrupador_delito
            FROM #agrupador_stage s
            LEFT JOIN cum.periodo p
              ON p.anio = s.anio AND p.ciclo = s.ciclo
            LEFT JOIN cum.cnp_periodo cp
              ON cp.periodo_id = p.periodo_id AND cp.cum = s.cum
            WHERE cp.cum IS NULL
            ORDER BY s.anio, s.ciclo, s.cum;
            """
        )
        unmatched_rows = [tuple(row) for row in cur.fetchall()]
        unmatched_stage_df = pd.DataFrame(
            unmatched_rows,
            columns=["anio", "ciclo", "cum", "agrupador_delito"],
        )

        cur.execute(
            """
            SELECT p.anio, p.ciclo, cp.cum
            FROM cum.cnp_periodo cp
            INNER JOIN cum.periodo p
              ON p.periodo_id = cp.periodo_id
            INNER JOIN (
                SELECT DISTINCT anio, ciclo
                FROM #agrupador_stage
            ) d
              ON d.anio = p.anio AND d.ciclo = p.ciclo
            LEFT JOIN #agrupador_stage s
              ON s.anio = p.anio AND s.ciclo = p.ciclo AND s.cum = cp.cum
            WHERE s.cum IS NULL
            ORDER BY p.anio, p.ciclo, cp.cum;
            """
        )
        missing_rows = [tuple(row) for row in cur.fetchall()]
        missing_stage_df = pd.DataFrame(missing_rows, columns=["anio", "ciclo", "cum"])

        conn.commit()
        return updated_rows, unmatched_stage_df, missing_stage_df
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def summarize_staging(stage_df: pd.DataFrame, sources: List[PeriodSource], missing_xlsx: List[str]) -> None:
    print(f"Periodos fuente detectados (2022-2025): {len(sources)}")
    for source in sources:
        label = "enero" if source.ciclo == 1 else "julio"
        print(f"  - {source.anio}_{label}: {console_text(source.xlsx_path.name)}")

    if missing_xlsx:
        print("Periodos sin archivo codificacion (omitidos):")
        for period_name in missing_xlsx:
            print(f"  - {period_name}")

    print(f"Filas staging agrupador_delito: {len(stage_df)}")
    by_period = stage_df.groupby(["anio", "ciclo"]).size().reset_index(name="filas")
    print("Filas por periodo:")
    for row in by_period.itertuples(index=False):
        label = "enero" if int(row.ciclo) == 1 else "julio"
        print(f"  - {int(row.anio)}_{label}: {int(row.filas)}")

    null_count = int(stage_df["agrupador_delito"].isna().sum())
    print(f"Filas con agrupador_delito nulo en staging: {null_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Carga agrupador_delito en cum.cnp_periodo desde Excel CNP 2022-2025."
    )
    parser.add_argument(
        "--load-sql",
        action="store_true",
        help="Aplica alter/update en SQL Server. Si no se usa, solo genera staging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources, missing_xlsx = discover_period_sources()
    if not sources:
        raise RuntimeError("No se encontraron periodos 2022-2025 con archivo de codificacion.")

    stage_df = build_staging_dataframe(sources)
    save_staging(stage_df)
    summarize_staging(stage_df, sources, missing_xlsx)

    if args.load_sql:
        env_values = load_dotenv_file(ENV_FILE)
        updated_rows, unmatched_stage_df, missing_stage_df = load_to_sql_server(env_values, stage_df)

        unmatched_path = STAGING_DIR / "staging_agrupador_delito_sin_match_bd.csv"
        missing_path = STAGING_DIR / "staging_agrupador_delito_faltante_en_excel.csv"

        if unmatched_stage_df.empty:
            if unmatched_path.exists():
                unmatched_path.unlink()
        else:
            unmatched_stage_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

        if missing_stage_df.empty:
            if missing_path.exists():
                missing_path.unlink()
        else:
            missing_stage_df.to_csv(missing_path, index=False, encoding="utf-8-sig")

        print(f"Filas actualizadas en cum.cnp_periodo: {updated_rows}")
        print(f"Filas en Excel sin match en BD: {len(unmatched_stage_df)}")
        print(f"Filas en BD (periodos cubiertos) sin match en Excel: {len(missing_stage_df)}")
        if not unmatched_stage_df.empty:
            print(f"Detalle no-matcheado Excel->BD: {unmatched_path}")
        if not missing_stage_df.empty:
            print(f"Detalle no-matcheado BD->Excel: {missing_path}")
        print("Carga SQL finalizada.")
    else:
        print("Modo staging finalizado. Usa --load-sql para aplicar cambios en SQL Server.")


if __name__ == "__main__":
    main()
