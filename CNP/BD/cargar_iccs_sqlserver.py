#!/usr/bin/env python3
"""
Carga ICCS y correspondencias CUM-ICCS (manual + automatica) a SQL Server.

Fuentes:
- ICCS jerarquia: Correspondencia automatica/outputs/iccs_tabla.csv
- ICCS metadatos: Correspondencia automatica/outputs/iccs_descripcion.csv
- Correspondencia manual: Correspondencia manual/2024/04022026_TC_Final_2023-2024_v1.3.xlsx (TC_2023, TC_2024)
- Correspondencia automatica: Correspondencia automatica/llm_filter/outputs/clasificacion_final.csv
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

WINDOWS_BASE_DIR = Path(
    r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS"
)
LOCAL_BASE_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = WINDOWS_BASE_DIR if WINDOWS_BASE_DIR.exists() else LOCAL_BASE_DIR
BD_DIR = BASE_DIR / "CNP" / "BD"

ENV_FILE = BD_DIR / ".env"
SQL_MODEL_FILE = BD_DIR / "modelo_tablas_iccs_cod.sql"
STAGING_DIR = BD_DIR / "staging_iccs"

ICCS_TABLA_CSV = BASE_DIR / "Correspondencia automatica" / "outputs" / "iccs_tabla.csv"
ICCS_DESC_CSV = BASE_DIR / "Correspondencia automatica" / "outputs" / "iccs_descripcion.csv"
AUTO_CSV = BASE_DIR / "Correspondencia automatica" / "llm_filter" / "outputs" / "clasificacion_final.csv"
MANUAL_XLSX = (
    BASE_DIR
    / "Correspondencia manual"
    / "2024"
    / "04022026_TC_Final_2023-2024_v1.3.xlsx"
)
MANUAL_SHEETS = [("TC_2023", 2023), ("TC_2024", 2024)]

REQUIRED_ENV_KEYS = [
    "SQLSERVER_USER",
    "SQLSERVER_PASSWORD",
    "SQLSERVER_HOST",
    "SQLSERVER_INSTANCE",
    "SQLSERVER_DATABASE",
    "SQLSERVER_DRIVER",
]

# Equivalencias declaradas en scripts README (diferencia PDF vs metadatos).
ICCS_ALIAS_MAP = {
    "1049": "1042",
    "909": "908",
}
ICCS_ALIAS_ROWS = [
    ("1049", "1042", "pdf_to_canon", "Equivalencia oficial PDF/CSV parseado"),
    ("909", "908", "pdf_to_canon", "Equivalencia oficial PDF/CSV parseado"),
]


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {label}: {path}")


def safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\xa0", " ").strip()
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def none_if_empty(value: object) -> Optional[object]:
    if value is None:
        return None
    text = safe_text(value)
    if text == "":
        return None
    return text


def parse_int_or_none(value: object) -> Optional[int]:
    text = safe_text(value).replace(" ", "")
    if text == "":
        return None
    text = re.sub(r"\.0+$", "", text)
    if not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def parse_float_or_none(value: object) -> Optional[float]:
    text = safe_text(value)
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_cum(value: object) -> Optional[str]:
    parsed = parse_int_or_none(value)
    if parsed is None:
        return None
    if parsed < 0:
        return None
    return str(parsed)


def normalize_iccs_code(value: object) -> str:
    text = safe_text(value).replace(" ", "")
    if text == "":
        return ""
    upper = text.upper()
    if upper == "NINGUNO":
        return "NINGUNO"
    text = re.sub(r"\.0+$", "", text)
    if not re.fullmatch(r"\d+", text):
        return text
    text = str(int(text))  # quita ceros a la izquierda
    text = ICCS_ALIAS_MAP.get(text, text)
    return text


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
        parsed = shlex.split(raw) if raw else [""]
        values[key] = parsed[0] if parsed else ""

    missing = [k for k in REQUIRED_ENV_KEYS if not values.get(k)]
    if missing:
        raise ValueError(f"Faltan claves en .env: {', '.join(missing)}")
    return values


def ensure_pyodbc():
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise ImportError("pyodbc no esta instalado para este Python.") from exc
    return pyodbc


def choose_sqlserver_driver(pyodbc, requested_driver: str) -> str:
    available = [d for d in pyodbc.drivers() if "SQL Server" in d]
    if requested_driver in available:
        return requested_driver

    def rank(name: str) -> int:
        match = re.search(r"ODBC Driver\s+(\d+)\s+for SQL Server", name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else -1

    ranked = sorted([d for d in available if rank(d) >= 0], key=rank, reverse=True)
    if ranked:
        return ranked[0]
    if available:
        return available[0]
    raise RuntimeError("No hay drivers ODBC de SQL Server instalados.")


def connect_sql_server(env_values: Dict[str, str]):
    pyodbc = ensure_pyodbc()
    req_driver = env_values["SQLSERVER_DRIVER"]
    driver = choose_sqlserver_driver(pyodbc, req_driver)
    if driver != req_driver:
        print(f"Aviso: driver solicitado '{req_driver}' no esta instalado. Se usara '{driver}'.")

    server = env_values["SQLSERVER_HOST"]
    instance = env_values.get("SQLSERVER_INSTANCE", "").strip()
    if instance:
        server = f"{server}\\{instance}"

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={env_values['SQLSERVER_DATABASE']};"
        f"UID={env_values['SQLSERVER_USER']};"
        f"PWD={env_values['SQLSERVER_PASSWORD']};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str, autocommit=False)


def split_sql_batches(sql_text: str) -> List[str]:
    batches: List[str] = []
    current: List[str] = []
    for line in sql_text.splitlines():
        if line.strip().upper() == "GO":
            batch = "\n".join(current).strip()
            if batch:
                batches.append(batch)
            current = []
        else:
            current.append(line)
    last = "\n".join(current).strip()
    if last:
        batches.append(last)
    return batches


def apply_schema(cursor) -> None:
    ensure_exists(SQL_MODEL_FILE, "modelo ICCS SQL")
    sql_text = SQL_MODEL_FILE.read_text(encoding="utf-8")
    for batch in split_sql_batches(sql_text):
        cursor.execute(batch)


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


def validate_schema(cursor) -> None:
    required_tables = [
        ("cods", "iccs_codigo"),
        ("cods", "iccs_codigo_alias"),
        ("cods", "cum_iccs_manual"),
        ("cods", "cum_iccs_automatica"),
        ("cods", "cum_iccs_periodo"),
        ("cum", "periodo"),
        ("cum", "cnp_periodo"),
    ]
    missing = [f"{s}.{t}" for s, t in required_tables if not has_table(cursor, s, t)]
    if missing:
        raise RuntimeError(f"Esquema incompleto, faltan tablas: {', '.join(missing)}")


def build_iccs_catalog() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, str]]]:
    ensure_exists(ICCS_TABLA_CSV, "iccs_tabla.csv")
    ensure_exists(ICCS_DESC_CSV, "iccs_descripcion.csv")

    tab = pd.read_csv(ICCS_TABLA_CSV, dtype=str).fillna("")
    desc = pd.read_csv(ICCS_DESC_CSV, dtype=str).fillna("")

    nodes: Dict[str, Dict[str, object]] = {}

    def ensure_node(
        code: str,
        nivel: int,
        parent: str,
        n1: str,
        n2: str,
        n3: str,
        n4: str,
    ) -> Dict[str, object]:
        if code not in nodes:
            nodes[code] = {
                "iccs_codigo": code,
                "nivel": nivel,
                "parent_iccs_codigo": parent,
                "nivel_1": n1,
                "nivel_2": n2,
                "nivel_3": n3,
                "nivel_4": n4,
                "glosa_iccs": "",
                "seccion": "",
                "descripcion": "",
                "inclusiones": "",
                "exclusiones": "",
                "notas": "",
                "tiene_metadata": 0,
            }
        return nodes[code]

    for row in tab.itertuples(index=False):
        n1 = normalize_iccs_code(getattr(row, "nivel_1", ""))
        n2 = normalize_iccs_code(getattr(row, "nivel_2", ""))
        n3 = normalize_iccs_code(getattr(row, "nivel_3", ""))
        n4 = normalize_iccs_code(getattr(row, "nivel_4", ""))
        seccion = safe_text(getattr(row, "seccion", ""))
        glosa_row = safe_text(getattr(row, "delito_iccs", ""))

        if not n1 or not n1.isdigit():
            continue

        # nivel 1
        n1_node = ensure_node(n1, 1, "", n1, "", "", "")
        if seccion and not safe_text(n1_node["glosa_iccs"]):
            n1_node["glosa_iccs"] = seccion
        if seccion and not safe_text(n1_node["seccion"]):
            n1_node["seccion"] = seccion

        deepest = n1
        nivel = 1
        parent = ""
        if n2:
            deepest = n2
            nivel = 2
            parent = n1
        if n3:
            deepest = n3
            nivel = 3
            parent = n2
        if n4:
            deepest = n4
            nivel = 4
            parent = n3

        node = ensure_node(deepest, nivel, parent, n1, n2, n3, n4)
        if glosa_row:
            node["glosa_iccs"] = glosa_row
        if seccion and not safe_text(node["seccion"]):
            node["seccion"] = seccion

    missing_from_tabla: List[str] = []
    for row in desc.itertuples(index=False):
        code = normalize_iccs_code(getattr(row, "codigo_iccs", ""))
        if not code or code == "NINGUNO":
            continue
        if code not in nodes:
            missing_from_tabla.append(code)
            continue
        node = nodes[code]
        glosa = safe_text(getattr(row, "glosa_iccs", ""))
        seccion = safe_text(getattr(row, "seccion", ""))
        descripcion = safe_text(getattr(row, "descripcion", ""))
        inclusiones = safe_text(getattr(row, "inclusiones", ""))
        exclusiones = safe_text(getattr(row, "exclusiones", ""))
        notas = safe_text(getattr(row, "notas", ""))

        if glosa:
            node["glosa_iccs"] = glosa
        if seccion:
            node["seccion"] = seccion
        node["descripcion"] = descripcion
        node["inclusiones"] = inclusiones
        node["exclusiones"] = exclusiones
        node["notas"] = notas
        node["tiene_metadata"] = 1

    if missing_from_tabla:
        miss = sorted(set(missing_from_tabla))
        raise RuntimeError(
            "Codigos de iccs_descripcion sin nodo en iccs_tabla (tras alias): "
            + ", ".join(miss[:20])
        )

    records = []
    for code, row in nodes.items():
        records.append(
            {
                "iccs_codigo": code,
                "nivel": int(row["nivel"]),
                "parent_iccs_codigo": safe_text(row["parent_iccs_codigo"]),
                "nivel_1": safe_text(row["nivel_1"]),
                "nivel_2": safe_text(row["nivel_2"]),
                "nivel_3": safe_text(row["nivel_3"]),
                "nivel_4": safe_text(row["nivel_4"]),
                "glosa_iccs": safe_text(row["glosa_iccs"]),
                "seccion": safe_text(row["seccion"]),
                "descripcion": safe_text(row["descripcion"]),
                "inclusiones": safe_text(row["inclusiones"]),
                "exclusiones": safe_text(row["exclusiones"]),
                "notas": safe_text(row["notas"]),
                "tiene_metadata": int(row["tiene_metadata"]),
            }
        )

    iccs_df = pd.DataFrame(records)
    iccs_df = iccs_df.sort_values(
        by=["nivel", "nivel_1", "nivel_2", "nivel_3", "nivel_4", "iccs_codigo"],
        kind="stable",
    ).reset_index(drop=True)

    alias_df = pd.DataFrame(
        ICCS_ALIAS_ROWS,
        columns=["source_codigo", "iccs_codigo", "alias_tipo", "comentario"],
    )

    path_by_code: Dict[str, Dict[str, str]] = {}
    for row in iccs_df.itertuples(index=False):
        path_by_code[row.iccs_codigo] = {
            "iccs_n1": safe_text(row.nivel_1),
            "iccs_n2": safe_text(row.nivel_2),
            "iccs_n3": safe_text(row.nivel_3),
            "iccs_n4": safe_text(row.nivel_4),
            "glosa_iccs": safe_text(row.glosa_iccs),
            "seccion": safe_text(row.seccion),
        }

    return iccs_df, alias_df, path_by_code


def load_manual_correspondence(path_by_code: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    ensure_exists(MANUAL_XLSX, "archivo de correspondencia manual")
    rows: List[dict] = []

    for sheet_name, anio in MANUAL_SHEETS:
        df = pd.read_excel(MANUAL_XLSX, sheet_name=sheet_name, skiprows=1, dtype=str).fillna("")
        if len(df.columns) < 26:
            raise RuntimeError(f"La hoja {sheet_name} no tiene al menos 26 columnas (A..Z).")

        sub = df.iloc[:, [0, 22, 23, 24, 25]].copy()
        sub.columns = ["cum", "n1_raw", "n2_raw", "n3_raw", "n4_raw"]
        for col in sub.columns:
            sub[col] = sub[col].map(safe_text)

        for r in sub.itertuples(index=False):
            cum = normalize_cum(r.cum)
            if cum is None:
                continue

            deepest_raw = next((x for x in [r.n4_raw, r.n3_raw, r.n2_raw, r.n1_raw] if x), "")
            deepest_upper = deepest_raw.upper()

            if "EXCLUIDO" in deepest_upper:
                estado = "excluido"
                iccs_codigo = ""
            elif deepest_raw == "":
                estado = "sin_dato"
                iccs_codigo = ""
            else:
                estado = "asignado"
                iccs_codigo = normalize_iccs_code(deepest_raw)
                if not iccs_codigo.isdigit():
                    estado = "sin_dato"
                    iccs_codigo = ""

            iccs_n1 = iccs_n2 = iccs_n3 = iccs_n4 = ""
            if iccs_codigo:
                if iccs_codigo not in path_by_code:
                    raise RuntimeError(
                        f"Codigo ICCS manual no existe en catalogo: {iccs_codigo} (cum={cum}, hoja={sheet_name})"
                    )
                path = path_by_code[iccs_codigo]
                iccs_n1 = path["iccs_n1"]
                iccs_n2 = path["iccs_n2"]
                iccs_n3 = path["iccs_n3"]
                iccs_n4 = path["iccs_n4"]

            rows.append(
                {
                    "anio": int(anio),
                    "cum": int(cum),
                    "hoja": sheet_name,
                    "iccs_n1_raw": r.n1_raw,
                    "iccs_n2_raw": r.n2_raw,
                    "iccs_n3_raw": r.n3_raw,
                    "iccs_n4_raw": r.n4_raw,
                    "iccs_codigo_raw": deepest_raw,
                    "estado": estado,
                    "iccs_codigo": iccs_codigo,
                    "iccs_n1": iccs_n1,
                    "iccs_n2": iccs_n2,
                    "iccs_n3": iccs_n3,
                    "iccs_n4": iccs_n4,
                    "fuente_archivo": str(MANUAL_XLSX),
                }
            )

    manual_df = pd.DataFrame(rows)
    manual_df = manual_df.sort_values(["anio", "cum", "hoja"]).drop_duplicates(["anio", "cum"], keep="last")
    manual_df = manual_df.reset_index(drop=True)
    return manual_df


def load_auto_correspondence(path_by_code: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    ensure_exists(AUTO_CSV, "archivo de correspondencia automatica")
    df = pd.read_csv(AUTO_CSV, dtype=str).fillna("")
    required = [
        "cnp_codigo",
        "cnp_glosa",
        "iccs_elegido",
        "iccs_glosa_elegida",
        "confianza",
        "top1_codigo",
        "top1_score",
        "top2_codigo",
        "top2_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"clasificacion_final.csv sin columnas requeridas: {', '.join(missing)}")

    rows: List[dict] = []
    for r in df.itertuples(index=False):
        cum = normalize_cum(getattr(r, "cnp_codigo", ""))
        if cum is None:
            continue

        iccs_raw = safe_text(getattr(r, "iccs_elegido", ""))
        iccs_norm = normalize_iccs_code(iccs_raw)
        estado = "sin_match"
        iccs_codigo = ""
        if iccs_norm and iccs_norm != "NINGUNO" and iccs_norm.isdigit():
            estado = "asignado"
            iccs_codigo = iccs_norm

        iccs_n1 = iccs_n2 = iccs_n3 = iccs_n4 = ""
        if iccs_codigo:
            if iccs_codigo not in path_by_code:
                raise RuntimeError(
                    f"Codigo ICCS automatico no existe en catalogo: {iccs_codigo} (cum={cum})"
                )
            path = path_by_code[iccs_codigo]
            iccs_n1 = path["iccs_n1"]
            iccs_n2 = path["iccs_n2"]
            iccs_n3 = path["iccs_n3"]
            iccs_n4 = path["iccs_n4"]

        top1_raw = safe_text(getattr(r, "top1_codigo", ""))
        top2_raw = safe_text(getattr(r, "top2_codigo", ""))
        top1_norm = normalize_iccs_code(top1_raw)
        top2_norm = normalize_iccs_code(top2_raw)
        if not top1_norm.isdigit():
            top1_norm = ""
        if not top2_norm.isdigit():
            top2_norm = ""

        rows.append(
            {
                "cum": int(cum),
                "cnp_glosa": safe_text(getattr(r, "cnp_glosa", "")),
                "iccs_codigo_raw": iccs_raw,
                "estado": estado,
                "iccs_codigo": iccs_codigo,
                "iccs_n1": iccs_n1,
                "iccs_n2": iccs_n2,
                "iccs_n3": iccs_n3,
                "iccs_n4": iccs_n4,
                "iccs_glosa_elegida": safe_text(getattr(r, "iccs_glosa_elegida", "")),
                "confianza": safe_text(getattr(r, "confianza", "")),
                "top1_codigo_raw": top1_raw,
                "top1_codigo": top1_norm,
                "top1_score": parse_float_or_none(getattr(r, "top1_score", "")),
                "top2_codigo_raw": top2_raw,
                "top2_codigo": top2_norm,
                "top2_score": parse_float_or_none(getattr(r, "top2_score", "")),
                "fuente_archivo": str(AUTO_CSV),
            }
        )

    auto_df = pd.DataFrame(rows)
    auto_df = auto_df.sort_values(["cum"]).drop_duplicates(["cum"], keep="last").reset_index(drop=True)
    return auto_df


def fetch_cnp_periodo_rows(cursor) -> pd.DataFrame:
    cursor.execute(
        """
        SELECT cp.periodo_id, p.anio, p.ciclo, cp.cum
        FROM cum.cnp_periodo cp
        INNER JOIN cum.periodo p ON p.periodo_id = cp.periodo_id
        ORDER BY p.anio, p.ciclo, cp.cum
        """
    )
    rows = cursor.fetchall()
    return pd.DataFrame(
        [
            {
                "periodo_id": int(r.periodo_id),
                "anio": int(r.anio),
                "ciclo": int(r.ciclo),
                "cum": int(r.cum),
            }
            for r in rows
        ]
    )


def fetch_cnp_hist_codes(cursor) -> set[int]:
    cursor.execute("SELECT cum FROM cum.cnp_hist")
    return {int(r.cum) for r in cursor.fetchall()}


def build_final_mapping(
    cnp_periodo_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    path_by_code: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    manual_map = {
        (int(r.anio), int(r.cum)): {
            "anio": int(r.anio),
            "estado": safe_text(r.estado),
            "iccs_codigo": safe_text(r.iccs_codigo),
        }
        for r in manual_df.itertuples(index=False)
    }
    auto_map = {
        int(r.cum): {
            "estado": safe_text(r.estado),
            "iccs_codigo": safe_text(r.iccs_codigo),
        }
        for r in auto_df.itertuples(index=False)
    }

    out: List[dict] = []
    for r in cnp_periodo_df.itertuples(index=False):
        periodo_id = int(r.periodo_id)
        anio = int(r.anio)
        cum = int(r.cum)

        man = manual_map.get((anio, cum))
        auto = auto_map.get(cum)

        manual_anio = man["anio"] if man else None
        manual_estado = man["estado"] if man else ""
        manual_code = man["iccs_codigo"] if man else ""

        auto_estado = auto["estado"] if auto else ""
        auto_code = auto["iccs_codigo"] if auto else ""

        fuente_final = "sin_fuente"
        estado_final = "sin_fuente"
        final_code = ""

        if man and manual_estado in {"asignado", "excluido"}:
            fuente_final = "manual"
            estado_final = manual_estado
            final_code = manual_code if manual_estado == "asignado" else ""
        else:
            if auto:
                fuente_final = "automatica"
                estado_final = auto_estado or "sin_match"
                final_code = auto_code if estado_final == "asignado" else ""

        iccs_n1 = iccs_n2 = iccs_n3 = iccs_n4 = ""
        if final_code:
            if final_code not in path_by_code:
                raise RuntimeError(
                    f"Codigo ICCS final no existe en catalogo: {final_code} (periodo_id={periodo_id}, cum={cum})"
                )
            path = path_by_code[final_code]
            iccs_n1 = path["iccs_n1"]
            iccs_n2 = path["iccs_n2"]
            iccs_n3 = path["iccs_n3"]
            iccs_n4 = path["iccs_n4"]

        out.append(
            {
                "periodo_id": periodo_id,
                "cum": cum,
                "manual_anio": manual_anio,
                "manual_estado": manual_estado,
                "manual_iccs_codigo": manual_code,
                "auto_estado": auto_estado,
                "auto_iccs_codigo": auto_code,
                "fuente_final": fuente_final,
                "estado_final": estado_final,
                "iccs_codigo": final_code,
                "iccs_n1": iccs_n1,
                "iccs_n2": iccs_n2,
                "iccs_n3": iccs_n3,
                "iccs_n4": iccs_n4,
            }
        )

    return pd.DataFrame(out)


def save_staging(
    iccs_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    iccs_df.to_csv(STAGING_DIR / "staging_iccs_codigo.csv", index=False, encoding="utf-8-sig")
    manual_df.to_csv(STAGING_DIR / "staging_cum_iccs_manual.csv", index=False, encoding="utf-8-sig")
    auto_df.to_csv(STAGING_DIR / "staging_cum_iccs_automatica.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(STAGING_DIR / "staging_cum_iccs_periodo.csv", index=False, encoding="utf-8-sig")


def summarize(
    iccs_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> None:
    print(f"ICCS codigos (todos los niveles): {len(iccs_df)}")
    print(f"ICCS con metadata: {int(iccs_df['tiene_metadata'].sum())}")
    print(f"Correspondencia manual filas: {len(manual_df)}")
    print(f"Correspondencia automatica filas: {len(auto_df)}")
    print(f"Correspondencia final por periodo: {len(final_df)}")

    if not manual_df.empty:
        print("Manual por estado:", manual_df["estado"].value_counts(dropna=False).to_dict())
    if not auto_df.empty:
        print("Automatica por estado:", auto_df["estado"].value_counts(dropna=False).to_dict())
    if not final_df.empty:
        print("Final por estado:", final_df["estado_final"].value_counts(dropna=False).to_dict())
        print("Final por fuente:", final_df["fuente_final"].value_counts(dropna=False).to_dict())


def load_to_sql_server(
    env_values: Dict[str, str],
    iccs_df: pd.DataFrame,
    alias_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    path_by_code: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    conn = connect_sql_server(env_values)
    try:
        cur = conn.cursor()

        apply_schema(cur)
        validate_schema(cur)

        valid_cums = fetch_cnp_hist_codes(cur)
        referenced_cums = sorted(
            set(manual_df["cum"].astype(int).tolist()) | set(auto_df["cum"].astype(int).tolist())
        )
        missing_cums = [cum for cum in referenced_cums if cum not in valid_cums]
        if missing_cums:
            sample = ", ".join(str(cum) for cum in missing_cums[:20])
            suffix = "..." if len(missing_cums) > 20 else ""
            raise RuntimeError(
                "Las correspondencias ICCS referencian CUM inexistentes en cum.cnp_hist: "
                f"{sample}{suffix}"
            )

        cnp_periodo_df = fetch_cnp_periodo_rows(cur)
        if cnp_periodo_df.empty:
            raise RuntimeError("No hay filas en cum.cnp_periodo para construir correspondencia final.")

        final_df = build_final_mapping(cnp_periodo_df, manual_df, auto_df, path_by_code)

        # Refresh completo
        cur.execute("DELETE FROM cods.cum_iccs_periodo")
        cur.execute("DELETE FROM cods.cum_iccs_manual")
        cur.execute("DELETE FROM cods.cum_iccs_automatica")
        cur.execute("DELETE FROM cods.iccs_codigo_alias")
        cur.execute("DELETE FROM cods.iccs_codigo")

        cur.fast_executemany = True
        cur.executemany(
            """
            INSERT INTO cods.iccs_codigo (
              iccs_codigo, nivel, parent_iccs_codigo, nivel_1, nivel_2, nivel_3, nivel_4,
              glosa_iccs, seccion, descripcion, inclusiones, exclusiones, notas, tiene_metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    safe_text(r.iccs_codigo),
                    int(r.nivel),
                    none_if_empty(r.parent_iccs_codigo),
                    safe_text(r.nivel_1),
                    none_if_empty(r.nivel_2),
                    none_if_empty(r.nivel_3),
                    none_if_empty(r.nivel_4),
                    none_if_empty(r.glosa_iccs),
                    none_if_empty(r.seccion),
                    none_if_empty(r.descripcion),
                    none_if_empty(r.inclusiones),
                    none_if_empty(r.exclusiones),
                    none_if_empty(r.notas),
                    int(r.tiene_metadata),
                )
                for r in iccs_df.sort_values(["nivel", "iccs_codigo"]).itertuples(index=False)
            ],
        )

        cur.executemany(
            """
            INSERT INTO cods.iccs_codigo_alias (source_codigo, iccs_codigo, alias_tipo, comentario)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    safe_text(r.source_codigo),
                    safe_text(r.iccs_codigo),
                    safe_text(r.alias_tipo),
                    none_if_empty(r.comentario),
                )
                for r in alias_df.itertuples(index=False)
            ],
        )

        cur.executemany(
            """
            INSERT INTO cods.cum_iccs_manual (
              anio, cum, hoja, iccs_n1_raw, iccs_n2_raw, iccs_n3_raw, iccs_n4_raw,
              iccs_codigo_raw, estado, iccs_codigo, iccs_n1, iccs_n2, iccs_n3, iccs_n4, fuente_archivo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(r.anio),
                    int(r.cum),
                    safe_text(r.hoja),
                    none_if_empty(r.iccs_n1_raw),
                    none_if_empty(r.iccs_n2_raw),
                    none_if_empty(r.iccs_n3_raw),
                    none_if_empty(r.iccs_n4_raw),
                    none_if_empty(r.iccs_codigo_raw),
                    safe_text(r.estado),
                    none_if_empty(r.iccs_codigo),
                    none_if_empty(r.iccs_n1),
                    none_if_empty(r.iccs_n2),
                    none_if_empty(r.iccs_n3),
                    none_if_empty(r.iccs_n4),
                    safe_text(r.fuente_archivo),
                )
                for r in manual_df.itertuples(index=False)
            ],
        )

        cur.executemany(
            """
            INSERT INTO cods.cum_iccs_automatica (
              cum, cnp_glosa, iccs_codigo_raw, estado, iccs_codigo, iccs_n1, iccs_n2, iccs_n3, iccs_n4,
              iccs_glosa_elegida, confianza, top1_codigo_raw, top1_codigo, top1_score,
              top2_codigo_raw, top2_codigo, top2_score, fuente_archivo
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(r.cum),
                    none_if_empty(r.cnp_glosa),
                    safe_text(r.iccs_codigo_raw),
                    safe_text(r.estado),
                    none_if_empty(r.iccs_codigo),
                    none_if_empty(r.iccs_n1),
                    none_if_empty(r.iccs_n2),
                    none_if_empty(r.iccs_n3),
                    none_if_empty(r.iccs_n4),
                    none_if_empty(r.iccs_glosa_elegida),
                    none_if_empty(r.confianza),
                    none_if_empty(r.top1_codigo_raw),
                    none_if_empty(r.top1_codigo),
                    parse_float_or_none(r.top1_score),
                    none_if_empty(r.top2_codigo_raw),
                    none_if_empty(r.top2_codigo),
                    parse_float_or_none(r.top2_score),
                    safe_text(r.fuente_archivo),
                )
                for r in auto_df.itertuples(index=False)
            ],
        )

        cur.executemany(
            """
            INSERT INTO cods.cum_iccs_periodo (
              periodo_id, cum, manual_anio, manual_estado, manual_iccs_codigo,
              auto_estado, auto_iccs_codigo, fuente_final, estado_final, iccs_codigo,
              iccs_n1, iccs_n2, iccs_n3, iccs_n4
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(r.periodo_id),
                    int(r.cum),
                    int(r.manual_anio) if pd.notna(r.manual_anio) and safe_text(r.manual_anio) != "" else None,
                    none_if_empty(r.manual_estado),
                    none_if_empty(r.manual_iccs_codigo),
                    none_if_empty(r.auto_estado),
                    none_if_empty(r.auto_iccs_codigo),
                    safe_text(r.fuente_final),
                    safe_text(r.estado_final),
                    none_if_empty(r.iccs_codigo),
                    none_if_empty(r.iccs_n1),
                    none_if_empty(r.iccs_n2),
                    none_if_empty(r.iccs_n3),
                    none_if_empty(r.iccs_n4),
                )
                for r in final_df.itertuples(index=False)
            ],
        )

        conn.commit()
        return final_df
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carga ICCS + correspondencias a SQL Server.")
    parser.add_argument(
        "--load-sql",
        action="store_true",
        help="Inserta datos en SQL Server (ademas de generar staging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    iccs_df, alias_df, path_by_code = build_iccs_catalog()
    manual_df = load_manual_correspondence(path_by_code)
    auto_df = load_auto_correspondence(path_by_code)

    if args.load_sql:
        env_values = load_dotenv_file(ENV_FILE)
        final_df = load_to_sql_server(
            env_values=env_values,
            iccs_df=iccs_df,
            alias_df=alias_df,
            manual_df=manual_df,
            auto_df=auto_df,
            path_by_code=path_by_code,
        )
        save_staging(iccs_df, manual_df, auto_df, final_df)
        summarize(iccs_df, manual_df, auto_df, final_df)
        print("Carga ICCS/correspondencias finalizada correctamente.")
    else:
        # Preview sin SQL: usa cnp_periodo desde staging CNP si existe.
        cnp_stage = BD_DIR / "staging" / "staging_cnp_periodo.csv"
        if cnp_stage.exists():
            base_df = pd.read_csv(cnp_stage, dtype=str).fillna("")
            base_df["periodo_id"] = (
                base_df[["anio", "ciclo"]]
                .astype(str)
                .agg("_".join, axis=1)
                .astype("category")
                .cat.codes
                + 1
            )
            base_df["anio"] = base_df["anio"].astype(int)
            base_df["ciclo"] = base_df["ciclo"].astype(int)
            base_df["cum"] = base_df["cum"].astype(int)
            final_df = build_final_mapping(
                base_df[["periodo_id", "anio", "ciclo", "cum"]],
                manual_df,
                auto_df,
                path_by_code,
            )
        else:
            final_df = pd.DataFrame(
                columns=[
                    "periodo_id",
                    "cum",
                    "manual_anio",
                    "manual_estado",
                    "manual_iccs_codigo",
                    "auto_estado",
                    "auto_iccs_codigo",
                    "fuente_final",
                    "estado_final",
                    "iccs_codigo",
                    "iccs_n1",
                    "iccs_n2",
                    "iccs_n3",
                    "iccs_n4",
                ]
            )
        save_staging(iccs_df, manual_df, auto_df, final_df)
        summarize(iccs_df, manual_df, auto_df, final_df)
        print("Modo staging completado. Usa --load-sql para cargar en SQL Server.")


if __name__ == "__main__":
    main()
