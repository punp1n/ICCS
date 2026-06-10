#!/usr/bin/env python3
"""
Carga CNP (CUM) a SQL Server usando:
- DOCX CAPJ por periodo (2021_enero ... 2025_julio)
- Plantilla INE (glosa/familia oficial)
- Historico manual (cum_2025_enero_interno_AV.xlsx)

Regla de mapeo de familia_id (prioridad):
1) Match directo por CUM en INE
2) Match por CUM en historico manual
3) Match por texto de familia CAPJ contra glosa familia INE
"""

from __future__ import annotations

import argparse
import importlib.util
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

INE_XLSX = BASE_DIR / "PLANTILLA UNICA 2023-2025_CUM.xlsx"
INE_SHEET = "2025-MODIFICADA"
INE_SKIPROWS = 1

HIST_XLSX = BASE_DIR / "cum_2025_enero_interno_AV.xlsx"
HIST_SHEET = "adaptado_2025_1s"

CUM_2025_JULIO_XLSX = (
    BASE_DIR.parent.parent / "CUM y Reg" / "CUM 2025" / "cum_2025_julio.xlsx"
)
CUM_2025_JULIO_SHEET = "cum_2025_julio_adaptado"

ENV_FILE = BD_DIR / ".env"
SQL_SCHEMA_FILE = BD_DIR / "modelo_tablas_cum_cod.sql"
STAGING_DIR = BD_DIR / "staging"

PERIOD_PATTERN = re.compile(r"^(?P<anio>\d{4})_(?P<mes>enero|julio)$", re.IGNORECASE)
CYCLE_MAP = {"enero": 1, "julio": 2}

REQUIRED_ENV_KEYS = [
    "SQLSERVER_USER",
    "SQLSERVER_PASSWORD",
    "SQLSERVER_HOST",
    "SQLSERVER_INSTANCE",
    "SQLSERVER_DATABASE",
    "SQLSERVER_DRIVER",
]

SRC_INE = "ine_cum"
SRC_HIST = "historico_ene2025"
SRC_CAPJ_FAM = "capj_familia_texto"
SRC_MANUAL = "manual_override"

MANUAL_CUM_OVERRIDES = {
    "22300": {
        "familia_id": 22,
        "glosa_ine": "INFRACCIÓN A OTROS TEXTOS LEGALES",
        "fuente": SRC_MANUAL,
    }
}


@dataclass(frozen=True)
class Periodo:
    anio: int
    ciclo: int
    nombre: str
    docx_path: Path
    xlsx_path: Optional[Path]


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


def parse_int_or_none(value: object) -> Optional[int]:
    text = safe_text(value)
    if not text:
        return None
    text = text.replace(" ", "")
    text = re.sub(r"\.0+$", "", text)
    if not re.fullmatch(r"-?\d+", text):
        return None
    return int(text)


def normalize_code(value: object) -> Optional[str]:
    parsed = parse_int_or_none(value)
    if parsed is None:
        return None
    if parsed < 0:
        return None
    return str(parsed)


def load_ine_catalog() -> Tuple[pd.DataFrame, Dict[str, dict], Dict[int, str], Dict[str, int]]:
    ensure_exists(INE_XLSX, "plantilla INE")
    df = pd.read_excel(INE_XLSX, sheet_name=INE_SHEET, skiprows=INE_SKIPROWS, dtype=str)

    cum_col = find_column(df, ["Codigo Materia", "Código Materia"])
    glosa_ine_col = find_column(df, ["Glosa corregida a 2025-1s"])
    familia_id_col = find_column(df, ["Familia_1"])
    familia_ine_col = find_column(df, ["Familia_2"])

    keep = df[[cum_col, glosa_ine_col, familia_id_col, familia_ine_col]].copy()
    keep.columns = ["cum", "glosa_ine", "familia_id", "familia_ine"]
    for col in keep.columns:
        keep[col] = keep[col].map(safe_text)

    keep = keep[keep["cum"] != ""].copy()
    keep["cum"] = keep["cum"].str.replace(r"\.0$", "", regex=True)
    keep["familia_id"] = keep["familia_id"].str.replace(r"\.0$", "", regex=True)

    invalid_cum = keep[~keep["cum"].str.fullmatch(r"\d+")]
    if not invalid_cum.empty:
        raise ValueError(f"INE trae CUM no numericos: {invalid_cum['cum'].head(10).tolist()}")

    invalid_family = keep[~keep["familia_id"].str.fullmatch(r"\d+")]
    if not invalid_family.empty:
        raise ValueError(
            f"INE trae familia_id no numericos: {invalid_family['familia_id'].head(10).tolist()}"
        )

    keep["familia_id"] = keep["familia_id"].astype(int)
    keep["cum"] = keep["cum"].astype(str)

    dedup = keep.sort_values(["cum"]).drop_duplicates(subset=["cum"], keep="last")
    ine_by_cum = {
        row.cum: {
            "glosa_ine": row.glosa_ine,
            "familia_id": int(row.familia_id),
            "familia_ine": row.familia_ine,
        }
        for row in dedup.itertuples(index=False)
    }

    def choose_canonical_family_name(names: pd.Series) -> str:
        cleaned = [safe_text(x) for x in names.tolist() if safe_text(x)]
        if not cleaned:
            return ""
        counts = pd.Series(cleaned).value_counts()
        top_count = int(counts.max())
        top_names = [name for name, c in counts.items() if int(c) == top_count]
        top_names.sort(key=lambda x: (-len(x), x))
        return top_names[0]

    familias_df = (
        keep.groupby("familia_id", as_index=False)["familia_ine"]
        .apply(choose_canonical_family_name)
        .sort_values("familia_id")
        .reset_index(drop=True)
    )
    familias_df.columns = ["familia_id", "familia_ine"]

    familia_ine_by_id: Dict[int, str] = {}
    for row in familias_df.itertuples(index=False):
        if row.familia_id not in familia_ine_by_id:
            familia_ine_by_id[int(row.familia_id)] = safe_text(row.familia_ine)

    familia_name_to_id: Dict[str, int] = {}
    for row in keep[["familia_id", "familia_ine"]].drop_duplicates().itertuples(index=False):
        key = normalize_key(row.familia_ine)
        if key:
            familia_name_to_id[key] = int(row.familia_id)
    for fid, fam_name in familia_ine_by_id.items():
        key = normalize_key(fam_name)
        if key:
            familia_name_to_id[key] = fid

    return familias_df, ine_by_cum, familia_ine_by_id, familia_name_to_id


def load_hist_mapping() -> Tuple[Dict[str, int], Dict[str, str], Dict[int, str], pd.DataFrame]:
    ensure_exists(HIST_XLSX, "historico CUM interno")
    df = pd.read_excel(HIST_XLSX, sheet_name=HIST_SHEET, dtype=str)

    cum_col = find_column(df, ["cum", "Codigo Materia", "Código Materia"])
    familia_id_col = find_column(df, ["familia_id", "Familia_1"])

    glosa_ine_col: Optional[str] = None
    for candidate in ["glosa_cum", "Glosa corregida a 2025-1s"]:
        try:
            glosa_ine_col = find_column(df, [candidate])
            break
        except KeyError:
            continue

    familia_ine_col: Optional[str] = None
    for candidate in ["Familia_INE", "Familia_2"]:
        try:
            familia_ine_col = find_column(df, [candidate])
            break
        except KeyError:
            continue

    anio_vigente_col: Optional[str] = None
    for candidate in ["anio_vigente", "Año vigente", "anio vigente"]:
        try:
            anio_vigente_col = find_column(df, [candidate])
            break
        except KeyError:
            continue

    cols = [cum_col, familia_id_col]
    if glosa_ine_col:
        cols.append(glosa_ine_col)
    if familia_ine_col:
        cols.append(familia_ine_col)
    if anio_vigente_col:
        cols.append(anio_vigente_col)

    hist = df[cols].copy()
    hist.columns = (
        ["cum", "familia_id"]
        + (["glosa_ine_hist"] if glosa_ine_col else [])
        + (["familia_ine_hist"] if familia_ine_col else [])
        + (["anio_vigente"] if anio_vigente_col else [])
    )
    for col in hist.columns:
        hist[col] = hist[col].map(safe_text)

    hist = hist[hist["cum"] != ""].copy()
    hist["cum"] = hist["cum"].map(normalize_code)
    hist["familia_id"] = hist["familia_id"].map(parse_int_or_none)
    hist = hist[hist["cum"].notna()].copy()
    hist = hist[hist["familia_id"].notna()].copy()
    hist["familia_id"] = hist["familia_id"].astype(int)

    hist = hist.drop_duplicates(subset=["cum"], keep="last")
    family_by_cum = {row.cum: int(row.familia_id) for row in hist.itertuples(index=False)}

    glosa_by_cum: Dict[str, str] = {}
    if "glosa_ine_hist" in hist.columns:
        for row in hist.itertuples(index=False):
            glosa_by_cum[row.cum] = safe_text(getattr(row, "glosa_ine_hist", ""))

    family_name_by_id: Dict[int, str] = {}
    if "familia_ine_hist" in hist.columns:
        for row in hist.itertuples(index=False):
            family_name = safe_text(getattr(row, "familia_ine_hist", ""))
            if not family_name:
                continue
            fid = int(row.familia_id)
            existing = family_name_by_id.get(fid, "")
            if not existing or len(family_name) > len(existing):
                family_name_by_id[fid] = family_name

    return family_by_cum, glosa_by_cum, family_name_by_id, hist


def load_parser_module():
    parser_script = BASE_DIR / "procesar_consolidado.py"
    ensure_exists(parser_script, "script parser procesar_consolidado.py")
    spec = importlib.util.spec_from_file_location("procesar_consolidado", parser_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar procesar_consolidado.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_codificacion_xlsx(period_folder: Path) -> Optional[Path]:
    xlsx_files = sorted(period_folder.glob("*.xlsx"))
    if not xlsx_files:
        return None

    candidates = [p for p in xlsx_files if "compar" not in normalize_key(p.name)]
    if not candidates:
        return None

    codif_candidates = [p for p in candidates if "codific" in normalize_key(p.name)]
    chosen = codif_candidates if codif_candidates else candidates
    chosen = sorted(chosen, key=lambda p: (len(p.name), p.name))
    return chosen[0]


def extract_vigentes_codes(xlsx_path: Path) -> set[str]:
    try:
        workbook = pd.read_excel(xlsx_path, sheet_name=None, header=None, dtype=str)
    except Exception:
        return set()

    vig_sheet_name: Optional[str] = None
    for sheet_name in workbook.keys():
        key = normalize_key(sheet_name)
        if "vigente" in key and "no vigente" not in key:
            vig_sheet_name = sheet_name
            break

    if not vig_sheet_name:
        return set()

    df = workbook[vig_sheet_name].fillna("")

    header_row: Optional[int] = None
    header_col: Optional[int] = None
    for ridx in range(len(df)):
        row = df.iloc[ridx]
        for cidx, value in enumerate(row):
            text = normalize_key(value)
            if "codigo materia" in text or "cod. materia" in text or "cod materia" in text:
                header_row = ridx
                header_col = cidx
                break
        if header_col is not None:
            break

    start_row = (header_row + 1) if header_row is not None else 0
    candidate_cols = [header_col] if header_col is not None else [0, 1]

    codes: set[str] = set()
    for cidx in candidate_cols:
        if cidx is None or cidx >= df.shape[1]:
            continue
        for value in df.iloc[start_row:, cidx].tolist():
            code = normalize_code(value)
            if code is not None:
                codes.add(code)
    return codes


def discover_periods() -> List[Periodo]:
    periods: List[Periodo] = []
    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue
        m = PERIOD_PATTERN.match(folder.name)
        if not m:
            continue
        mes = m.group("mes").lower()
        cycle = CYCLE_MAP[mes]
        year = int(m.group("anio"))
        docx_files = sorted(folder.glob("*.docx"))
        if not docx_files:
            continue
        periods.append(
            Periodo(
                anio=year,
                ciclo=cycle,
                nombre=folder.name,
                docx_path=docx_files[0],
                xlsx_path=choose_codificacion_xlsx(folder),
            )
        )

    periods.sort(key=lambda p: (p.anio, p.ciclo))
    return periods


def merge_docx_rows(file_rows: List[dict]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for row in file_rows:
        code = safe_text(row.get("codigo", ""))
        if not code:
            continue
        if not re.fullmatch(r"\d+", code):
            continue

        if code not in merged:
            merged[code] = {
                "codigo": code,
                "familia_nombre": safe_text(row.get("familia_nombre", "")),
                "glosa": safe_text(row.get("glosa", "")),
                "descriptions": list(row.get("descriptions", [])),
            }
        else:
            merged[code]["descriptions"].extend(row.get("descriptions", []))
    return merged


def parse_capj_periods(parser_module) -> Tuple[pd.DataFrame, pd.DataFrame]:
    periods = discover_periods()
    if not periods:
        raise RuntimeError("No se encontraron periodos CNP con DOCX.")

    period_rows: List[dict] = []
    cnp_rows: List[dict] = []
    for period in periods:
        file_rows = parser_module.parse_docx(period.docx_path)
        merged = merge_docx_rows(file_rows)
        if period.xlsx_path is not None:
            vigentes = extract_vigentes_codes(period.xlsx_path)
            if vigentes:
                merged = {code: record for code, record in merged.items() if code in vigentes}

        period_rows.append({"anio": period.anio, "ciclo": period.ciclo, "fecha_publicacion": None})

        for code, record in merged.items():
            desc = " ".join([safe_text(x) for x in record.get("descriptions", []) if safe_text(x)]).strip()
            cnp_rows.append(
                {
                    "anio": period.anio,
                    "ciclo": period.ciclo,
                    "cum": code,
                    "glosa_cum": safe_text(record.get("glosa", "")).strip(" .;:-"),
                    "descripcion_delito": desc,
                    "glosa_familia_capj": safe_text(record.get("familia_nombre", "")),
                }
            )

    period_df = pd.DataFrame(period_rows).drop_duplicates().sort_values(["anio", "ciclo"])
    cnp_df = pd.DataFrame(cnp_rows).sort_values(["anio", "ciclo", "cum"]).reset_index(drop=True)
    return period_df, cnp_df


def apply_family_mapping(
    cnp_df: pd.DataFrame,
    ine_by_cum: Dict[str, dict],
    hist_family_by_cum: Dict[str, int],
    hist_glosa_by_cum: Dict[str, str],
    family_name_to_id: Dict[str, int],
    family_ine_by_id: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = cnp_df.copy()
    out["glosa_ine"] = ""
    out["familia_id"] = pd.NA
    out["fuente_familia"] = ""

    unresolved_rows: List[dict] = []

    for idx, row in out.iterrows():
        cum = safe_text(row["cum"])
        glosa_fam_capj = safe_text(row["glosa_familia_capj"])

        familia_id: Optional[int] = None
        glosa_ine = ""
        fuente = ""

        if cum in ine_by_cum:
            familia_id = int(ine_by_cum[cum]["familia_id"])
            glosa_ine = safe_text(ine_by_cum[cum]["glosa_ine"])
            fuente = SRC_INE
        elif cum in hist_family_by_cum:
            familia_id = int(hist_family_by_cum[cum])
            glosa_ine = safe_text(hist_glosa_by_cum.get(cum, ""))
            fuente = SRC_HIST
        else:
            override = MANUAL_CUM_OVERRIDES.get(cum)
            if override:
                familia_id = int(override["familia_id"])
                glosa_ine = safe_text(override.get("glosa_ine", ""))
                fuente = safe_text(override.get("fuente", SRC_MANUAL)) or SRC_MANUAL
            else:
                key = normalize_key(glosa_fam_capj)
                if key and key in family_name_to_id:
                    familia_id = int(family_name_to_id[key])
                    fuente = SRC_CAPJ_FAM

        if familia_id is not None and not glosa_ine:
            glosa_ine = safe_text(family_ine_by_id.get(familia_id, ""))

        if familia_id is None:
            unresolved_rows.append(
                {
                    "anio": int(row["anio"]),
                    "ciclo": int(row["ciclo"]),
                    "cum": cum,
                    "glosa_cum": safe_text(row["glosa_cum"]),
                    "glosa_familia_capj": glosa_fam_capj,
                }
            )
            continue

        out.at[idx, "familia_id"] = int(familia_id)
        out.at[idx, "glosa_ine"] = glosa_ine
        out.at[idx, "fuente_familia"] = fuente

    unresolved_df = pd.DataFrame(unresolved_rows).sort_values(["anio", "ciclo", "cum"]) if unresolved_rows else pd.DataFrame(columns=["anio", "ciclo", "cum", "glosa_cum", "glosa_familia_capj"])

    out = out[out["familia_id"].notna()].copy()
    out["familia_id"] = out["familia_id"].astype(int)
    return out, unresolved_df


def load_no_vigente_source() -> pd.DataFrame:
    ensure_exists(CUM_2025_JULIO_XLSX, "archivo CUM 2025 julio adaptado")
    df = pd.read_excel(CUM_2025_JULIO_XLSX, sheet_name=CUM_2025_JULIO_SHEET, dtype=str)

    cum_col = find_column(df, ["cum", "Codigo Materia", "Código Materia"])
    glosa_col = find_column(df, ["glosa_ine", "Glosa INE", "Glosa"])
    familia_id_col = find_column(df, ["familia_id", "Familia_1"])
    familia_col = find_column(df, ["glosa_familia", "Familia_2", "Familia INE"])
    razon_col = find_column(df, ["razon_agregado", "razón agregado", "razon agregado"])

    keep = df[[cum_col, glosa_col, familia_id_col, familia_col, razon_col]].copy()
    keep.columns = ["cum", "glosa_ine", "familia_id", "glosa_familia", "razon_agregado"]
    for col in keep.columns:
        keep[col] = keep[col].map(safe_text)

    keep = keep[keep["cum"] != ""].copy()
    keep["cum"] = keep["cum"].map(normalize_code)
    keep = keep[keep["cum"].notna()].copy()
    keep["familia_id"] = keep["familia_id"].map(parse_int_or_none)
    keep = keep.drop_duplicates(subset=["cum"], keep="last").reset_index(drop=True)
    return keep


def get_latest_period_key(cnp_df: pd.DataFrame) -> Tuple[int, int]:
    if cnp_df.empty:
        raise RuntimeError("No hay filas CNP para determinar el ultimo periodo vigente.")
    periods = cnp_df[["anio", "ciclo"]].drop_duplicates().copy()
    periods["anio"] = periods["anio"].astype(int)
    periods["ciclo"] = periods["ciclo"].astype(int)
    last = periods.sort_values(["anio", "ciclo"]).iloc[-1]
    return int(last["anio"]), int(last["ciclo"])


def build_no_vigente_df(
    cnp_periodo_df: pd.DataFrame,
    source_df: pd.DataFrame,
    family_ine_by_id: Dict[int, str],
) -> pd.DataFrame:
    latest_anio, latest_ciclo = get_latest_period_key(cnp_periodo_df)
    current = cnp_periodo_df[
        (cnp_periodo_df["anio"].astype(int) == latest_anio)
        & (cnp_periodo_df["ciclo"].astype(int) == latest_ciclo)
    ].copy()

    current_cums = {safe_text(cum) for cum in current["cum"].tolist() if safe_text(cum)}
    historical_cums = {safe_text(cum) for cum in cnp_periodo_df["cum"].tolist() if safe_text(cum)}
    source_cums = {safe_text(cum) for cum in source_df["cum"].tolist() if safe_text(cum)}
    no_vigente_cums = sorted((source_cums | historical_cums) - current_cums, key=lambda x: int(x))

    source_by_cum = {
        safe_text(row.cum): {
            "glosa_ine": safe_text(row.glosa_ine),
            "familia_id": int(row.familia_id) if pd.notna(row.familia_id) else None,
            "glosa_familia": safe_text(row.glosa_familia),
            "razon_agregado": safe_text(row.razon_agregado),
        }
        for row in source_df.itertuples(index=False)
    }

    latest_hist = cnp_periodo_df.copy()
    latest_hist["anio"] = latest_hist["anio"].astype(int)
    latest_hist["ciclo"] = latest_hist["ciclo"].astype(int)
    latest_hist = (
        latest_hist.sort_values(["cum", "anio", "ciclo"])
        .drop_duplicates(subset=["cum"], keep="last")
        .set_index("cum")
    )

    rows: List[dict] = []
    for cum in no_vigente_cums:
        in_source = cum in source_cums
        in_historical = cum in historical_cums
        source = source_by_cum.get(cum, {})
        hist_row = latest_hist.loc[cum] if in_historical and cum in latest_hist.index else None

        familia_id = source.get("familia_id")
        if familia_id is None and hist_row is not None:
            familia_id = int(hist_row["familia_id"])

        glosa_ine = safe_text(source.get("glosa_ine", ""))
        if not glosa_ine and hist_row is not None:
            glosa_ine = safe_text(hist_row.get("glosa_ine", "")) or safe_text(hist_row.get("glosa_cum", ""))

        glosa_familia = safe_text(source.get("glosa_familia", ""))
        if not glosa_familia and familia_id is not None:
            glosa_familia = safe_text(family_ine_by_id.get(int(familia_id), ""))
        if not glosa_familia and hist_row is not None:
            glosa_familia = safe_text(hist_row.get("glosa_familia_capj", ""))

        if in_source and in_historical:
            motivo = "fuente_2025_e_historico_no_vigente"
        elif in_source:
            motivo = "fuente_2025_no_vigente_capj"
        else:
            motivo = "historico_no_vigente_actual"

        rows.append(
            {
                "cum": cum,
                "glosa_ine": glosa_ine,
                "familia_id": familia_id,
                "glosa_familia": glosa_familia,
                "razon_agregado": safe_text(source.get("razon_agregado", "")),
                "motivo_no_vigente": motivo,
                "presente_fuente_2025_julio": int(in_source),
                "presente_historico_cnp_periodo": int(in_historical),
                "ultimo_periodo_id": "",
                "anio_ultimo_vigente": int(hist_row["anio"]) if hist_row is not None else pd.NA,
                "ciclo_ultimo_vigente": int(hist_row["ciclo"]) if hist_row is not None else pd.NA,
                "archivo_fuente": str(CUM_2025_JULIO_XLSX),
                "hoja_fuente": CUM_2025_JULIO_SHEET,
            }
        )

    return pd.DataFrame(rows)


def extend_familias_with_no_vigentes(
    familias_df: pd.DataFrame,
    no_vigente_df: pd.DataFrame,
) -> pd.DataFrame:
    if no_vigente_df.empty:
        return familias_df

    out = familias_df.copy()
    existing = {int(row.familia_id) for row in out.itertuples(index=False)}
    extra_rows: List[dict] = []
    for row in no_vigente_df.itertuples(index=False):
        if pd.isna(row.familia_id):
            continue
        familia_id = int(row.familia_id)
        if familia_id in existing:
            continue
        family_name = safe_text(row.glosa_familia)
        if not family_name:
            continue
        extra_rows.append({"familia_id": familia_id, "familia_ine": family_name})
        existing.add(familia_id)

    if extra_rows:
        out = pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)
    return out.sort_values("familia_id").reset_index(drop=True)


def build_cnp_hist(cnp_df: pd.DataFrame, no_vigente_df: pd.DataFrame) -> pd.DataFrame:
    capj_cums = {int(safe_text(cum)) for cum in cnp_df["cum"].tolist() if safe_text(cum)}
    no_vigente_cums = {
        int(safe_text(cum)) for cum in no_vigente_df["cum"].tolist() if safe_text(cum)
    }
    capj_cums = sorted(capj_cums | no_vigente_cums)
    return pd.DataFrame({"cum": capj_cums})


def save_staging_tables(
    period_df: pd.DataFrame,
    familias_df: pd.DataFrame,
    cnp_hist_df: pd.DataFrame,
    cnp_periodo_df: pd.DataFrame,
    no_vigente_df: pd.DataFrame,
    unresolved_df: pd.DataFrame,
) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    period_df.to_csv(STAGING_DIR / "staging_periodo.csv", index=False, encoding="utf-8-sig")

    fam_out = familias_df.copy()
    fam_out.columns = ["familia_id", "glosa_familia"]
    fam_out.to_csv(STAGING_DIR / "staging_familia_delito.csv", index=False, encoding="utf-8-sig")

    cnp_hist_df.to_csv(STAGING_DIR / "staging_cnp_hist.csv", index=False, encoding="utf-8-sig")

    cnp_periodo_df.to_csv(STAGING_DIR / "staging_cnp_periodo.csv", index=False, encoding="utf-8-sig")

    no_vigente_df.to_csv(
        STAGING_DIR / "staging_cnp_no_vigente.csv", index=False, encoding="utf-8-sig"
    )

    if unresolved_df.empty:
        unresolved_path = STAGING_DIR / "staging_cum_sin_familia.csv"
        if unresolved_path.exists():
            unresolved_path.unlink()
    else:
        unresolved_df.to_csv(STAGING_DIR / "staging_cum_sin_familia.csv", index=False, encoding="utf-8-sig")


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
    final_batch = "\n".join(current).strip()
    if final_batch:
        batches.append(final_batch)
    return batches


def apply_schema(cursor) -> None:
    ensure_exists(SQL_SCHEMA_FILE, "modelo SQL")
    sql_text = SQL_SCHEMA_FILE.read_text(encoding="utf-8")
    for batch in split_sql_batches(sql_text):
        cursor.execute(batch)


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
    required = {
        ("cum", "periodo"): ["periodo_id", "anio", "ciclo", "fecha_publicacion"],
        ("cum", "familia_delito"): ["familia_id", "glosa_familia"],
        ("cum", "cnp_hist"): ["cum"],
        (
            "cum",
            "cnp_periodo",
        ): [
            "periodo_id",
            "cum",
            "glosa_cum",
            "descripcion_delito",
            "glosa_ine",
            "familia_id",
            "glosa_familia_capj",
            "fuente_familia",
        ],
        (
            "cum",
            "cnp_no_vigente",
        ): [
            "cum",
            "glosa_ine",
            "familia_id",
            "glosa_familia",
            "razon_agregado",
            "motivo_no_vigente",
            "presente_fuente_2025_julio",
            "presente_historico_cnp_periodo",
            "ultimo_periodo_id",
            "anio_ultimo_vigente",
            "ciclo_ultimo_vigente",
            "archivo_fuente",
            "hoja_fuente",
        ],
    }

    missing_tables: List[str] = []
    missing_columns: List[str] = []
    for (schema_name, table_name), columns in required.items():
        if not has_table(cursor, schema_name, table_name):
            missing_tables.append(f"{schema_name}.{table_name}")
            continue
        for col in columns:
            if not has_column(cursor, schema_name, table_name, col):
                missing_columns.append(f"{schema_name}.{table_name}.{col}")

    if missing_tables or missing_columns:
        parts = []
        if missing_tables:
            parts.append(f"Tablas faltantes: {', '.join(missing_tables)}")
        if missing_columns:
            parts.append(f"Columnas faltantes: {', '.join(missing_columns)}")
        raise RuntimeError("Esquema SQL incompleto. " + " | ".join(parts))


def load_to_sql_server(
    env_values: Dict[str, str],
    period_df: pd.DataFrame,
    familias_df: pd.DataFrame,
    cnp_hist_df: pd.DataFrame,
    cnp_periodo_df: pd.DataFrame,
    no_vigente_df: pd.DataFrame,
) -> None:
    conn = connect_sql_server(env_values)
    try:
        cur = conn.cursor()
        # El script SQL es idempotente, por lo que se aplica siempre para mantener esquema alineado.
        apply_schema(cur)
        validate_schema(cur)

        for row in period_df.itertuples(index=False):
            cur.execute(
                """
                MERGE cum.periodo AS target
                USING (SELECT ? AS anio, ? AS ciclo, ? AS fecha_publicacion) AS src
                ON target.anio = src.anio AND target.ciclo = src.ciclo
                WHEN MATCHED THEN
                    UPDATE SET fecha_publicacion = COALESCE(src.fecha_publicacion, target.fecha_publicacion)
                WHEN NOT MATCHED THEN
                    INSERT (anio, ciclo, fecha_publicacion)
                    VALUES (src.anio, src.ciclo, src.fecha_publicacion);
                """,
                int(row.anio),
                int(row.ciclo),
                None,
            )

        fam_upsert = """
            MERGE cum.familia_delito AS target
            USING (SELECT ? AS familia_id, ? AS glosa_familia) AS src
            ON target.familia_id = src.familia_id
            WHEN MATCHED THEN
                UPDATE SET glosa_familia = src.glosa_familia
            WHEN NOT MATCHED THEN
                INSERT (familia_id, glosa_familia)
                VALUES (src.familia_id, src.glosa_familia);
        """
        fam_insert = familias_df.rename(columns={"familia_ine": "glosa_familia"})[
            ["familia_id", "glosa_familia"]
        ].copy()
        for row in fam_insert.itertuples(index=False):
            cur.execute(fam_upsert, int(row.familia_id), safe_text(row.glosa_familia))

        cur.fast_executemany = True
        cur.executemany(
            """
            MERGE cum.cnp_hist AS target
            USING (SELECT ? AS cum) AS src
            ON target.cum = src.cum
            WHEN NOT MATCHED THEN
                INSERT (cum) VALUES (src.cum);
            """,
            [(int(row.cum),) for row in cnp_hist_df.itertuples(index=False)],
        )

        cur.execute("IF OBJECT_ID('tempdb..#valid_cum_stage') IS NOT NULL DROP TABLE #valid_cum_stage;")
        cur.execute("CREATE TABLE #valid_cum_stage (cum int NOT NULL PRIMARY KEY)")
        cur.fast_executemany = True
        cur.executemany(
            "INSERT INTO #valid_cum_stage (cum) VALUES (?)",
            [(int(row.cum),) for row in cnp_hist_df.itertuples(index=False)],
        )

        cur.execute("SELECT periodo_id, anio, ciclo FROM cum.periodo")
        period_id_map = {(int(r.anio), int(r.ciclo)): int(r.periodo_id) for r in cur.fetchall()}

        period_ids_to_reload = sorted({period_id_map[(int(r.anio), int(r.ciclo))] for r in period_df.itertuples(index=False)})
        cur.execute("IF OBJECT_ID('tempdb..#periodo_reload_stage') IS NOT NULL DROP TABLE #periodo_reload_stage;")
        cur.execute("CREATE TABLE #periodo_reload_stage (periodo_id int NOT NULL PRIMARY KEY)")
        if period_ids_to_reload:
            cur.fast_executemany = True
            cur.executemany(
                "INSERT INTO #periodo_reload_stage (periodo_id) VALUES (?)",
                [(pid,) for pid in period_ids_to_reload],
            )

        has_glosa_fam_capj = has_column(cur, "cum", "cnp_periodo", "glosa_familia_capj")
        has_fuente_familia = has_column(cur, "cum", "cnp_periodo", "fuente_familia")

        stage_cols = [
            "periodo_id",
            "cum",
            "glosa_cum",
            "descripcion_delito",
            "glosa_ine",
            "familia_id",
            "glosa_familia_capj",
            "fuente_familia",
        ]

        insert_values = []
        for row in cnp_periodo_df.itertuples(index=False):
            key = (int(row.anio), int(row.ciclo))
            periodo_id = period_id_map[key]
            current = [
                periodo_id,
                int(row.cum),
                safe_text(row.glosa_cum),
                safe_text(row.descripcion_delito) or None,
                safe_text(row.glosa_ine) or None,
                int(row.familia_id),
                safe_text(row.glosa_familia_capj) or None,
                safe_text(row.fuente_familia) or None,
            ]
            insert_values.append(tuple(current))

        cur.execute("IF OBJECT_ID('tempdb..#cnp_periodo_stage') IS NOT NULL DROP TABLE #cnp_periodo_stage;")
        cur.execute(
            """
            CREATE TABLE #cnp_periodo_stage (
              periodo_id int NOT NULL,
              cum int NOT NULL,
              glosa_cum nvarchar(max) NOT NULL,
              descripcion_delito nvarchar(max) NULL,
              glosa_ine nvarchar(255) NULL,
              familia_id int NOT NULL,
              glosa_familia_capj nvarchar(max) NULL,
              fuente_familia nvarchar(40) NULL,
              CONSTRAINT PK_cnp_periodo_stage PRIMARY KEY (periodo_id, cum)
            )
            """
        )

        # Textos largos en nvarchar(max) fallan con fast_executemany por buffers de pyodbc.
        cur.fast_executemany = False
        cur.executemany(
            f"INSERT INTO #cnp_periodo_stage ({', '.join(stage_cols)}) VALUES ({', '.join(['?'] * len(stage_cols))})",
            insert_values,
        )

        if has_table(cur, "cods", "cum_iccs_periodo"):
            cur.execute(
                """
                DELETE target
                FROM cods.cum_iccs_periodo AS target
                INNER JOIN #periodo_reload_stage AS pr ON pr.periodo_id = target.periodo_id
                LEFT JOIN #cnp_periodo_stage AS stage
                  ON stage.periodo_id = target.periodo_id
                 AND stage.cum = target.cum
                WHERE stage.cum IS NULL
                """
            )

        cur.execute(
            """
            DELETE target
            FROM cum.cnp_periodo AS target
            INNER JOIN #periodo_reload_stage AS pr ON pr.periodo_id = target.periodo_id
            LEFT JOIN #cnp_periodo_stage AS stage
              ON stage.periodo_id = target.periodo_id
             AND stage.cum = target.cum
            WHERE stage.cum IS NULL
            """
        )

        update_assignments = [
            "glosa_cum = src.glosa_cum",
            "descripcion_delito = src.descripcion_delito",
            "glosa_ine = src.glosa_ine",
            "familia_id = src.familia_id",
        ]
        insert_cols = ["periodo_id", "cum", "glosa_cum", "descripcion_delito", "glosa_ine", "familia_id"]
        insert_src = [
            "src.periodo_id",
            "src.cum",
            "src.glosa_cum",
            "src.descripcion_delito",
            "src.glosa_ine",
            "src.familia_id",
        ]
        if has_glosa_fam_capj:
            update_assignments.append("glosa_familia_capj = src.glosa_familia_capj")
            insert_cols.append("glosa_familia_capj")
            insert_src.append("src.glosa_familia_capj")
        if has_fuente_familia:
            update_assignments.append("fuente_familia = src.fuente_familia")
            insert_cols.append("fuente_familia")
            insert_src.append("src.fuente_familia")

        cur.execute(
            f"""
            MERGE cum.cnp_periodo AS target
            USING #cnp_periodo_stage AS src
              ON target.periodo_id = src.periodo_id AND target.cum = src.cum
            WHEN MATCHED THEN
              UPDATE SET {', '.join(update_assignments)}
            WHEN NOT MATCHED THEN
              INSERT ({', '.join(insert_cols)})
              VALUES ({', '.join(insert_src)});
            """
        )

        no_vigente_stage_cols = [
            "cum",
            "glosa_ine",
            "familia_id",
            "glosa_familia",
            "razon_agregado",
            "motivo_no_vigente",
            "presente_fuente_2025_julio",
            "presente_historico_cnp_periodo",
            "ultimo_periodo_id",
            "anio_ultimo_vigente",
            "ciclo_ultimo_vigente",
            "archivo_fuente",
            "hoja_fuente",
        ]
        no_vigente_values = []
        for row in no_vigente_df.itertuples(index=False):
            anio_ultimo = parse_int_or_none(getattr(row, "anio_ultimo_vigente", None))
            ciclo_ultimo = parse_int_or_none(getattr(row, "ciclo_ultimo_vigente", None))
            ultimo_periodo_id = None
            if anio_ultimo is not None and ciclo_ultimo is not None:
                ultimo_periodo_id = period_id_map.get((anio_ultimo, ciclo_ultimo))

            familia_id = parse_int_or_none(getattr(row, "familia_id", None))
            no_vigente_values.append(
                (
                    int(row.cum),
                    safe_text(row.glosa_ine) or None,
                    familia_id,
                    safe_text(row.glosa_familia) or None,
                    safe_text(row.razon_agregado) or None,
                    safe_text(row.motivo_no_vigente),
                    int(row.presente_fuente_2025_julio),
                    int(row.presente_historico_cnp_periodo),
                    ultimo_periodo_id,
                    anio_ultimo,
                    ciclo_ultimo,
                    safe_text(row.archivo_fuente),
                    safe_text(row.hoja_fuente),
                )
            )

        cur.execute("IF OBJECT_ID('tempdb..#cnp_no_vigente_stage') IS NOT NULL DROP TABLE #cnp_no_vigente_stage;")
        cur.execute(
            """
            CREATE TABLE #cnp_no_vigente_stage (
              cum int NOT NULL PRIMARY KEY,
              glosa_ine nvarchar(1000) NULL,
              familia_id int NULL,
              glosa_familia nvarchar(1000) NULL,
              razon_agregado nvarchar(500) NULL,
              motivo_no_vigente nvarchar(60) NOT NULL,
              presente_fuente_2025_julio bit NOT NULL,
              presente_historico_cnp_periodo bit NOT NULL,
              ultimo_periodo_id int NULL,
              anio_ultimo_vigente smallint NULL,
              ciclo_ultimo_vigente tinyint NULL,
              archivo_fuente nvarchar(500) NOT NULL,
              hoja_fuente nvarchar(128) NOT NULL
            )
            """
        )
        if no_vigente_values:
            cur.fast_executemany = False
            cur.executemany(
                f"INSERT INTO #cnp_no_vigente_stage ({', '.join(no_vigente_stage_cols)}) VALUES ({', '.join(['?'] * len(no_vigente_stage_cols))})",
                no_vigente_values,
            )

        cur.execute(
            """
            DELETE target
            FROM cum.cnp_no_vigente AS target
            LEFT JOIN #cnp_no_vigente_stage AS stage
              ON stage.cum = target.cum
            WHERE stage.cum IS NULL
            """
        )

        cur.execute(
            """
            MERGE cum.cnp_no_vigente AS target
            USING #cnp_no_vigente_stage AS src
              ON target.cum = src.cum
            WHEN MATCHED THEN
              UPDATE SET
                glosa_ine = src.glosa_ine,
                familia_id = src.familia_id,
                glosa_familia = src.glosa_familia,
                razon_agregado = src.razon_agregado,
                motivo_no_vigente = src.motivo_no_vigente,
                presente_fuente_2025_julio = src.presente_fuente_2025_julio,
                presente_historico_cnp_periodo = src.presente_historico_cnp_periodo,
                ultimo_periodo_id = src.ultimo_periodo_id,
                anio_ultimo_vigente = src.anio_ultimo_vigente,
                ciclo_ultimo_vigente = src.ciclo_ultimo_vigente,
                archivo_fuente = src.archivo_fuente,
                hoja_fuente = src.hoja_fuente,
                updated_at = sysdatetime()
            WHEN NOT MATCHED THEN
              INSERT (
                cum, glosa_ine, familia_id, glosa_familia, razon_agregado,
                motivo_no_vigente, presente_fuente_2025_julio,
                presente_historico_cnp_periodo, ultimo_periodo_id,
                anio_ultimo_vigente, ciclo_ultimo_vigente, archivo_fuente, hoja_fuente
              )
              VALUES (
                src.cum, src.glosa_ine, src.familia_id, src.glosa_familia,
                src.razon_agregado, src.motivo_no_vigente,
                src.presente_fuente_2025_julio, src.presente_historico_cnp_periodo,
                src.ultimo_periodo_id, src.anio_ultimo_vigente,
                src.ciclo_ultimo_vigente, src.archivo_fuente, src.hoja_fuente
              );
            """
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def summarize(
    period_df: pd.DataFrame,
    familias_df: pd.DataFrame,
    cnp_hist_df: pd.DataFrame,
    cnp_periodo_df: pd.DataFrame,
    no_vigente_df: pd.DataFrame,
    unresolved_df: pd.DataFrame,
) -> None:
    print(f"Periodos detectados: {len(period_df)}")
    print(f"Familias INE: {len(familias_df)}")
    print(f"CUM historico catalogo (vigentes + no vigentes): {len(cnp_hist_df)}")
    print(f"Filas CNP por periodo: {len(cnp_periodo_df)}")
    print(f"CUM no vigentes: {len(no_vigente_df)}")

    per_period = cnp_periodo_df.groupby(["anio", "ciclo"]).size().reset_index(name="filas")
    print("Filas por periodo:")
    for row in per_period.itertuples(index=False):
        label = "enero" if int(row.ciclo) == 1 else "julio"
        print(f"  - {int(row.anio)}_{label}: {int(row.filas)}")

    src_counts = cnp_periodo_df["fuente_familia"].value_counts(dropna=False)
    print("Fuente de familia_id:")
    for source, count in src_counts.items():
        print(f"  - {source}: {int(count)}")

    if not no_vigente_df.empty:
        motivo_counts = no_vigente_df["motivo_no_vigente"].value_counts(dropna=False)
        print("Motivo no vigente:")
        for motivo, count in motivo_counts.items():
            print(f"  - {motivo}: {int(count)}")

    if unresolved_df.empty:
        print("CUM sin familia_id: 0")
    else:
        print(f"CUM sin familia_id: {len(unresolved_df)} (ver staging/staging_cum_sin_familia.csv)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carga CNP (CUM) hacia SQL Server.")
    parser.add_argument(
        "--load-sql",
        action="store_true",
        help="Inserta/actualiza datos en SQL Server. Si no se usa, corre solo validacion + staging CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parser_module = load_parser_module()
    period_df, capj_df = parse_capj_periods(parser_module)

    familias_df, ine_by_cum, family_ine_by_id, family_name_to_id = load_ine_catalog()
    hist_family_by_cum, hist_glosa_by_cum, hist_family_name_by_id, _hist_df = load_hist_mapping()

    for family_id, family_name in hist_family_name_by_id.items():
        if family_id not in family_ine_by_id and family_name:
            family_ine_by_id[family_id] = family_name
        key = normalize_key(family_name)
        if key:
            family_name_to_id[key] = family_id

    familias_df = (
        pd.DataFrame(
            sorted(family_ine_by_id.items(), key=lambda item: item[0]),
            columns=["familia_id", "familia_ine"],
        )
        .sort_values("familia_id")
        .reset_index(drop=True)
    )

    cnp_periodo_df, unresolved_df = apply_family_mapping(
        cnp_df=capj_df,
        ine_by_cum=ine_by_cum,
        hist_family_by_cum=hist_family_by_cum,
        hist_glosa_by_cum=hist_glosa_by_cum,
        family_name_to_id=family_name_to_id,
        family_ine_by_id=family_ine_by_id,
    )

    no_vigente_source_df = load_no_vigente_source()
    no_vigente_df = build_no_vigente_df(
        cnp_periodo_df=cnp_periodo_df,
        source_df=no_vigente_source_df,
        family_ine_by_id=family_ine_by_id,
    )
    familias_df = extend_familias_with_no_vigentes(familias_df, no_vigente_df)

    cnp_hist_df = build_cnp_hist(capj_df, no_vigente_df)

    save_staging_tables(
        period_df,
        familias_df,
        cnp_hist_df,
        cnp_periodo_df,
        no_vigente_df,
        unresolved_df,
    )
    summarize(period_df, familias_df, cnp_hist_df, cnp_periodo_df, no_vigente_df, unresolved_df)

    if not unresolved_df.empty:
        raise RuntimeError(
            "Existen CUM sin mapeo de familia_id. Completa staging/staging_cum_sin_familia.csv antes de cargar."
        )

    if args.load_sql:
        env_values = load_dotenv_file(ENV_FILE)
        try:
            load_to_sql_server(
                env_values=env_values,
                period_df=period_df,
                familias_df=familias_df,
                cnp_hist_df=cnp_hist_df,
                cnp_periodo_df=cnp_periodo_df,
                no_vigente_df=no_vigente_df,
            )
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        except Exception as exc:
            raise SystemExit(f"Error durante carga SQL: {exc}") from exc
        print("Carga a SQL Server finalizada correctamente.")
    else:
        print("Ejecucion en modo validacion/staging completada. Usa --load-sql para cargar en SQL Server.")


if __name__ == "__main__":
    main()
