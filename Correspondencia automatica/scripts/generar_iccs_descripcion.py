from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber


PDF_PATH = Path(
    r"C:\\Users\\Asvaldebenitom\\OneDrive - Instituto Nacional de Estadisticas\\Seguridad y justicia\\ICCS\\ICSS_PDF\\ICCS_SPANISH_2016_web.pdf"
)
BASE_DIR = Path(
    r"C:\\Users\\Asvaldebenitom\\OneDrive - Instituto Nacional de Estadisticas\\Seguridad y justicia\\ICCS\\Correspondencia automatica"
)
PARSE_DEFS_DIR = BASE_DIR / "parse_defs"
OUTPUT_DIR = BASE_DIR / "outputs"
START_PAGE_INDEX = 25  # pagina 26 (0-indexado)
END_PAGE_INDEX = 33  # pagina 34 (0-indexado)
MAX_SECTION = 11
OUTPUT_BASENAME = "iccs_descripcion"

SECTION_PATTERN = re.compile(r"Secci\u00f3n\s+(\d+)\s+(.*)", re.IGNORECASE)
CODE_PATTERN = re.compile(r"^(\d{4,6})\s*(.*)$")


def _normalize_text(parts: List[str]) -> str:
    raw = " ".join(p.strip() for p in parts if p and p.strip())
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", raw).strip()


def _load_pdf_lines() -> List[str]:
    lines: List[str] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for idx in range(START_PAGE_INDEX, END_PAGE_INDEX + 1):
            text = pdf.pages[idx].extract_text()
            if text:
                lines.extend(text.splitlines())
    return lines


def _find_next_code_line(
    lines: List[str], start_idx: int, current_section: str
) -> Tuple[Optional[int], Optional[re.Match[str]]]:
    for idx in range(start_idx + 1, len(lines)):
        candidate = lines[idx].strip()
        if not candidate:
            continue
        if candidate.upper().startswith("NIVEL"):
            continue
        if SECTION_PATTERN.match(candidate):
            break
        match = CODE_PATTERN.match(candidate)
        if match:
            code = match.group(1)
            if code.startswith(current_section):
                return idx, match
            break
    return None, None


def extract_section_mapping() -> Dict[int, str]:
    """Extrae el mapeo de codigo_iccs -> seccion desde el PDF."""
    lines = _load_pdf_lines()
    mapping: Dict[int, str] = {}
    current_section: Optional[str] = None
    current_section_name: Optional[str] = None
    current_entry: Optional[Dict[str, Any]] = None
    pending_lines: List[str] = []

    def finalize_current() -> None:
        nonlocal current_entry
        if not current_entry:
            return
        code = current_entry["code"]
        codigo_iccs = int(code)
        mapping[codigo_iccs] = current_entry["section_name"]
        current_entry = None

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()
        if not line:
            i += 1
            continue
        if line.isdigit() and len(line) <= 3:
            i += 1
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            finalize_current()
            section_number = section_match.group(1).zfill(2)
            if int(section_number) > MAX_SECTION:
                break
            name_parts = []
            first_part = section_match.group(2).strip()
            if first_part:
                name_parts.append(first_part)
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                if next_line.upper().startswith("NIVEL"):
                    break
                if SECTION_PATTERN.match(next_line):
                    break
                if CODE_PATTERN.match(next_line):
                    break
                name_parts.append(next_line)
                j += 1
            current_section = section_number
            current_section_name = _normalize_text(name_parts) or f"Seccion {int(section_number)}"
            pending_lines = []
            current_entry = None
            i = j
            continue

        if current_section is None:
            i += 1
            continue
        if line.upper().startswith("NIVEL") or line.upper() == "DELITO":
            i += 1
            continue

        code_match = CODE_PATTERN.match(line)
        if code_match and code_match.group(1).startswith(current_section):
            finalize_current()
            code = code_match.group(1)
            current_entry = {
                "code": code,
                "section_name": current_section_name or f"Seccion {int(current_section)}",
            }
        else:
            # No necesitamos procesar las descripciones, solo necesitamos el mapeo
            pass

        i += 1

    finalize_current()
    return mapping


def load_parse_defs_data() -> pd.DataFrame:
    """Loads and concatenates data from parse_defs CSV files."""
    csv_files = list(PARSE_DEFS_DIR.glob("parse_defs_secc_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {PARSE_DEFS_DIR}")

    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    return df


def save_outputs(df: pd.DataFrame, base_name: str) -> None:
    """Saves the DataFrame to CSV, Excel, and JSON formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_path = OUTPUT_DIR / base_name
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_excel(base_path.with_suffix(".xlsx"), index=False, sheet_name="iccs_descripcion")
    df.to_json(base_path.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    print(f"Se generaron {len(df)} registros en CSV, Excel y JSON en {OUTPUT_DIR} con el nombre base '{base_name}'")


def main() -> None:
    # 1. Extract section mapping from PDF
    print("Extrayendo mapeo de secciones desde el PDF...")
    section_mapping = extract_section_mapping()
    print(f"Se encontraron {len(section_mapping)} códigos con su sección correspondiente")

    # 2. Manual adjustments for codes that differ between PDF and CSV
    manual_sections = {
        1042: "Actos que causan la muerte o que tienen la intencion de causar la muerte",
        908: "Actos contra la seguridad publica y la seguridad del Estado"
    }
    section_mapping.update(manual_sections)
    print(f"Se agregaron {len(manual_sections)} ajustes manuales para códigos que difieren entre PDF y CSV")

    # 3. Load data from parse_defs CSVs
    print("Cargando datos desde parse_defs CSVs...")
    df_csv = load_parse_defs_data()
    print(f"Se cargaron {len(df_csv)} registros desde los CSVs")

    # 4. Add section column using the mapping
    df_csv["seccion"] = df_csv["codigo_iccs"].map(section_mapping)

    # 5. Select and reorder columns
    final_columns = [
        "codigo_iccs",
        "glosa_iccs",
        "seccion",
        "descripcion",
        "inclusiones",
        "exclusiones",
        "notas"
    ]
    df_final = df_csv[final_columns]

    # 6. Save outputs
    save_outputs(df_final, OUTPUT_BASENAME)

    # 7. Report missing sections
    missing_sections = df_final["seccion"].isna().sum()
    if missing_sections > 0:
        print(f"\nAdvertencia: {missing_sections} códigos no tienen sección asignada (no se encontraron en el PDF)")


if __name__ == "__main__":
    main()
