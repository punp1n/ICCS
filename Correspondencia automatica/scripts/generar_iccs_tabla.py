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
OUTPUT_DIR = Path(
    r"C:\\Users\\Asvaldebenitom\\OneDrive - Instituto Nacional de Estadisticas\\Seguridad y justicia\\ICCS\\Correspondencia automatica\\outputs"
)
START_PAGE_INDEX = 25  # pagina 26 (0-indexado)
END_PAGE_INDEX = 33  # pagina 34 (0-indexado)
MAX_SECTION = 11
OUTPUT_BASENAME = "iccs_tabla"

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


def extract_section_data() -> List[Dict[str, Any]]:
    lines = _load_pdf_lines()
    records: List[Dict[str, Any]] = []
    current_section: Optional[str] = None
    current_section_name: Optional[str] = None
    current_entry: Optional[Dict[str, Any]] = None
    pending_lines: List[str] = []

    def finalize_current() -> None:
        nonlocal current_entry
        if not current_entry:
            return
        description = _normalize_text(current_entry["desc_parts"])
        code = current_entry["code"]
        code_len = len(code)
        entry = {
            "nivel_1": int(code[:2]),
            "nivel_2": int(code[:4]),
            "nivel_3": int(code[:5]) if code_len >= 5 else None,
            "nivel_4": int(code) if code_len >= 6 else None,
            "delito_iccs": description,
            "seccion": current_entry["section_name"],
        }
        records.append(entry)
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
            inline_desc = (code_match.group(2) or "").strip()
            desc_parts: List[str] = []
            if pending_lines:
                desc_parts.extend(pending_lines)
                pending_lines = []
            if inline_desc:
                desc_parts.append(inline_desc)
            current_entry = {
                "code": code,
                "desc_parts": desc_parts,
                "section_name": current_section_name or f"Seccion {int(current_section)}",
            }
        else:
            attach_to_next = False
            if current_section:
                next_idx, next_match = _find_next_code_line(lines, i, current_section)
                if (
                    next_match
                    and not (next_match.group(2) or "").strip()
                    and next_idx is not None
                    and next_idx - i <= 2
                ):
                    attach_to_next = True
            if attach_to_next or current_entry is None:
                pending_lines.append(line)
            else:
                current_entry["desc_parts"].append(line)

        i += 1

    finalize_current()
    return records


def save_outputs(df: pd.DataFrame, base_name: str) -> None:
    """Saves the DataFrame to CSV, Excel, and JSON formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure integer columns with possible NaNs are handled correctly
    for col in ["nivel_1", "nivel_2", "nivel_3", "nivel_4"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    base_path = OUTPUT_DIR / base_name
    df.to_csv(base_path.with_suffix(".csv"), index=False)
    df.to_excel(base_path.with_suffix(".xlsx"), index=False, sheet_name="iccs_tabla")
    df.to_json(base_path.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    print(f"Se generaron {len(df)} registros en CSV, Excel y JSON en {OUTPUT_DIR} con el nombre base '{base_name}'")


def main() -> None:
    # Extract data from PDF
    pdf_data = extract_section_data()
    if not pdf_data:
        raise RuntimeError("No se encontraron registros en el PDF para las secciones solicitadas.")
    df = pd.DataFrame(pdf_data)

    # Save outputs
    save_outputs(df, OUTPUT_BASENAME)


if __name__ == "__main__":
    main()
