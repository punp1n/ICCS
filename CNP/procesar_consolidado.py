import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from unicodedata import normalize as uni_normalize, combining

BASE_DIR = Path(
    r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\CNP"
)
OUTPUT_XLSX = BASE_DIR / "consolidado_CNP_2025_2021.xlsx"
OUTPUT_PARQUET = BASE_DIR / "consolidado_CNP_2025_2021.parquet"

MANUAL_XLSX = (
    BASE_DIR.parent
    / "Correspondencia manual"
    / "2024"
    / "28072025_TC_Final_2023-2024_version completa.xlsx"
)
MANUAL_SHEET = "TC_2024"
MANUAL_SKIPROWS = 1
MANUAL_FLAG_VALUE = "agregado_correspondencia_manual"

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

MONTH_MAP = {
    "enero": 1,
    "julio": 7,
    "junio": 6,  # Por si acaso
    "diciembre": 12,
}

ARTICLE_REF = (
    r"(?:ARTS?\.?|ARTICULO|ARTICULOS|Articulo|Art\u00edculo)"
    r"\s*(?:N[\u00b0\u00ba\ufffd]?)?\s*\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|[A-Z]))?"
    r"(?:\s*,?\s*(?:INC\.?|INCISO|NUM\.?|N[\u00b0\u00ba\ufffd])\s*[\w\u00ba\u00b0]+)?"
    r"(?:\s*(?:,|/|y|e|o)\s*\d+(?:\s*(?:BIS|TER|QUATER|QUINQUIES|[A-Z]))?)*"
    r"(?=\s|$|[.;,:-])"
)
LAW_NUMERIC = (
    r"(?:LEY|DL|D\.L\.|D\.F\.L\.|DFL|DECRETO\s+LEY)"
    r"\s*(?:N[\u00b0\u00ba\ufffd]?\s*)?[\d\.]+(?=\s|$|[.;,:-])"
)
LAW_NAMED = (
    r"LEY\s+(?:[A-Z0-9\.\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1]{2,}"
    r"(?:\s+[A-Z0-9\.\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1]{2,})*)(?=\s|$|[.;,:-])"
)
LEGAL_REF_PATTERN = re.compile(rf"(?:{ARTICLE_REF}|{LAW_NUMERIC}|{LAW_NAMED})", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    """Reemplaza saltos de linea/tabs y colapsa espacios multiples."""
    if not text:
        return ""
    text = (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_no_accents_lower(text: str) -> str:
    """Minimiza diferencias de tildes para comparaciones de control."""
    if not text:
        return ""
    decomposed = uni_normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not combining(ch))
    return stripped.lower()


def get_text_from_cell(tc):
    """Extrae texto limpio de una celda, eliminando saltos de linea y caracteres de control."""
    texts = []
    for t in tc.findall(".//w:t", NS):
        if t.text:
            texts.append(t.text)
    full_text = "".join(texts)
    return normalize_whitespace(full_text)


def clean_family_name(text):
    """
    Limpia el nombre de la familia eliminando referencias legales,
    numeros de libros/titulos y rangos de codigos que no aportan valor semantico.
    """
    if not text:
        return ""

    text = re.sub(r"Libro\s+[IVXLCDM]+\s+T[i\u00ed]tulo\s+[IVXLCDM]+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:LEY|D\.F\.L\.?|DFL|DL)\s*(?:N[\u00b0\u00ba\ufffd])?\s*[\d\.]+(?:\s+DE\s+\d{4})?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\(\s*\d+\s*-\s*\d+\s*\)", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip(" ;.,-")


def split_leading_legal_refs(text):
    """
    Extrae referencias legales (articulos/leyes) al inicio de un texto.
    Retorna (refs, remainder) donde remainder ya no incluye esos prefijos.
    """
    if not text:
        return [], ""

    remainder = text.strip()
    refs = []

    while remainder:
        match = LEGAL_REF_PATTERN.match(remainder)
        if not match:
            break
        refs.append(match.group(0).strip(" .;:-"))
        remainder = remainder[match.end():]
        remainder = re.sub(r"^[\s\.;,:-]+", "", remainder)

    return refs, remainder


def split_glosa_content(raw_text):
    """
    Retorna la glosa completa sin procesar referencias legales.
    """
    if not raw_text:
        return "", [], []

    text = normalize_whitespace(raw_text)

    # Devolver la glosa completa sin intentar extraer referencias legales
    return text, [], []


def parse_article_description(text):
    """Retorna la descripción completa sin procesar referencias legales."""
    if not text:
        return [], ""

    cleaned = normalize_whitespace(text)
    cleaned = re.sub(r"^[\s\.;:-]+", "", cleaned)

    # Devolver la descripción completa sin intentar extraer referencias legales
    return [], cleaned.strip()


def safe_cell(value) -> str:
    """Normaliza valores de celdas de Excel a cadena sin 'nan'."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def parse_docx(file_path):
    """
    Parsea un archivo .docx y devuelve una lista de diccionarios.
    Agrupa toda la descripcion de un mismo codigo en una sola entrada.
    """
    extracted_rows = []

    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)

            tables = tree.findall(".//w:tbl", NS)
            if not tables:
                print(f"Warning: No tables found in {file_path.name}")
                return []

            table = tables[0]
            rows = table.findall(".//w:tr", NS)

            current_family = ""
            current_entry = None

            def flush_entry():
                nonlocal current_entry
                if current_entry:
                    extracted_rows.append(
                        {
                            "codigo": current_entry["codigo"],
                            "familia_nombre": current_entry["familia_nombre"],
                            "glosa": current_entry["glosa"],
                            "articles": current_entry["articles"],
                            "descriptions": current_entry["descriptions"],
                        }
                    )
                    current_entry = None

            for row in rows:
                cells = row.findall(".//w:tc", NS)
                if not cells:
                    continue

                col1 = get_text_from_cell(cells[0]).replace("\u200b", "").strip()
                col2 = ""
                if len(cells) > 1:
                    col2 = get_text_from_cell(cells[1]).replace("\u200b", "").strip()

                if col1:
                    if re.match(r"^\d+", col1):
                        flush_entry()
                        glosa_text, header_refs, header_descs = split_glosa_content(col2)
                        current_entry = {
                            "codigo": col1,
                            "familia_nombre": current_family,
                            "glosa": glosa_text,
                            "articles": header_refs,
                            "descriptions": header_descs,
                        }
                    else:
                        text_lower = normalize_no_accents_lower(col1)
                        ignore_patterns = [
                            "codigos que comprende",
                            "articulos del codigo",
                            "articulos del codigo penal",
                            "infracciones al codigo penal",
                            "codigo procesal penal",
                            "leyes especiales",
                        ]
                        if any(pat in text_lower for pat in ignore_patterns):
                            continue

                        current_family = clean_family_name(col1)
                        flush_entry()

                elif current_entry and col2:
                    refs, desc = parse_article_description(col2)
                    current_entry["articles"].extend(refs)
                    if desc:
                        current_entry["descriptions"].append(desc)
                elif (not col1) and col2 and not current_entry:
                    text_lower = normalize_no_accents_lower(col2)
                    ignore_patterns = [
                        "codigos que comprende",
                        "articulos del codigo",
                        "articulos del codigo penal",
                        "infracciones al codigo penal",
                        "codigo procesal penal",
                        "leyes especiales",
                    ]
                    if any(pat in text_lower for pat in ignore_patterns):
                        continue
                    current_family = clean_family_name(col2)

            flush_entry()

    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return []

    return extracted_rows


def get_period_score(period_name):
    try:
        year, month = period_name.split("_")
        year_val = int(year)
        month_val = MONTH_MAP.get(month.lower(), 0)
        return year_val * 100 + month_val
    except Exception:
        return 0


def main():
    print("Iniciando consolidacion (Modo: fila unica por codigo)...")

    all_records = {}

    for folder in BASE_DIR.iterdir():
        if not folder.is_dir():
            continue

        period_name = folder.name
        period_score = get_period_score(period_name)

        if period_score == 0:
            continue

        print(f"Procesando periodo: {period_name}...")
        docx_files = list(folder.glob("*.docx"))
        if not docx_files:
            continue
        target_file = docx_files[0]

        file_rows = parse_docx(target_file)
        print(f"  -> {len(file_rows)} registros extraidos.")

        merged_file_records = {}

        for row in file_rows:
            code = row["codigo"]
            if code not in merged_file_records:
                merged_file_records[code] = {
                    "codigo": code,
                    "familia_nombre": row["familia_nombre"],
                    "glosa": row["glosa"],
                    "articles": row["articles"],
                    "descriptions": row["descriptions"],
                }
            else:
                merged_file_records[code]["articles"].extend(row["articles"])
                merged_file_records[code]["descriptions"].extend(row["descriptions"])

        for code, record in merged_file_records.items():
            if code not in all_records or period_score > all_records[code]["score"]:
                all_records[code] = {
                    "score": period_score,
                    "period": period_name,
                    "data": record,
                }

    print("\nGenerando archivos finales...")

    try:
        final_rows = []
        sorted_codes = sorted(all_records.keys(), key=lambda x: (len(x), x))

        for code in sorted_codes:
            entry = all_records[code]
            data = entry["data"]
            period = entry["period"]

            # Unir todas las descripciones sin procesar referencias legales
            valid_descs = [d for d in data["descriptions"] if d]
            desc_str = " ".join(valid_descs).strip()

            final_rows.append(
                {
                    "codigo": data["codigo"],
                    "familia_nombre": data["familia_nombre"],
                    "glosa": data["glosa"].strip(" .;:-"),
                    "descripcion": desc_str,
                    "ultimo_vigente": period,
                }
            )

        existing_codes = {row["codigo"] for row in final_rows}
        manual_added = []

        if MANUAL_XLSX.exists():
            manual_df = pd.read_excel(
                MANUAL_XLSX, sheet_name=MANUAL_SHEET, skiprows=MANUAL_SKIPROWS, dtype=str
            )
            for _, row in manual_df.iterrows():
                cum_value = safe_cell(row.get("CUM", ""))
                if not cum_value or cum_value in existing_codes:
                    continue

                glosa_manual = safe_cell(row.get("GLOSA 2024", ""))
                desc_manual = safe_cell(row.get("Descripci\u00f3n 2024", ""))

                final_rows.append(
                    {
                        "codigo": cum_value,
                        "familia_nombre": "",
                        "glosa": glosa_manual,
                        "descripcion": desc_manual,
                        "ultimo_vigente": MANUAL_FLAG_VALUE,
                    }
                )
                existing_codes.add(cum_value)
                manual_added.append(cum_value)

            print(f"Correspondencia manual agregada: {len(manual_added)} codigos.")
        else:
            print(f"Warning: Manual correspondence file not found: {MANUAL_XLSX}")

        df = pd.DataFrame(final_rows)

        print(f"Guardando XLSX: {OUTPUT_XLSX.name}")
        df.to_excel(OUTPUT_XLSX, index=False)

        print(f"Guardando Parquet: {OUTPUT_PARQUET.name}")
        df = df.astype(str)
        df.to_parquet(OUTPUT_PARQUET, index=False)

        print("Consolidacion completada.")

    except Exception as e:
        print(f"Error escribiendo archivos finales: {e}")


if __name__ == "__main__":
    main()
