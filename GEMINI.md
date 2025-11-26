# ICCS Data Processing Project

## Overview
This project is dedicated to the processing, standardization, and management of the **International Classification of Crime for Statistical Purposes (ICCS)** data. It involves extracting structured data from official PDF documentation, mapping crime codes, and generating correspondence tables for statistical analysis.

The core automation logic resides in Python scripts that parse PDF documents to extract crime definitions, inclusions, exclusions, and notes, merging them with existing CSV datasets to produce a comprehensive master table.

## Project Structure

### Root Directory

*   **`GEMINI.md`**: This context documentation file.

### `Correspondencia automatica/`
Contains the main automation scripts for ICCS PDF parsing.
*   **`iccs_tabla.py`**: The primary Python script for ICCS PDF extraction.
*   **`parse_defs/`**: Directory containing CSV fragments with parsed definitions.
*   **Output Files**: `iccs_parse_final.csv`, `.json`, `.xlsx`.

### `CNP/`
Contains historical data and the output of the consolidation process.
*   **`2021_enero/`** to **`2025_julio/`**: Directories containing source `.docx` files.
*   **Output Files**: `consolidado_CNP_2025_2021.xlsx`, `consolidado_CNP_2025_2021.parquet`.
*   **`procesar_consolidado.py`**: Script for processing and consolidating historical CNP (Codificación Nacional Penal) data from Word documents (2021-2025).
### `Correspondencia manual/` & `ICCS_UNODC/` & `ICSS_PDF/`
Storage for manual correspondence tables, reference UNODC datasets, and source PDF documentation.

---

## Detailed Script Analysis

### 1. `Correspondencia automatica/iccs_tabla.py`

**Goal:** Extracts hierarchically structured crime data (levels 1-4) directly from the `ICCS_SPANISH_2016_web.pdf` and merges it with detailed definitions stored in auxiliary CSV files.

**Key Functions & Logic:**

*   **`_load_pdf_lines() -> List[str]`**:
    *   Uses `pdfplumber` to open the PDF defined in `PDF_PATH`.
    *   Iterates through a specific page range (`START_PAGE_INDEX` to `END_PAGE_INDEX`) where the summary tables are located.
    *   Extracts raw text line by line.

*   **`extract_section_data() -> List[Dict]`**:
    *   **The Core Engine:** Implements a state-machine approach to parse the unstructured text lines.
    *   **State Tracking:** Keeps track of the `current_section` (e.g., "01"), `current_entry` (incomplete crime record being built), and `pending_lines` (description text spanning multiple lines).
    *   **Regex Matching:** Uses `SECTION_PATTERN` to identify new sections and `CODE_PATTERN` to identify crime codes (e.g., "0101").
    *   **Logic:** When a new code or section is found, the previous entry is "finalized" and added to the list. It handles multi-line descriptions by accumulating text until a new structural element is found.

*   **`load_parse_defs_data() -> pd.DataFrame`**:
    *   Scans the `parse_defs/` directory for CSV files matching `parse_defs_secc_*.csv`.
    *   Concatenates them into a single Pandas DataFrame containing rich metadata (inclusions, exclusions, notes).

*   **`main()`**:
    *   Orchestrates the flow: Extracts PDF data -> Loads CSV definitions -> Merges them on `codigo_iccs`.
    *   **Validation:** Creates a `glosa_match` column to verify if the PDF description matches the CSV description.
    *   **Export:** Calls `save_outputs()` to save the final clean dataset.

### 2. `procesar_consolidado.py`

**Goal:** specific script to process a set of Word documents (`.docx`) located in `CNP/YYYY_mes` folders. It extracts crime codes, glosses, articles, and descriptions, identifies the "family" (legal title) of the crime, and consolidates the data into a single "master" list, keeping the most recent version of each crime.

**Key Functions & Logic:**

*   **`parse_docx(file_path) -> List[Dict]`**:
    *   **Native XML Parsing:** Instead of using heavy libraries like `python-docx`, this function uses `zipfile` to open the `.docx` (which is a zip of XMLs) and `xml.etree.ElementTree` to parse `word/document.xml`.
    *   **Row Iteration:** Iterates through table rows (`<w:tr>`) and cells (`<w:tc>`).
    *   **Family Extraction:** Checks if the first column contains non-numeric text (e.g., "LEY Nº 20.000..."). If so, sets this as the `current_family` for subsequent rows.
    *   **Data Extraction:** If column 1 is a number, it's a **Code**. It captures the Code and Glosa. If column 1 is empty but column 2 has content, it treats it as **Article** (if starting with "ART") or **Description**.
    *   **Safety:** Uses `get_text_from_cell` to handle complex XML text structures.

*   **`get_text_from_cell(tc) -> str`**:
    *   Extracts text from all `<w:t>` tags within a cell.
    *   **Crucial Cleanup:** Aggressively removes newlines (`\n`, `\r`) and tabs, replacing them with spaces to ensure the output is flat and clean for tabular formats.

*   **`get_period_score(period_name) -> int`**:
    *   Converts folder names like "2025_julio" into an integer (e.g., `202507`) to allow mathematical comparison of "freshness".

*   **`main()`**:
    *   **Discovery:** Loops through all folders in `CNP/`.
    *   **Ranking Logic:** Uses a dictionary `all_records` keyed by crime code.
    *   **Update Rule:** If a code exists in `2021_enero` and later in `2025_julio`, the script compares their scores. Since 202507 > 202101, it overwrites the entry with the data from 2025, ensuring only the *latest valid definition* survives.
    *   **Export:** Saves the consolidated data to **Excel (`.xlsx`)** and **Parquet (`.parquet`)** using Pandas.

## Key Technologies

*   **`pandas`**: The backbone for data manipulation, merging, and exporting to Excel/Parquet.
*   **`pdfplumber`**: Used in `iccs_tabla.py` for robust text extraction from PDFs.
*   **`xml.etree.ElementTree`**: Used in `procesar_consolidado.py` for fast, dependency-free parsing of Word document internals.
*   **`unicodedata`**: Used for normalizing text (removing accents, standardizing characters).

## Usage

### Regenerate ICCS Master Table
```bash
python "Correspondencia automatica/iccs_tabla.py"
```

### Regenerate CNP Consolidated History
```bash
python procesar_consolidado.py
```
*Output will be saved to `CNP/consolidado_CNP_2025_2021.xlsx` and `.parquet`.*