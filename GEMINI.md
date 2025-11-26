# ICCS Data Processing Project

## Overview
This project is dedicated to the processing, standardization, and management of the **International Classification of Crime for Statistical Purposes (ICCS)** data. It involves extracting structured data from official PDF documentation, mapping crime codes, and generating correspondence tables for statistical analysis.

The core automation logic resides in Python scripts that parse PDF documents to extract crime definitions, inclusions, exclusions, and notes, merging them with existing CSV datasets to produce a comprehensive master table.

## Project Structure

### `Correspondencia automatica/`
Contains the main automation scripts and their outputs.
*   **`iccs_tabla.py`**: The primary Python script. It performs the following operations:
    1.  **PDF Extraction:** Reads `ICCS_SPANISH_2016_web.pdf` to extract crime codes (Levels 1-4) and descriptions using `pdfplumber`.
    2.  **Data Merging:** Loads CSV definitions from `parse_defs/` and merges them with the PDF data.
    3.  **Validation:** Compares descriptions (`glosa_iccs` vs `delito_iccs`) to ensure accuracy.
    4.  **Output:** Generates `iccs_parse_final` in CSV, JSON, and Excel formats.
*   **`parse_defs/`**: Directory containing CSV fragments with parsed definitions (Sections 01-03, 04-08, 09-11) used as inputs for the merge process.
*   **Output Files**: `iccs_parse_final.csv`, `.json`, `.xlsx`.

### `Correspondencia manual/`
Stores manually curated correspondence tables, organized by year.
*   **`2023/`**: Correspondence files for the 2022-2023 period.
*   **`2024/`**: Final correspondence versions for 2023-2024.

### `ICCS_UNODC/`
Contains reference datasets provided by UNODC.
*   `Delitos_Espanol_170221.csv`: Base list of crimes in Spanish.
*   `Delitos_Ingles_170221.csv`: Base list of crimes in English.

### `ICSS_PDF/`
Source documentation.
*   `ICCS_SPANISH_2016_web.pdf`: The primary source document parsed by the script.
*   Other versions (English, rotated) are also stored here.

## Key Technologies & Dependencies

The project relies on Python for data processing. Key libraries inferred from imports:
*   **`pandas`**: For data manipulation and CSV/Excel export.
*   **`pdfplumber`**: For extracting text and structure from PDF files.
*   **`unicodedata` & `re`**: For text normalization and pattern matching.

## Usage & Configuration

### Running the Script
To regenerate the master tables, execute the main script from the `Correspondencia automatica` directory:

```bash
python iccs_tabla.py
```

### Configuration Note
The script `iccs_tabla.py` currently uses **absolute paths** specific to the local environment:
*   `PDF_PATH`: Points to the PDF in `ICSS_PDF`.
*   `OUTPUT_DIR`: Points to `Correspondencia automatica`.

**Caution:** When moving this project to a new machine or environment, these paths in `iccs_tabla.py` must be updated to reflect the new directory structure or converted to relative paths.

## Data Schema (`iccs_parse_final`)

The generated master table includes the following key fields:
*   **`codigo_iccs`**: The unique identifier for the crime.
*   **`nivel_1` - `nivel_4`**: Hierarchical breakdown of the crime code.
*   **`delito_iccs`**: Description extracted from the PDF.
*   **`glosa_iccs`**: Description from the definition CSVs.
*   **`glosa_match`**: Boolean indicating if the descriptions match.
*   **`descripcion`**: Detailed description of the crime.
*   **`inclusiones`**: Specific acts included in this category.
*   **`exclusiones`**: Acts excluded from this category.
*   **`notas`**: Additional context or footnotes.
