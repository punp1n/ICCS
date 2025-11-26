import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import sys
import pandas as pd

# Configuración
BASE_DIR = Path(r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\CNP")
OUTPUT_XLSX = BASE_DIR / "consolidado_CNP_2025_2021.xlsx"
OUTPUT_PARQUET = BASE_DIR / "consolidado_CNP_2025_2021.parquet"

# Namespaces XML
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

# Mapeo de meses a números para ranking
MONTH_MAP = {
    "enero": 1,
    "julio": 7,
    "junio": 6, # Por si acaso
    "diciembre": 12
}

def get_text_from_cell(tc):
    """Extrae texto limpio de una celda, eliminando saltos de linea y caracteres de control."""
    texts = []
    for t in tc.findall('.//w:t', NS):
        if t.text:
            texts.append(t.text)
    full_text = "".join(texts)
    # Limpieza agresiva: reemplazar saltos de línea, retornos y tabs por espacio
    full_text = full_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Eliminar espacios múltiples resultantes
    full_text = re.sub(r'\s+', ' ', full_text)
    return full_text.strip()

def is_article_start(text):
    """Detecta si un texto parece el inicio de una mención de artículos."""
    return re.match(r'^(ART|Art)', text, re.IGNORECASE) is not None

def parse_docx(file_path):
    """
    Parsea un archivo .docx y devuelve una lista de diccionarios.
    """
    extracted_rows = []
    
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            tables = tree.findall('.//w:tbl', NS)
            if not tables:
                print(f"Warning: No tables found in {file_path.name}")
                return []
            
            # Asumimos tabla 1
            table = tables[0]
            rows = table.findall('.//w:tr', NS)
            
            current_family = "" # Variable para guardar el Título/Familia actual
            current_code = None
            current_glosa = None
            last_article = None
            
            for row in rows:
                cells = row.findall('.//w:tc', NS)
                if not cells:
                    continue
                
                # Extraer textos con seguridad
                col1 = get_text_from_cell(cells[0]).replace('\u200b', '').strip()
                col2 = ""
                if len(cells) > 1:
                    col2 = get_text_from_cell(cells[1]).replace('\u200b', '').strip()
                
                # Caso 1: La columna 1 tiene contenido (Puede ser Código o Título de Familia)
                if col1:
                    # Validar si es código numérico (ej: "101", "205")
                    if re.match(r'^\d+', col1):
                        current_code = col1
                        current_glosa = col2
                        last_article = None
                    else:
                        # NO es número. Es un Título de Familia (ej: "CRIMENES Y SIMPLES...")
                        
                        # Filtros: Ignorar encabezados de estructura interna
                        text_lower = col1.lower()
                        ignore_patterns = [
                            "códigos que comprende", 
                            "artículos del código", 
                            "infracciones al código penal", # A veces es el título del doc
                            "código procesal penal",
                            "leyes especiales" # A veces es muy genérico, pero lo dejamos si es subtítulo
                        ]
                        
                        if any(pat in text_lower for pat in ignore_patterns):
                            continue
                        
                        # Asumimos que es un nuevo Título/Familia válido
                        current_family = col1
                        
                        # Al cambiar de familia, reseteamos el código actual
                        current_code = None
                        current_glosa = None
                        last_article = None
                
                # Caso 2: Detalle (Sin contenido en col1, es continuación del código actual)
                elif current_code and col2:
                    if is_article_start(col2):
                        last_article = col2
                    elif last_article:
                        # Encontramos descripción para el artículo previo
                        extracted_rows.append({
                            'codigo': current_code,
                            'familia_nombre': current_family, # Guardamos la familia vigente
                            'glosa': current_glosa,
                            'articulado': last_article,
                            'descripcion': col2
                        })
                        last_article = None # Consumimos el artículo
                        
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return []
        
    return extracted_rows

def get_period_score(period_name):
    """
    Convierte '2025_julio' en un entero para comparar (ej: 202507).
    """
    try:
        year, month = period_name.split('_')
        year_val = int(year)
        month_val = MONTH_MAP.get(month.lower(), 0)
        return year_val * 100 + month_val
    except:
        return 0

def main():
    print("Iniciando consolidación con salida XLSX/Parquet...")
    
    all_records = {} # {codigo: { 'score': int, 'period': str, 'rows': [dict] } } 
    
    # 1. Encontrar carpetas de periodo
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
        
        # Parsear
        file_rows = parse_docx(target_file)
        print(f"  -> {len(file_rows)} registros extraídos.")
        
        # 2. Agrupar por código para manejo de versiones
        code_groups = {} 
        for row in file_rows:
            code = row['codigo']
            if code not in code_groups:
                code_groups[code] = []
            code_groups[code].append(row)
            
        # 3. Actualizar registro global "Máximo Vigente"
        for code, rows in code_groups.items():
            if code not in all_records:
                all_records[code] = {
                    'score': period_score,
                    'period': period_name,
                    'rows': rows
                }
            else:
                if period_score > all_records[code]['score']:
                    all_records[code] = {
                        'score': period_score,
                        'period': period_name,
                        'rows': rows
                    }
    
    # 4. Escribir Salidas (XLSX y Parquet)
    print(f"\nGenerando archivos finales...")
    
    try:
        final_rows = []
        sorted_codes = sorted(all_records.keys(), key=lambda x: (len(x), x))
        
        for code in sorted_codes:
            entry = all_records[code]
            period = entry['period']
            
            for row in entry['rows']:
                final_rows.append({
                    'codigo': row['codigo'],
                    'familia_nombre': row.get('familia_nombre', ''),
                    'glosa': row['glosa'],
                    'articulado': row['articulado'],
                    'descripcion': row['descripcion'],
                    'ultimo_vigente': period
                })
        
        # Crear DataFrame
        df = pd.DataFrame(final_rows)
                
        # Exportar a Excel
        print(f"Guardando XLSX: {OUTPUT_XLSX.name}")
        df.to_excel(OUTPUT_XLSX, index=False)
        
        # Exportar a Parquet
        print(f"Guardando Parquet: {OUTPUT_PARQUET.name}")
        # Convertir columnas de texto explícitamente para evitar problemas de tipo mixto si los hubiera
        df = df.astype(str) 
        df.to_parquet(OUTPUT_PARQUET, index=False)
        
        print("Consolidación completada con éxito (XLSX y Parquet).")
        
    except Exception as e:
        print(f"Error escribiendo archivos finales: {e}")

if __name__ == "__main__":
    main()