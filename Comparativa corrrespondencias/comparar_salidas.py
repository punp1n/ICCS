import pandas as pd
import numpy as np

def normalize_code(code):
    """
    Normaliza el código a una cadena de texto.
    Convierte a string, elimina espacios y ".0" al final.
    """
    if pd.isna(code) or code == "":
        return ""
    code_str = str(code).strip()
    if code_str.endswith(".0"):
        return code_str[:-2]
    return code_str

def compare_codes(manual_code, llm_code):
    """
    Compara dos códigos normalizados.
    Retorna True si hay coincidencia jerárquica (uno es prefijo del otro).
    """
    if not manual_code or not llm_code:
        return False
    return manual_code.startswith(llm_code) or llm_code.startswith(manual_code)

def find_and_rename(df, potential_names, new_name):
    """
    Busca una columna en el df a partir de una lista de nombres potenciales y la renombra.
    """
    for name in potential_names:
        if name in df.columns:
            df.rename(columns={name: new_name}, inplace=True)
            return True
    return False

def main():
    """
    Función principal para cargar, comparar y reportar los resultados.
    """
    llm_output_path = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\llm_filter\outputs\clasificacion_con_justificacion.csv"
    manual_output_path = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia manual\2024\28072025_TC_Final_2023-2024_version completa.xlsx"
    manual_sheet_name = "TC_2024"

    print("Cargando archivos...")
    try:
        df_llm = pd.read_csv(llm_output_path)
        df_manual = pd.read_excel(manual_output_path, sheet_name=manual_sheet_name, skiprows=1)
        print("Archivos cargados.")
        print("Columnas originales LLM:", df_llm.columns.tolist())
        print("Columnas originales Manual:", df_manual.columns.tolist())

    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo {e.filename}")
        return
    except Exception as e:
        print(f"Ocurrió un error al cargar los archivos: {e}")
        return

    # --- Estandarización de Nombres de Columnas ---
    print("\nEstandarizando nombres de columnas...")
    
    # Para df_llm
    if not find_and_rename(df_llm, ['cnp', 'CNP', 'CODIGO_CNP', 'cnp_codigo'], 'CNP'):
        print(f"Error: No se encontró columna para 'CNP' en archivo LLM. Columnas: {df_llm.columns.tolist()}")
        return
    if not find_and_rename(df_llm, ['codigo_iccs', 'codigo_llm', 'iccs_elegido'], 'codigo_llm'):
        print(f"Error: No se encontró columna para código ICCS en archivo LLM. Columnas: {df_llm.columns.tolist()}")
        return
    find_and_rename(df_llm, ['justificacion', 'descripcion_iccs', 'descripcion_llm', 'iccs_glosa_elegida'], 'descripcion_llm')

    # Para df_manual: encontrar CNP y luego extraer el código más granular
    if not find_and_rename(df_manual, ['CODIGO CNP', 'cnp', 'CNP', 'CUM'], 'CNP'):
        print(f"Error: No se encontró columna para 'CNP' en archivo manual. Columnas: {df_manual.columns.tolist()}")
        return
    
    print("Extrayendo código manual más granular...")
    level_cols = ['N4-2024 UNODC', 'N3-2024 UNODC', 'N2-2024 UNODC', 'N1-2024 FINAL']
    
    existing_level_cols = [col for col in level_cols if col in df_manual.columns]
    if not existing_level_cols:
        print(f"Error: No se encontró ninguna de las columnas de nivel ({level_cols}) en el archivo manual.")
        return
    
    def get_granular_code(row):
        for col in existing_level_cols:
            if pd.notna(row[col]) and str(row[col]).strip() != '':
                return row[col]
        return np.nan

    df_manual['codigo_manual'] = df_manual.apply(get_granular_code, axis=1)
    find_and_rename(df_manual, ['DESCRIPCIÓN DELITO CNP', 'descripcion_manual'], 'descripcion_manual')

    # --- Preparación para la fusión ---
    llm_cols = ['CNP', 'codigo_llm'] + (['descripcion_llm'] if 'descripcion_llm' in df_llm.columns else [])
    manual_cols = ['CNP', 'codigo_manual'] + (['descripcion_manual'] if 'descripcion_manual' in df_manual.columns else [])
    
    df_llm_subset = df_llm[llm_cols].copy()
    df_manual_subset = df_manual[manual_cols].copy()

    df_llm_subset['CNP'] = df_llm_subset['CNP'].apply(normalize_code)
    df_manual_subset['CNP'] = df_manual_subset['CNP'].apply(normalize_code)

    print("Fusionando datos...")
    df_comparison = pd.merge(df_manual_subset, df_llm_subset, on="CNP", how="outer")

    df_comparison['norm_manual_code'] = df_comparison['codigo_manual'].apply(normalize_code)
    df_comparison['norm_llm_code'] = df_comparison['codigo_llm'].apply(normalize_code)

    df_comparison['coincidencia'] = df_comparison.apply(
        lambda row: compare_codes(row['norm_manual_code'], row['norm_llm_code']),
        axis=1
    )

    # --- Reporte ---
    total_manual = df_comparison['codigo_manual'].notna().sum()
    llm_assigned = df_comparison['codigo_llm'].notna().sum()
    coincidencias = df_comparison['coincidencia'].sum()
    discrepancias = df_comparison[df_comparison['codigo_llm'].notna() & df_comparison['codigo_manual'].notna()]['coincidencia'].eq(False).sum()
    manual_sin_llm = df_comparison[df_comparison['codigo_manual'].notna() & df_comparison['codigo_llm'].isna()].shape[0]

    print("\n============================================================")
    print("EVALUACIÓN VS CORRESPONDENCIA MANUAL (CON LÓGICA JERÁRQUICA):")
    print(f"  Total con etiqueta manual: {total_manual}")
    print(f"  LLM con código asignado: {llm_assigned}")
    print(f"  Coincidencias jerárquicas: {coincidencias}")
    print(f"  Discrepancias: {discrepancias}")
    print(f"  Manual sin respuesta LLM: {manual_sin_llm}")

    df_discrepancies = df_comparison[~df_comparison['coincidencia'] & df_comparison['codigo_llm'].notna() & df_comparison['codigo_manual'].notna()]
    
    if not df_discrepancies.empty:
        print("\n  Muestras de discrepancias (codigo_manual -> codigo_llm):")
        for _, row in df_discrepancies.head(5).iterrows():
            manual_desc = row.get('descripcion_manual', 'N/A') or 'N/A'
            llm_desc = row.get('descripcion_llm', 'N/A') or 'N/A'
            manual_info = f"{row['codigo_manual']} ({manual_desc})"
            llm_info = f"{row['codigo_llm']} ({llm_desc})"
            print(f"    CNP {row['CNP']}: {manual_info} vs {llm_info}")
    
    print("============================================================\n")
    
    output_path = "comparativa_detallada.csv"
    df_comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Análisis detallado guardado en: {output_path}")

if __name__ == "__main__":
    main()
