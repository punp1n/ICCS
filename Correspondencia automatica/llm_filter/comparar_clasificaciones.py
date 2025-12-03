"""
Script para comparar clasificación automática vs manual de CNP a ICCS.
Compara desde el nivel más granular (N4) hasta el más general (N1).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Rutas de archivos
AUTO_PATH = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\llm_filter\outputs\clasificacion_con_justificacion.csv"
MANUAL_PATH = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia manual\2024\07102025_TC_Final_2023-2024_v1.2.xlsx"
OUTPUT_PATH = r"C:\Users\Asvaldebenitom\OneDrive - Instituto Nacional de Estadisticas\Seguridad y justicia\ICCS\Correspondencia automatica\outputs\comparacion_auto_vs_manual.xlsx"

def normalizar_codigo_iccs(codigo):
    """Normaliza códigos ICCS removiendo puntos, guiones y convirtiendo a string."""
    if pd.isna(codigo):
        return None
    # Convertir a string y remover caracteres no numéricos
    codigo_str = str(codigo).strip()
    # Remover puntos y espacios
    codigo_str = codigo_str.replace('.', '').replace('-', '').replace(' ', '')
    # Si quedó vacío o es 'nan', retornar None
    if not codigo_str or codigo_str.lower() == 'nan':
        return None
    # Convertir a entero y luego a string para remover ceros a la izquierda innecesarios
    try:
        # Intentar convertir a int para normalizar
        codigo_int = int(float(codigo_str))
        return str(codigo_int)
    except:
        return codigo_str

def obtener_niveles_iccs(codigo):
    """
    Extrae los diferentes niveles jerárquicos de un código ICCS.
    Por ejemplo: 10203 -> {N1: 1, N2: 102, N3: 1020, N4: 10203}
    """
    if not codigo:
        return {'N1': None, 'N2': None, 'N3': None, 'N4': None}

    codigo_str = str(codigo)
    niveles = {}

    # N4: código completo (si tiene 4+ dígitos)
    if len(codigo_str) >= 4:
        niveles['N4'] = codigo_str
    else:
        niveles['N4'] = None

    # N3: primeros 4 dígitos (si existe)
    if len(codigo_str) >= 4:
        niveles['N3'] = codigo_str[:4]
    else:
        niveles['N3'] = None

    # N2: primeros 3 dígitos
    if len(codigo_str) >= 3:
        niveles['N2'] = codigo_str[:3]
    else:
        niveles['N2'] = None

    # N1: primer dígito
    if len(codigo_str) >= 1:
        niveles['N1'] = codigo_str[0]
    else:
        niveles['N1'] = None

    return niveles

def comparar_por_nivel(auto_codigo, manual_n4, manual_n3, manual_n2, manual_n1):
    """
    Compara códigos desde el nivel más granular (N4) al más general (N1).
    Retorna el nivel en que coinciden o None si no coinciden.
    """
    # Normalizar código automático
    auto_codigo_norm = normalizar_codigo_iccs(auto_codigo)
    if not auto_codigo_norm:
        return None, None

    auto_niveles = obtener_niveles_iccs(auto_codigo_norm)

    # Normalizar códigos manuales
    manual_n4_norm = normalizar_codigo_iccs(manual_n4)
    manual_n3_norm = normalizar_codigo_iccs(manual_n3)
    manual_n2_norm = normalizar_codigo_iccs(manual_n2)
    manual_n1_norm = normalizar_codigo_iccs(manual_n1)

    # Comparar desde N4 hasta N1
    if manual_n4_norm and auto_niveles['N4']:
        if auto_niveles['N4'] == manual_n4_norm:
            return 'N4', manual_n4_norm

    if manual_n3_norm and auto_niveles['N3']:
        manual_n3_niveles = obtener_niveles_iccs(manual_n3_norm)
        if auto_niveles['N3'] == manual_n3_niveles['N3']:
            return 'N3', manual_n3_norm

    if manual_n2_norm and auto_niveles['N2']:
        manual_n2_niveles = obtener_niveles_iccs(manual_n2_norm)
        if auto_niveles['N2'] == manual_n2_niveles['N2']:
            return 'N2', manual_n2_norm

    if manual_n1_norm and auto_niveles['N1']:
        manual_n1_niveles = obtener_niveles_iccs(manual_n1_norm)
        if auto_niveles['N1'] == manual_n1_niveles['N1']:
            return 'N1', manual_n1_norm

    return None, None

def main():
    print("=" * 80)
    print("COMPARACIÓN: CLASIFICACIÓN AUTOMÁTICA vs MANUAL")
    print("=" * 80)

    # Cargar datos
    print("\n1. Cargando archivos...")
    df_auto = pd.read_csv(AUTO_PATH)
    df_manual = pd.read_excel(MANUAL_PATH, sheet_name='TC_2024', skiprows=1)

    print(f"   - Clasificación automática: {len(df_auto)} registros")
    print(f"   - Clasificación manual: {len(df_manual)} registros")

    # Renombrar columnas para facilitar merge
    df_auto = df_auto.rename(columns={'cnp_codigo': 'CUM'})

    # Merge de ambos datasets
    print("\n2. Combinando datasets...")
    df_merged = pd.merge(
        df_auto[['CUM', 'cnp_glosa', 'iccs_elegido', 'iccs_glosa_elegida', 'confianza']],
        df_manual[['CUM', 'GLOSA 2024', 'N4-2024 UNODC', 'N3-2024 UNODC',
                   'N2-2024 UNODC', 'N1-2024 FINAL', 'Nombre delito.1']],
        on='CUM',
        how='outer',
        indicator=True
    )

    print(f"   - Total registros combinados: {len(df_merged)}")
    print(f"   - Solo en automático: {len(df_merged[df_merged['_merge'] == 'left_only'])}")
    print(f"   - Solo en manual: {len(df_merged[df_merged['_merge'] == 'right_only'])}")
    print(f"   - En ambos: {len(df_merged[df_merged['_merge'] == 'both'])}")

    # Realizar comparación para registros en ambos
    print("\n3. Comparando clasificaciones...")
    df_ambos = df_merged[df_merged['_merge'] == 'both'].copy()

    # Aplicar comparación por niveles
    resultados = df_ambos.apply(
        lambda row: comparar_por_nivel(
            row['iccs_elegido'],
            row['N4-2024 UNODC'],
            row['N3-2024 UNODC'],
            row['N2-2024 UNODC'],
            row['N1-2024 FINAL']
        ),
        axis=1
    )

    df_ambos['nivel_coincidencia'] = resultados.apply(lambda x: x[0])
    df_ambos['codigo_manual_coincidente'] = resultados.apply(lambda x: x[1])

    # Clasificar resultados
    df_ambos['resultado'] = df_ambos['nivel_coincidencia'].apply(
        lambda x: 'DIVERGENCIA' if x is None else f'CONVERGENCIA_{x}'
    )

    # Calcular métricas
    print("\n" + "=" * 80)
    print("MÉTRICAS DE CONVERGENCIA/DIVERGENCIA")
    print("=" * 80)

    total_comparables = len(df_ambos)

    convergencia_n4 = len(df_ambos[df_ambos['nivel_coincidencia'] == 'N4'])
    convergencia_n3 = len(df_ambos[df_ambos['nivel_coincidencia'] == 'N3'])
    convergencia_n2 = len(df_ambos[df_ambos['nivel_coincidencia'] == 'N2'])
    convergencia_n1 = len(df_ambos[df_ambos['nivel_coincidencia'] == 'N1'])
    divergencia = len(df_ambos[df_ambos['nivel_coincidencia'].isna()])

    print(f"\nTotal de registros comparables: {total_comparables}")
    print(f"\nCONVERGENCIA:")
    print(f"  - Nivel N4 (más específico): {convergencia_n4} ({convergencia_n4/total_comparables*100:.1f}%)")
    print(f"  - Nivel N3: {convergencia_n3} ({convergencia_n3/total_comparables*100:.1f}%)")
    print(f"  - Nivel N2: {convergencia_n2} ({convergencia_n2/total_comparables*100:.1f}%)")
    print(f"  - Nivel N1 (más general): {convergencia_n1} ({convergencia_n1/total_comparables*100:.1f}%)")
    print(f"  - TOTAL CONVERGENCIA: {total_comparables - divergencia} ({(total_comparables - divergencia)/total_comparables*100:.1f}%)")
    print(f"\nDIVERGENCIA TOTAL: {divergencia} ({divergencia/total_comparables*100:.1f}%)")

    # Preparar outputs detallados
    print("\n4. Generando reportes detallados...")

    # Normalizar códigos para visualización
    df_ambos['iccs_auto_norm'] = df_ambos['iccs_elegido'].apply(normalizar_codigo_iccs)
    df_ambos['N4_norm'] = df_ambos['N4-2024 UNODC'].apply(normalizar_codigo_iccs)
    df_ambos['N3_norm'] = df_ambos['N3-2024 UNODC'].apply(normalizar_codigo_iccs)
    df_ambos['N2_norm'] = df_ambos['N2-2024 UNODC'].apply(normalizar_codigo_iccs)
    df_ambos['N1_norm'] = df_ambos['N1-2024 FINAL'].apply(normalizar_codigo_iccs)

    # Sheet 1: Resumen general
    resumen = pd.DataFrame({
        'Métrica': [
            'Total registros automático',
            'Total registros manual',
            'Registros comparables (en ambos)',
            'Solo en automático',
            'Solo en manual',
            '',
            'Convergencia N4',
            'Convergencia N3',
            'Convergencia N2',
            'Convergencia N1',
            'Total Convergencia',
            'Divergencia Total',
            '',
            '% Convergencia N4',
            '% Convergencia N3',
            '% Convergencia N2',
            '% Convergencia N1',
            '% Total Convergencia',
            '% Divergencia'
        ],
        'Valor': [
            len(df_auto),
            len(df_manual),
            total_comparables,
            len(df_merged[df_merged['_merge'] == 'left_only']),
            len(df_merged[df_merged['_merge'] == 'right_only']),
            '',
            convergencia_n4,
            convergencia_n3,
            convergencia_n2,
            convergencia_n1,
            total_comparables - divergencia,
            divergencia,
            '',
            f"{convergencia_n4/total_comparables*100:.2f}%",
            f"{convergencia_n3/total_comparables*100:.2f}%",
            f"{convergencia_n2/total_comparables*100:.2f}%",
            f"{convergencia_n1/total_comparables*100:.2f}%",
            f"{(total_comparables - divergencia)/total_comparables*100:.2f}%",
            f"{divergencia/total_comparables*100:.2f}%"
        ]
    })

    # Sheet 2: Divergencias con glosas
    divergencias = df_ambos[df_ambos['resultado'] == 'DIVERGENCIA'].copy()
    divergencias_detalle = divergencias[[
        'CUM', 'cnp_glosa',
        'iccs_auto_norm', 'iccs_glosa_elegida', 'confianza',
        'N4_norm', 'N3_norm', 'N2_norm', 'N1_norm', 'Nombre delito.1'
    ]].copy()
    divergencias_detalle.columns = [
        'CUM', 'CNP_Glosa',
        'ICCS_Automático', 'ICCS_Glosa_Auto', 'Confianza_Auto',
        'ICCS_Manual_N4', 'ICCS_Manual_N3', 'ICCS_Manual_N2', 'ICCS_Manual_N1',
        'ICCS_Nombre_Manual'
    ]

    # Sheet 3: Convergencias por nivel
    convergencias = df_ambos[df_ambos['resultado'] != 'DIVERGENCIA'].copy()
    convergencias_detalle = convergencias[[
        'CUM', 'cnp_glosa', 'nivel_coincidencia',
        'iccs_auto_norm', 'iccs_glosa_elegida', 'confianza',
        'N4_norm', 'N3_norm', 'N2_norm', 'N1_norm', 'Nombre delito.1'
    ]].copy()
    convergencias_detalle.columns = [
        'CUM', 'CNP_Glosa', 'Nivel_Coincidencia',
        'ICCS_Automático', 'ICCS_Glosa_Auto', 'Confianza_Auto',
        'ICCS_Manual_N4', 'ICCS_Manual_N3', 'ICCS_Manual_N2', 'ICCS_Manual_N1',
        'ICCS_Nombre_Manual'
    ]

    # Sheet 4: Solo en automático
    solo_auto = df_merged[df_merged['_merge'] == 'left_only'][[
        'CUM', 'cnp_glosa', 'iccs_elegido', 'iccs_glosa_elegida', 'confianza'
    ]].copy()
    solo_auto.columns = ['CUM', 'CNP_Glosa', 'ICCS_Automático', 'ICCS_Glosa_Auto', 'Confianza_Auto']

    # Sheet 5: Solo en manual
    solo_manual = df_merged[df_merged['_merge'] == 'right_only'][[
        'CUM', 'GLOSA 2024', 'N4-2024 UNODC', 'N3-2024 UNODC',
        'N2-2024 UNODC', 'N1-2024 FINAL', 'Nombre delito.1'
    ]].copy()
    solo_manual.columns = [
        'CUM', 'CNP_Glosa', 'ICCS_Manual_N4', 'ICCS_Manual_N3',
        'ICCS_Manual_N2', 'ICCS_Manual_N1', 'ICCS_Nombre_Manual'
    ]

    # Guardar en Excel
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        resumen.to_excel(writer, sheet_name='Resumen', index=False)
        divergencias_detalle.to_excel(writer, sheet_name='Divergencias', index=False)
        convergencias_detalle.to_excel(writer, sheet_name='Convergencias', index=False)
        solo_auto.to_excel(writer, sheet_name='Solo_Automático', index=False)
        solo_manual.to_excel(writer, sheet_name='Solo_Manual', index=False)

    print(f"\n[OK] Reporte guardado en: {OUTPUT_PATH}")

    # Mostrar ejemplos de divergencias
    if len(divergencias) > 0:
        print("\n" + "=" * 80)
        print("EJEMPLOS DE DIVERGENCIAS (primeros 5)")
        print("=" * 80)
        for idx, row in divergencias.head(5).iterrows():
            print(f"\nCUM {row['CUM']}: {row['cnp_glosa']}")
            print(f"  Automático: {row['iccs_auto_norm']} - {row['iccs_glosa_elegida']} (confianza: {row['confianza']})")
            print(f"  Manual N4: {row['N4_norm']}, N3: {row['N3_norm']}, N2: {row['N2_norm']}, N1: {row['N1_norm']}")
            print(f"  Manual nombre: {row['Nombre delito.1']}")

    print("\n" + "=" * 80)
    print("COMPARACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()
