#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para analizar el problema del CNP 702 y el ICCS 101."""

import pandas as pd
from pathlib import Path
import sys

# Configurar encoding para la salida
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Rutas
REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHES_PATH = Path(__file__).resolve().parent / "artifacts" / "matches_compacto.csv"
ICCS_PATH = REPO_ROOT / "Correspondencia automatica" / "outputs" / "iccs_descripcion.csv"
CNP_PATH = REPO_ROOT / "CNP" / "consolidado_CNP_2025_2021.xlsx"
CNP_PREP_PATH = Path(__file__).resolve().parent / "artifacts" / "cnp_preparado.csv"
ICCS_PREP_PATH = Path(__file__).resolve().parent / "artifacts" / "iccs_preparado.csv"

print("=" * 100)
print("ANALISIS DEL PROBLEMA: CNP 702 (Homicidio) -> ICCS 101 (Homicidio intencional)")
print("=" * 100)

# 1. Cargar datos del CNP 702
print("\n1. INFORMACION DEL CNP 702:")
print("-" * 100)
cnp_df = pd.read_excel(CNP_PATH)
cnp_702 = cnp_df[cnp_df['codigo'] == 702]  # Como int
if len(cnp_702) > 0:
    print(f"Codigo: {cnp_702['codigo'].values[0]}")
    print(f"Glosa: {cnp_702['glosa'].values[0]}")
    print(f"Descripcion: {cnp_702['descripcion'].values[0][:200]}...")
    if 'familia_nombre' in cnp_702.columns:
        print(f"Familia: {cnp_702['familia_nombre'].values[0]}")
else:
    print("CNP 702 no encontrado")

# 1b. Ver texto preparado
cnp_prep_df = pd.read_csv(CNP_PREP_PATH)
cnp_702_prep = cnp_prep_df[cnp_prep_df['codigo'].astype(str) == '702']
if len(cnp_702_prep) > 0:
    print(f"\nTexto embedding CNP 702:")
    print(f"  {cnp_702_prep['texto_embedding'].values[0]}")

# 2. Cargar datos del ICCS 101
print("\n\n2. INFORMACION DEL ICCS 101 (target esperado):")
print("-" * 100)
iccs_df = pd.read_csv(ICCS_PATH)
iccs_101 = iccs_df[iccs_df['codigo_iccs'].astype(str) == '101']
if len(iccs_101) > 0:
    print(f"Codigo: {iccs_101['codigo_iccs'].values[0]}")
    print(f"Glosa: {iccs_101['glosa_iccs'].values[0]}")
    print(f"Descripcion: {iccs_101['descripcion'].values[0][:200]}...")
    print(f"Inclusiones: {iccs_101['inclusiones'].values[0][:300]}...")
    if 'seccion' in iccs_101.columns:
        print(f"Seccion: {iccs_101['seccion'].values[0]}")
else:
    print("ICCS 101 no encontrado")

# 2b. Ver texto preparado
iccs_prep_df = pd.read_csv(ICCS_PREP_PATH)
iccs_101_prep = iccs_prep_df[iccs_prep_df['codigo_iccs'].astype(str) == '101']
if len(iccs_101_prep) > 0:
    print(f"\nTexto embedding ICCS 101:")
    print(f"  {iccs_101_prep['texto_embedding'].values[0][:300]}...")

# 3. Ver los resultados actuales del matching
print("\n\n3. TOP 10 MATCHES ACTUALES PARA CNP 702:")
print("-" * 100)
matches_df = pd.read_csv(MATCHES_PATH)
cnp_702_matches = matches_df[matches_df['cnp_codigo'].astype(str) == '702']

if len(cnp_702_matches) > 0:
    row = cnp_702_matches.iloc[0]
    print(f"CNP 702: {row['cnp_glosa']}")
    print(f"Descripcion: {row['cnp_descripcion'][:150]}...\n")

    iccs_101_rank = None
    for i in range(1, 11):
        codigo_col = f'top{i}_codigo'
        score_col = f'top{i}_score'
        glosa_col = f'top{i}_glosa'

        if codigo_col in row.index:
            codigo = str(row[codigo_col])
            score = row[score_col]
            glosa = row[glosa_col]

            marker = " <- DEBERIA SER #1" if codigo == '101' else ""
            if codigo == '101':
                iccs_101_rank = i
            print(f"  [{i}] Score: {score:.4f} | ICCS {codigo}: {glosa}{marker}")
else:
    print("CNP 702 no encontrado en matches")
    iccs_101_rank = None

# 4. Buscar si el ICCS 101 está en el top 10
print("\n\n4. DIAGNOSTICO:")
print("-" * 100)
if cnp_702_matches is not None and len(cnp_702_matches) > 0:
    if iccs_101_rank:
        print(f"[X] PROBLEMA: ICCS 101 esta en el ranking pero en posicion #{iccs_101_rank}")
        print(f"  El codigo ICCS 101 tiene un score bajo y no es el top-1")
    else:
        print("[X] PROBLEMA GRAVE: ICCS 101 NO esta ni siquiera en el top 10")
        print("  El embedding similarity esta fallando completamente para este caso")

print("\n\n5. ANALISIS DE TEXTOS:")
print("-" * 100)
if len(cnp_702_prep) > 0 and len(iccs_101_prep) > 0:
    cnp_text = cnp_702_prep['texto_embedding'].values[0]
    iccs_text = iccs_101_prep['texto_embedding'].values[0]

    # Remover prefijos
    cnp_text_clean = cnp_text.replace('query: ', '')
    iccs_text_clean = iccs_text.replace('passage: ', '')

    print("Texto CNP (sin prefijo):")
    print(f"  {cnp_text_clean}\n")

    print("Texto ICCS (sin prefijo):")
    print(f"  {iccs_text_clean[:400]}...\n")

    # Analizar palabras clave
    cnp_words = set(cnp_text_clean.lower().split())
    iccs_words = set(iccs_text_clean.lower().split())

    common_words = cnp_words & iccs_words
    print(f"Palabras en comun: {len(common_words)}")
    if len(common_words) > 0:
        print(f"  {', '.join(sorted(common_words))}\n")

    cnp_only = cnp_words - iccs_words
    iccs_only = iccs_words - cnp_words

    print(f"Palabras solo en CNP: {len(cnp_only)}")
    print(f"  {', '.join(sorted(cnp_only))}\n")

    print(f"Palabras solo en ICCS (primeras 50): {len(iccs_only)}")
    print(f"  {', '.join(sorted(list(iccs_only))[:50])}")

print("\n" + "=" * 100)
