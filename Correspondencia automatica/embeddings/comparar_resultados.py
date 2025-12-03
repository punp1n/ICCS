#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para comparar resultados antes y después de multi-field embeddings."""

import pandas as pd
from pathlib import Path
import sys

# Configurar encoding para la salida
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Rutas
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OLD_MATCHES = ARTIFACTS_DIR / "matches_compacto_original.csv"  # Backup del original
NEW_MATCHES = ARTIFACTS_DIR / "matches_compacto.csv"  # Nuevos resultados

def compare_cnp_match(codigo_cnp: str, old_df: pd.DataFrame, new_df: pd.DataFrame):
    """Compara los matches de un código CNP específico."""
    old_row = old_df[old_df['cnp_codigo'].astype(str) == str(codigo_cnp)]
    new_row = new_df[new_df['cnp_codigo'].astype(str) == str(codigo_cnp)]

    if len(old_row) == 0:
        print(f"CNP {codigo_cnp} no encontrado en resultados antiguos")
        return
    if len(new_row) == 0:
        print(f"CNP {codigo_cnp} no encontrado en resultados nuevos")
        return

    old_row = old_row.iloc[0]
    new_row = new_row.iloc[0]

    print("=" * 100)
    print(f"COMPARACION PARA CNP {codigo_cnp}: {old_row['cnp_glosa']}")
    print("=" * 100)
    print(f"Descripcion: {old_row['cnp_descripcion'][:150]}...\n")

    print("RESULTADOS ANTERIORES (concatenacion simple):")
    print("-" * 100)
    for i in range(1, 11):
        codigo_col = f'top{i}_codigo'
        score_col = f'top{i}_score'
        glosa_col = f'top{i}_glosa'

        if codigo_col in old_row.index:
            codigo = str(old_row[codigo_col])
            score = old_row[score_col]
            glosa = old_row[glosa_col]
            print(f"  [{i}] Score: {score:.4f} | ICCS {codigo}: {glosa}")

    print("\n\nRESULTADOS NUEVOS (multi-campo con pesos):")
    print("-" * 100)
    improvements = []
    for i in range(1, 11):
        codigo_col = f'top{i}_codigo'
        score_col = f'top{i}_score'
        glosa_col = f'top{i}_glosa'

        if codigo_col in new_row.index:
            codigo = str(new_row[codigo_col])
            score = new_row[score_col]
            glosa = new_row[glosa_col]

            # Buscar este código en resultados anteriores
            old_rank = None
            for j in range(1, 11):
                if str(old_row[f'top{j}_codigo']) == codigo:
                    old_rank = j
                    break

            marker = ""
            if old_rank is None:
                marker = " [NUEVO en top-10]"
                improvements.append(f"ICCS {codigo} ahora en posicion {i} (antes fuera del top-10)")
            elif old_rank > i:
                marker = f" [SUBIO desde #{old_rank}]"
                improvements.append(f"ICCS {codigo} subio de #{old_rank} a #{i}")
            elif old_rank < i:
                marker = f" [BAJO desde #{old_rank}]"

            print(f"  [{i}] Score: {score:.4f} | ICCS {codigo}: {glosa}{marker}")

    print("\n\nRESUMEN DE CAMBIOS:")
    print("-" * 100)
    if improvements:
        print("MEJORAS detectadas:")
        for imp in improvements:
            print(f"  + {imp}")
    else:
        print("  Sin mejoras significativas en ranking")

    # Ver si hubo cambio en el top-1
    old_top1 = str(old_row['top1_codigo'])
    new_top1 = str(new_row['top1_codigo'])
    old_top1_score = old_row['top1_score']
    new_top1_score = new_row['top1_score']

    print(f"\nTOP-1:")
    if old_top1 != new_top1:
        print(f"  CAMBIO: {old_top1} (score {old_top1_score:.4f}) -> {new_top1} (score {new_top1_score:.4f})")
    else:
        print(f"  SIN CAMBIO: {new_top1} (score: {old_top1_score:.4f} -> {new_top1_score:.4f})")

    print("\n" + "=" * 100 + "\n")


def main():
    if not OLD_MATCHES.exists():
        print(f"ERROR: No se encuentra el archivo de backup: {OLD_MATCHES}")
        print("Primero haz backup de los resultados actuales:")
        print(f"  cp {ARTIFACTS_DIR / 'matches_compacto.csv'} {OLD_MATCHES}")
        return

    if not NEW_MATCHES.exists():
        print(f"ERROR: No se encuentra el archivo de nuevos resultados: {NEW_MATCHES}")
        print("Ejecuta primero el script con --multi-field")
        return

    print("\nCargando resultados...")
    old_df = pd.read_csv(OLD_MATCHES)
    new_df = pd.read_csv(NEW_MATCHES)

    print(f"  - Resultados antiguos: {len(old_df)} codigos CNP")
    print(f"  - Resultados nuevos: {len(new_df)} codigos CNP\n")

    # Comparar CNP 702 (Homicidio)
    compare_cnp_match("702", old_df, new_df)

    # Estadísticas generales
    print("\n" + "=" * 100)
    print("ESTADISTICAS GENERALES")
    print("=" * 100)

    old_top1_avg = old_df['top1_score'].mean()
    new_top1_avg = new_df['top1_score'].mean()

    print(f"Score promedio top-1:")
    print(f"  Antiguo: {old_top1_avg:.4f}")
    print(f"  Nuevo:   {new_top1_avg:.4f}")
    print(f"  Cambio:  {(new_top1_avg - old_top1_avg):+.4f} ({((new_top1_avg - old_top1_avg) / old_top1_avg * 100):+.2f}%)")

    # Contar cuántos códigos mejoraron su top-1
    mejoras = 0
    empeoramientos = 0
    sin_cambio = 0

    for idx in range(len(old_df)):
        old_score = old_df.iloc[idx]['top1_score']
        new_score = new_df.iloc[idx]['top1_score']

        if new_score > old_score + 0.001:  # Threshold para considerar mejora
            mejoras += 1
        elif new_score < old_score - 0.001:
            empeoramientos += 1
        else:
            sin_cambio += 1

    print(f"\nDistribucion de cambios en score top-1:")
    print(f"  Mejoras:        {mejoras} ({mejoras / len(old_df) * 100:.1f}%)")
    print(f"  Empeoramientos: {empeoramientos} ({empeoramientos / len(old_df) * 100:.1f}%)")
    print(f"  Sin cambio:     {sin_cambio} ({sin_cambio / len(old_df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
