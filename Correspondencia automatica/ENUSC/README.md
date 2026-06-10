# ENUSC -> CNP/CUM por embeddings

Este flujo calcula los top 10 códigos CUM/CNP para 15 glosas ENUSC sin usar LLM.

## Ejecución

Desde la raíz del repositorio:

```bash
PYTHONPATH="Correspondencia automatica/2_embeddings/.venv/lib/python3.12/site-packages" \
python3 -S "Correspondencia automatica/ENUSC/generar_top10_enusc_cnp.py" --batch-size 64
```

Si el entorno local tiene las dependencias instaladas directamente:

```bash
python3 "Correspondencia automatica/ENUSC/generar_top10_enusc_cnp.py" --batch-size 64
```

## Metodología

- Modelo: `intfloat/multilingual-e5-large`.
- Backend: `transformers` con mean pooling y normalización L2.
- Query ENUSC: `query: glosa_enusc`.
- Passage CNP/CUM: `passage: glosa_cum | descripcion_cum`.
- Ranking: similitud coseno entre embeddings normalizados.
- No incluye etapa LLM.

## Salidas

La carpeta `outputs/` contiene:

- `top10_enusc_cnp.xlsx`: resultado principal.
- `top10_enusc_cnp_detallado.csv`: una fila por glosa ENUSC y candidato CUM.
- `top10_enusc_cnp_compacto.csv`: una fila por glosa ENUSC con columnas `top1...top10`.
- `metadata_top10_enusc_cnp.json`: trazabilidad de modelo, hashes e insumos.
