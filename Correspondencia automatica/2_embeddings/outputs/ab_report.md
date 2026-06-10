# Evaluación A/B — acierto de sección ICCS (N1) vs etiqueta manual ICCS_2025

CNP evaluables (etiqueta 1-11): **590**. Métrica: ¿la sección del código ICCS predicho coincide con la manual?

| Configuración | n | top-1 | top-3 | recall@10 |
|---|---:|---:|---:|---:|
| qwen3 (embeddings) | 590 | 0.597 | 0.788 | 0.922 |
| qwen3 + rerank | 590 | 0.578 | 0.814 | 0.903 |
| e5 (embeddings) | 590 | 0.573 | 0.773 | 0.912 |
| e5 + rerank | 590 | 0.571 | 0.783 | 0.883 |
