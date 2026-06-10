# Etapa 3 — Filtro LLM (CNP → ICCS) con Ollama

Razonamiento legal sobre el **top-10 reordenado** de la etapa 2: el LLM elige el
mejor código ICCS por delito CNP, considerando exclusiones, notas y el **móvil
del delito** (no la consecuencia más grave).

> Actualizado al 10/06/2026: **migración de la API de OpenAI a un modelo local
> con Ollama (`qwen3:8b`)**. Ya no se usa API key. El script nuevo es
> `filtrar_con_llm_ollama.py`. El script anterior (`filtrar_con_llm.py`, OpenAI)
> queda como referencia/legacy.

## Estado

⚠️ **Preparado pero NO ejecutado en esta máquina** (CPU-only). Está pensado para
correr en la máquina con GPU. Aquí solo se dejó listo el pipeline hasta
embeddings + rerank (etapa 2).

## Requisitos

```bash
# Instalar Ollama (https://ollama.com) y descargar el modelo:
ollama pull qwen3:8b
# Dejar el servicio activo (ollama serve o el servicio del SO).

pip install -r requirements.txt   # pandas, tqdm (la API se usa vía urllib)
```

## Insumo

`2_embeddings/outputs/qwen3/matches_rerank_detallado.csv` (top-10 reordenado).
Si no existe el rerank, usa `matches_detallado.csv` (solo embeddings).
Las exclusiones/notas se obtienen de `1_iccs/outputs/iccs_descripcion.csv`.

## Uso

```bash
python filtrar_con_llm_ollama.py --test           # 10 códigos (prueba)
python filtrar_con_llm_ollama.py                  # todos los CNP
python filtrar_con_llm_ollama.py --model qwen3:8b --host http://localhost:11434
```

Detalles de la llamada: API HTTP `POST /api/chat`, `format=json`,
`think=false` (desactiva el razonamiento extendido de Qwen3), `temperature=0.1`.
Reintentos automáticos (3) ante fallos de red o JSON inválido.

## Criterios de decisión (en el prompt)

- Elegir el código **más preciso**, no el más específico.
- **Exclusiones** como filtro crítico; **notas** como contexto.
- **Móvil del delito**: "Robo con homicidio" → ROBO; "Secuestro extorsivo" → SECUESTRO.
- Delitos sin descripción: clasificar con glosa + familia.
- Anti-alucinación: elegir de los candidatos o de un código mencionado en
  exclusiones/inclusiones; si no aplica ninguno, `NINGUNO`.

## Salidas (`outputs/`)

- `clasificacion_final.csv` — compacto (`cnp_codigo`, `cnp_glosa`, `iccs_elegido`, `iccs_glosa_elegida`, `confianza`).
- `clasificacion_con_justificacion.csv` — completo con `justificacion` y `exclusiones_aplicadas`.

## Archivos

- `filtrar_con_llm_ollama.py` — **filtro actual (Ollama)**.
- `filtrar_con_llm.py` — versión OpenAI (legacy, referencia).
- `comparar_clasificaciones.py` — comparación auto vs manual (diagnóstico).
