#!/bin/bash

# repo root, so the script works from any directory
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# modelli da testare, con offset EAGLE in formato "path:offset"
# L'offset e' il numero di layer equivalenti al VRAM del draft
MODELS=(
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf:1"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf:1"
)

# draft EAGLE1 (fisso in VRAM; senza lm_head - condivisa col target)
DRAFT_MODEL="/media/andrea/Volume3/ubuntu2404/Downloads/eagle1-llama3.1-8b-instruct-nohead-q4_k_m.gguf"
DRAFT_MAX=4

# numero di layer fissi sulla GPU
NGL=12    # llama 37,5%



echo "Inizio Benchmark Suite Fionda v1.0..."

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    eagle_offset="${entry##*:}"
    if [ ! -f "$model" ]; then
        echo "File non trovato: $model"
        continue
    fi
    NGL_Fionda=$((NGL - 2))
    NGL_Eagle=$((NGL - 2 - eagle_offset))
    echo ""
    echo "=========================================================="
    echo "Modello: $(basename "$model")"
    echo "=========================================================="

    # SCENARIO A: TG dominante
    echo "[SCENARIO A] TG dominante (Prompt 16, Gen 512)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL -p 16 -n 512 --mmap 0
    echo ">> FIONDA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Fionda -p 16 -n 512 --mmap 0 --fionda
    echo ">> FIONDA+EAGLE:"
    if [ "$NGL_Eagle" -ge 1 ]; then
        "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Eagle -p 16 -n 512 --mmap 0 --fionda \
            -md "$DRAFT_MODEL" --draft-max $DRAFT_MAX
    else
        echo "(skip: NGL_Eagle=$NGL_Eagle < 1 — budget VRAM insufficiente per il confronto fair)"
    fi
    echo "----------------------------------------------------------"

    # SCENARIO B: PP e TG bilanciati
    echo "[SCENARIO B] PP e TG bilanciati (Prompt 512, Gen 128)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL -p 512 -n 128 --mmap 0
    echo ">> FIONDA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Fionda -p 512 -n 128 --mmap 0 --fionda
    echo ">> FIONDA+EAGLE:"
    if [ "$NGL_Eagle" -ge 1 ]; then
        "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Eagle -p 512 -n 128 --mmap 0 --fionda \
            -md "$DRAFT_MODEL" --draft-max $DRAFT_MAX
    else
        echo "(skip: NGL_Eagle=$NGL_Eagle < 1 — budget VRAM insufficiente per il confronto fair)"
    fi
    echo "----------------------------------------------------------"

    # SCENARIO C: PP dominante
    echo "[SCENARIO C] PP dominante (Prompt 2048, Gen 16)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL -p 2048 -n 16 --mmap 0
    echo ">> FIONDA:"
    "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Fionda -p 2048 -n 16 --mmap 0 --fionda
    echo ">> FIONDA+EAGLE:"
    if [ "$NGL_Eagle" -ge 1 ]; then
        "$ROOT/build/bin/llama-bench" -m "$model" -ngl $NGL_Eagle -p 2048 -n 16 --mmap 0 --fionda \
            -md "$DRAFT_MODEL" --draft-max $DRAFT_MAX
    else
        echo "(skip: NGL_Eagle=$NGL_Eagle < 1 — budget VRAM insufficiente per il confronto fair)"
    fi
    echo "----------------------------------------------------------"
done

echo "Tutti i test completati!"