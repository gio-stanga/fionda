#!/bin/bash

# modelli da testare
MODELS=(
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
#    "/media/andrea/Volume3/ubuntu2404/Downloads/gemma4-v2-Q4_K_M.gguf"
#    "/media/andrea/Volume3/ubuntu2404/Downloads/gemma4-v2-Q8_0.gguf"
#    "/media/andrea/Volume3/ubuntu2404/Downloads/Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf"
#    "/media/andrea/Volume3/ubuntu2404/Downloads/Qwen3.6-14B-A3B-VibeForged-v2-Q8_0.gguf"
)

# numero di layer fissi sulla GPU
NGL_VALUES=(4 8 12 16 20 24 28 32)     # llama
#NGL_VALUES=(6 12 18 24 30 36 42 48)    # gemma
#NGL_VALUES=(5 10 15 20 25 30 35 40)    # qwen

# Parametri Scenario B (PP e TG bilanciati)
PP_TOKENS=512
TG_TOKENS=128



echo "Inizio NGL Sweep — Fionda v1.0 (Scenario B: pp${PP_TOKENS}, tg${TG_TOKENS})..."

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "File non trovato, skip: $model"
        continue
    fi
    echo ""
    echo "=========================================================="
    echo "Modello: $(basename "$model")"
    echo "=========================================================="

    for ngl in "${NGL_VALUES[@]}"; do
        ngl_fionda=$((ngl - 2))
        echo ""
        echo "[NGL = $ngl]"
        echo "----------------------------------------------------------"
        echo ">> VANILLA:"
        ../build/bin/llama-bench -m "$model" -ngl $ngl -p $PP_TOKENS -n $TG_TOKENS --mmap 1
        echo ">> FIONDA:"
        ../build/bin/llama-bench -m "$model" -ngl $ngl_fionda -p $PP_TOKENS -n $TG_TOKENS --mmap 0 --fionda
        echo "----------------------------------------------------------"
    done
done

echo ""
echo "NGL Sweep completato!"
