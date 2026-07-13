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
NGL=12    # llama
#NGL=18    # gemma
#NGL=15    # qwen



echo "Inizio Benchmark Suite Fionda v1.0..."

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "File non trovato: $model"
        continue
    fi
    NGL_Fionda=$((NGL - 2))
    echo ""
    echo "=========================================================="
    echo "Modello: $(basename "$model")"
    echo "=========================================================="

    # SCENARIO A: TG dominante
    echo "[SCENARIO A] TG dominante (Prompt 16, Gen 512)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 16 -n 512 --mmap 0
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL_Fionda -p 16 -n 512 --mmap 0 --fionda
    echo "----------------------------------------------------------"

    # SCENARIO B: PP e TG bilanciati
    echo "[SCENARIO B] PP e TG bilanciati (Prompt 512, Gen 128)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 512 -n 128 --mmap 0
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL_Fionda -p 512 -n 128 --mmap 0 --fionda
    echo "----------------------------------------------------------"

    # SCENARIO C: PP dominante
    echo "[SCENARIO C] PP dominante (Prompt 2048, Gen 16)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 2048 -n 16 --mmap 0
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL_Fionda -p 2048 -n 16 --mmap 0 --fionda
    echo "----------------------------------------------------------"
done

echo "Tutti i test completati!"