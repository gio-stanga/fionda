#!/bin/bash

# modelli da testare
MODELS=(
    "/media/andrea/Volume3/ubuntu2404/Downloads/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Hermes-3-Llama-3.1-8B.Q8_0.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/gemma4-v2-Q4_K_M.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/gemma4-v2-Q8_0.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Qwen3.6-14B-A3B-VibeForged-v2-Q8_0.gguf"
)

# numero di layer fissi sulla GPU
NGL=20

echo "Inizio Benchmark Suite Fionda v1.0..."

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "File non trovato: $model"
        continue
    fi
    echo ""
    echo "=========================================================="
    echo "Modello: $(basename "$model")"
    echo "=========================================================="

    # SCENARIO A: PP dominante
    echo "[SCENARIO A] PP dominante (Prompt 2048, Gen 16)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 2048 -n 16
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 2048 -n 16 --fionda
    echo "----------------------------------------------------------"

    # SCENARIO B: PP e TG bilanciati
    echo "[SCENARIO B] PP e TG bilanciati (Prompt 512, Gen 128)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 512 -n 128
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 512 -n 128 --fionda
    echo "----------------------------------------------------------"

    # SCENARIO C: TG dominante
    echo "[SCENARIO C] TG dominante (Prompt 16, Gen 512)"
    echo "----------------------------------------------------------"
    echo ">> VANILLA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 16 -n 512
    echo ">> FIONDA:"
    ../build/bin/llama-bench -m "$model" -ngl $NGL -p 16 -n 512 --fionda
    echo "----------------------------------------------------------"
done

echo "Tutti i test completati!"