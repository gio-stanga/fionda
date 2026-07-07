#!/bin/bash

# modelli da testare, con offset EAGLE quant-aware ("path:offset").
# L'offset e' il numero di layer equivalenti al VRAM del draft. Con il draft "nohead"
# (lm_head condivisa col target a runtime) il draft occupa solo ~160 MB in VRAM:
# ~1 layer sia per Q4_K_M che per Q8_0. Verificare con nvidia-smi alla prima run.
MODELS=(
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf:1"
    "/media/andrea/Volume3/ubuntu2404/Downloads/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf:1"
)

# draft EAGLE1 (residente in VRAM, mai streamato; senza lm_head - condivisa col target)
DRAFT_MODEL="/media/andrea/Volume3/ubuntu2404/Downloads/eagle1-llama3.1-8b-instruct-nohead-q4_k_m.gguf"
DRAFT_MAX=4

# numero di layer fissi sulla GPU
NGL_VALUES=(4 8 12 16 20 24 28 32 36)     # llama

# Parametri Scenario B (PP e TG bilanciati)
PP_TOKENS=512
TG_TOKENS=128



echo "Inizio NGL Sweep — Fionda v1.0 + EAGLE1 (Scenario B: pp${PP_TOKENS}, tg${TG_TOKENS})..."

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    eagle_offset="${entry##*:}"
    if [ ! -f "$model" ]; then
        echo "File non trovato, skip: $model"
        continue
    fi
    echo ""
    echo "=========================================================="
    echo "Modello: $(basename "$model") (EAGLE_OFFSET=$eagle_offset)"
    echo "=========================================================="

    for ngl in "${NGL_VALUES[@]}"; do
        # offset "fair": stesso budget VRAM totale in tutti i blocchi.
        # Fionda paga i 2 slot del double-buffer; Eagle paga anche il draft.
        ngl_fionda=$((ngl - 2))
        ngl_eagle=$((ngl - 2 - eagle_offset))
        echo ""
        echo "[NGL = $ngl]"
        echo "----------------------------------------------------------"
        echo ">> VANILLA:"
        ../build/bin/llama-bench -m "$model" -ngl $ngl -p $PP_TOKENS -n $TG_TOKENS --mmap 1
        echo ">> FIONDA:"
        ../build/bin/llama-bench -m "$model" -ngl $ngl_fionda -p $PP_TOKENS -n $TG_TOKENS --mmap 0 --fionda
        echo ">> FIONDA+EAGLE:"
        if [ "$ngl_eagle" -ge 1 ]; then
            ../build/bin/llama-bench -m "$model" -ngl $ngl_eagle -p $PP_TOKENS -n $TG_TOKENS --mmap 0 --fionda \
                -md "$DRAFT_MODEL" --draft-max $DRAFT_MAX
        else
            echo "(skip: ngl_eagle=$ngl_eagle < 1 — budget VRAM insufficiente per il confronto fair)"
        fi
        echo "----------------------------------------------------------"
    done
done

echo ""
echo "NGL Sweep completato!"
