# Fionda Benchmarking Guide

## Premessa

Lo scheduler di Fionda ha l'obiettivo di accelerare la fase di prompt processing attraverso la schedulazione dinamica dei layer in VRAM. Inoltre, l'implementazione di Fionda è stata realizzata interamente in CUDA e C++. Assumeremo dunque che l'hardware sul quale viene eseguito è dotato di una scheda video Nvidia con compute capabilities >= 3.7 e CUDA version >= 9.0

---

## 1. Scaricare i modelli

Per scaricare i modelli è necessario visitare https://huggingface.co/ e scaricare in versione `.gguf` i seguenti modelli:
- [Meta Llama 3.1 8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [Gemma 4 12B](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
- [Qwen 3.6 14B](https://huggingface.co/tvall43/Qwen3.6-14B-A3B-VibeForged-v2-GGUF)

Per ciascuno di questi modelli scaricare le quantizzazioni desiderate. Nei nostri test abbiamo utilizzato le quantizzazioni `Q4_K_M` e `Q8_0`.

---

## 2. Compilazione

Compiliamo il progetto in modalità Release
```bash
mkdir -p build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

Per ridurre il rumore dell'OS durante i benchmark, settiamo su performance il profilo energetico di CPU e GPU e disattiviamo la ricerca di aggiornamenti e l'indicizzazione dei files. Su Ubuntu eseguire questi comandi.
```bash
# imposta tutti i core su performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# abilita il persistence mode sulla GPU
sudo nvidia-smi -pm 1

# disattiva ricerca aggiornamenti indicizzazione
sudo systemctl stop unattended-upgrades
systemctl --user stop tracker-extract-3 tracker-miner-fs-3
```

---

## 3. Esecuzione manuale dei benchmark

`llama.cpp` include `llama-bench`, uno strumento ufficiale pensato apposta per realizzare benchmark in maniera precisa ed affidabile. È il modo migliore per testare diverse configurazioni senza l'overhead della UI
```bash
./build/bin/llama-bench -m /percorso/al/modello.gguf -p 512 -n 128 -ngl 12 --mmap 0 --fionda
```

- `-p 512`: prompt di 512 token (misura la fase di Prompt Processing)
- `-n 128`: genera 128 token (misura la fase di Text Generation)
- `-ngl 12`: numero di layer residenti in maniera fissa sulla VRAM della GPU
- `--mmap 0`: serve per pinnare la memoria del sistema (default a 1, pinned memory disattivata)
- `--fionda`: se presente, attiva lo scheduler dinamico di fionda

Come interpretare i risultati
- Prompt Processing: Indica quanti token al secondo vengono processati in fase di pre-fill
- Text Generation): È il throughput effettivo in Tokens/s durante la generazione

---

## 4. Esecuzione automatizzata dei benchmark

Per eseguire automaticamente una batteria di benchmark abbiamo creato i seguenti script bash:
- `run_benchmark.sh`  ->  testa diverse tipologie di prompt con lo stesso numero di layer fissi in VRAM
- `run_ngl_sweep.sh`  ->  testa lo stesso prompt aumentando gradualmente il numero di layer fissi in VRAM

Prima di eseguire questi script è necessario aggiornare i path dei modelli con quelli dei propri modelli precedentemente scaricati. Inoltre, è consigliato testare un modello alla volta in modo da personalizzare i valori di ngl con i quali eseguire i test. Nel nostro caso, i modelli selezionati hanno rispettivamente 32, 48 e 40 layer, dunque abbiamo deciso di creare 3 liste personalizzate di valori con step da 1/8 del numero totale di layer del modello.
