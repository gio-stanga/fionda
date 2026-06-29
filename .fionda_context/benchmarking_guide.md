# Fionda: Llama.cpp Benchmarking Guide

Prima di modificare la codebase di `llama.cpp` per inserire il double buffering, è fondamentale raccogliere le metriche di *baseline*. Questo ci permetterà di dimostrare l'effettivo miglioramento del *memory bottleneck* e del pipelining PCIe.

Di seguito la procedura per estrarre le metriche richieste sul tuo setup (Ubuntu 24.04, Ryzen 5 5600G, RTX 3060 Ti 8GB).

## 1. Compilazione Ottimizzata

Per avere metriche affidabili, assicurati di compilare in modalità **Release** e con il supporto CUDA abilitato.

```bash
mkdir -p build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

> [!WARNING]
> Mai eseguire benchmark su build di tipo `Debug`. Le prestazioni crollano drasticamente.

---

## 2. Misurare Throughput (Token/s) e Latenza (TTFT)

`llama.cpp` include uno strumento ufficiale molto preciso chiamato `llama-bench`. È il modo migliore per testare diverse configurazioni senza l'overhead dell'interfaccia chat.

**Comando:**
```bash
./build/bin/llama-bench -m /percorso/al/modello.gguf -p 512 -n 128 -ngl 99
```
*Spiegazione dei parametri:*
- `-p 512`: Prompt di 512 token (misura la fase di **Prompt Processing**, da cui deriva il TTFT).
- `-n 128`: Genera 128 token (misura la fase di **Text Generation**, il throughput).
- `-ngl 99`: Tenta di offloadare tutti i layer sulla GPU (fino a esaurimento VRAM).

**Cosa leggere nell'output:**
- **PP (Prompt Processing):** Indica quanti token al secondo vengono processati in fase di pre-fill. Da questo calcoli il **TTFT (Time To First Token)**: $TTFT = \frac{\text{Token del Prompt}}{\text{Throughput PP}}$.
- **TG (Text Generation):** È il throughput effettivo in **Tokens/s** durante la generazione.

---

## 3. Misurare TFLOPS e Memory Bandwidth (GB/s)

Sebbene `llama.cpp` stampi un report alla fine della generazione tramite `llama-cli`, per ottenere i TFLOPS hardware reali e l'effettivo GB/s sul bus di memoria/PCIe, il gold standard è usare **NVIDIA Nsight Compute (`ncu`)** e **Nsight Systems (`nsys`)**.

### Nsight Systems (nsys)
Perfetto per visualizzare se il bus PCIe sta limitando la GPU.
```bash
nsys profile -t cuda,nvtx,osrt --stats=true \
  ./build/bin/llama-cli -m /percorso/al/modello.gguf -p "Scrivi una breve poesia:" -n 10 -ngl 99
```
Questo genererà un file `.nsys-rep`. Aprendolo nella GUI di *Nsight Systems*, vedrai l'esatta timeline dei trasferimenti `cudaMemcpy` (Host-to-Device) contro le esecuzioni dei kernel. Questo sarà il tuo **riferimento visivo principale** per dimostrare che Fionda ha sovrapposto (pipelined) i trasferimenti!

### Report interno di llama.cpp
Alla fine dell'esecuzione di `./build/bin/llama-cli`, noterai delle righe come queste:
```text
llama_print_timings: prompt eval time =  XXX ms /   512 tokens (  YYY ms per token,  ZZZ tokens per second)
llama_print_timings:        eval time =  XXX ms /   128 runs   (  YYY ms per token,  ZZZ tokens per second)
```
Questo è utile per calcoli rapidi.

---

## 4. Rapporto Floating Ops / Memory Ops (Arithmetic Intensity)

L'*Arithmetic Intensity* (AI) si misura in **FLOPs / Byte** (operazioni in virgola mobile per ogni byte trasferito). Nell'inferenza autoregressiva (batch size = 1), l'AI è tipicamente molto basso (memory bound), mentre nel prompt processing (pre-fill) è alto (compute bound).

Per misurarlo con precisione a livello hardware sui tensor core della 3060 Ti:

```bash
sudo ncu --metrics sm__ops_path_tensor.sum,dram__bytes.sum \
  ./build/bin/llama-cli -m /percorso/al/modello.gguf -p "Ciao" -n 5 -ngl 99
```
*Nota: `ncu` richiede privilegi di amministratore per accedere ai contatori hardware (SM counters).*

Il calcolo sarà:
$$ \text{Arithmetic Intensity} = \frac{\text{sm\_\_ops\_path\_tensor.sum (FLOPs)}}{\text{dram\_\_bytes.sum (Bytes)}} $$

Se vuoi l'intensità aritmetica "teorica", per un layer lineare (es. $W \cdot x$):
- **FLOPs:** $2 \times N \times M$
- **Bytes (Modello Q4_K_M):** $\approx \frac{N \times M}{2}$ bytes (perché pesi a 4-bit) + size di $x$.
- Riducendo la precisione dei pesi (da FP16 a Q4), si dimezzano i byte trasferiti, **raddoppiando teoricamente l'Arithmetic Intensity**, mitigando il memory bottleneck.

---

## Attenzione ai Modelli vs. VRAM
Hai 8GB di VRAM.
- I modelli **8B Q4_K_M** occupano circa 4.8 GB. Ci staranno **interamente** in VRAM, e vedrai un throughput altissimo.
- I modelli **8B Q8_0** occupano circa 8.5 GB. Probabilmente sbaveranno sulla RAM e interverrà l'offload parziale (alcuni layer su CPU), facendo crollare le performance.
- I modelli **14B** superano largamente gli 8GB.

Questi modelli "che sbordano" (8B Q8, 14B) saranno i candidati perfetti per dimostrare il vantaggio di **Fionda**! Segnati attentamente il throughput di llama-bench su questi, perché è lì che vedremo se il double-buffering vince sul fallback via CPU.
