# Fionda Scheduler - Walkthrough Architetturale (v1.0)

Abbiamo implementato con successo "Fionda", uno scheduler custom per `llama.cpp` progettato per accelerare in modo massivo il *Prompt Processing* su sistemi vincolati (bassa VRAM, bus PCIe lento come il 3.0), tramite il pre-fetching intelligente dei pesi.

## Il Problema Iniziale
Nei sistemi in cui il modello non entra completamente nella VRAM (Partial Offload), `llama.cpp` standard esegue i layer in ordine strettamente sequenziale (es. primi 8 layer su CPU, restanti 20 su GPU). Questo causa un collo di bottiglia durante il **Prompt Processing (pp)**: la GPU rimane completamente inattiva (idle) mentre i layer in eccesso vengono calcolati dalla CPU, sprecando prezioso tempo di elaborazione. Con Fionda, invece, **TUTTI i calcoli avvengono sulla GPU**. I layer del modello si dividono semplicemente in **layer GPU fissi** (residenti in VRAM) e **layer GPU dinamici** (residenti in RAM di sistema e trasferiti al volo via PCIe).

## Architettura e Modifiche (Fasi 1-4)

### Fasi 1-2: Pinned Memory e Stream CUDA Dedicati
Per poter trasferire i dati in parallelo all'esecuzione, la GPU deve poter leggere dalla RAM di sistema senza bloccare la CPU e senza attendere.
- **[NEW] Parametro `--fionda`**: Aggiunto a `llama-bench` e `llama.h` per abilitare la modalità.
- **Pinned Memory (`llama-model.cpp`)**: Abbiamo forzato l'uso di memoria "pagelocked" (tramite `cudaHostAlloc` sotto il cofano) per i layer dinamici, allocando i buffer tramite un nuovo buffer type dedicato (`buft_fionda_cpu`). Questo massimizza la banda PCIe.
- **Gestione Asincrona (`ggml-backend.cpp`, `llama-context.cpp`)**: All'interno del `ggml_backend_sched`, abbiamo allocato due buffer ping-pong in VRAM (`fionda_slots`) grossi quanto un layer, e uno stream CUDA ausiliario (`fionda_stream`) separato dallo stream principale di computazione.

### Fase 3: DMA Double-Buffering (Look-Ahead)
Abbiamo stravolto il modo in cui i grafi (splits) vengono eseguiti dal backend.
- **Look-Ahead (`ggml_backend_sched_compute_splits`)**: Mentre lo stream principale della GPU sta calcolando il layer `L`, Fionda "guarda avanti" nel grafo. Se trova che il layer `L+G` è un layer GPU dinamico (residente in RAM), innesca **immediatamente** un trasferimento DMA asincrono dal buffer Pinned in RAM verso uno dei due slot ping-pong in VRAM usando il `fionda_stream`.
- **Event Synchronization**: Il nodo di computazione che necessiterà di quei pesi si metterà in ascolto (`cudaStreamWaitEvent`) sul completamento del DMA, assicurando correttezza ma azzerando i blocchi sincroni.

### Fase 4: Bresenham Adaptive Interleaving
Questa è stata la genialata che ha permesso al DMA di funzionare! 
Se i layer fissi e quelli dinamici sono raggruppati in due blocchi separati (come avviene con CPU e GPU nel caricamento vanilla), il DMA non ha layer fissi da far calcolare alla GPU "nel frattempo" per nascondere il caricamento di quelli dinamici. 
- **Interlacciamento Matematico (`llama-model.cpp`)**: Abbiamo modificato l'assegnazione iniziale dei layer (tramite `fionda_is_dynamic`) usando un algoritmo in stile Bresenham. Fionda **mescola** l'ordine di esecuzione in modo da spaziare i layer dinamici il più lontano possibile l'uno dall'altro. Questo genera un "Gap" ($G$) di layer GPU fissi tra ogni layer GPU dinamico.
- **L'Overlap Perfetto**: Mentre la GPU passa svariati millisecondi a calcolare il blocco dei layer fissi (es. $G=2$), il PCIe ha tutto il tempo (in background) di trasferire i pesi del layer dinamico successivo verso gli slot VRAM, (idealmente) senza mai bloccare la pipeline di calcolo.

### Fase 6: Hardware Specialization (Micro-Benchmarking Dinamico)
Per rendere Fionda consapevole dell'hardware sottostante (PCIe 3.0 vs 4.0, DDR4 vs DDR5), abbiamo introdotto una routine di **autocalibrazione al lancio**.
- **Test in tempo reale**: Durante il caricamento del modello (`llama_model_load`), Fionda alloca temporaneamente buffer dummy della dimensione esatta di un layer reale (senza alcun limite artificiale di size). Cronometra poi un trasferimento DMA asincrono ($T_{pcie}$) e una finta forward pass sulla GPU ($T_{gpu}$) per ottenere misurazioni di assoluta fedeltà.
- **Hardware Coefficient**: Confrontando questi due tempi, il sistema calcola matematicamente il Gap Ottimale ($G_{opt}$) richiesto dal tuo specifico PC per mascherare totalmente il DMA.
- **Warning Intelligente**: Se la tua macchina non ha abbastanza VRAM per soddisfare il $G_{opt}$, Fionda stampa un *Warning di Bottleneck*, avvertendoti preventivamente che l'overlap non sarà al 100%.

---

## Benchmark Realistici da Eseguire

Per toccare con mano l'impatto reale, puoi usare `llama-bench`. Assicurati che non ci siano altri carichi sulla GPU e lancia prima la versione *Vanilla* e poi la *Fionda* per misurare i t/s.

### Scenario A: Analisi Documenti / RAG
Questo scenario simula il caricamento di un PDF molto lungo (es. 2000 token) a cui viene chiesta una breve sintesi o estratta una specifica informazione.
> **Vanilla**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 1024,2048 -n 1`
> **Fionda**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 1024,2048 -n 1 --fionda`

### Scenario B: Chatbot Bilanciato / Uso Reale Medio
Simula un prompt di sistema + cronologia chat moderata (512 token) con una risposta mediamente lunga (128 token). 
> **Vanilla**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 512 -n 128`
> **Fionda**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 512 -n 128 --fionda`

### Scenario C: Chatbot Rapido / Code Autocomplete
Simula la digitazione rapida o l'autocomplete di codice, dove si fornisce un prompt brevissimo (es. 16 token) e il modello sputa codice/testo lungo (512 token).
> **Vanilla**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 16 -n 512`
> **Fionda**: `./build/bin/llama-bench -m <model.gguf> -ngl 20 -p 16 -n 512 --fionda`
