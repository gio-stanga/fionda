# Fionda Branching Strategy

Questo documento descrive la strategia di gestione dei branch Git adottata per lo sviluppo del progetto Fionda. L'utilizzo di branch separati per ogni "macro-upgrade" ci permette di isolare i problemi, semplificare il testing comparativo e avere punti di ripristino stabili.

## Struttura dei Branch

### 1. `main` (Upstream Reference)
- **Scopo**: Mantenere il codice originale di `llama.cpp` intatto (nel tuo clone locale equivaleva al branch `0cc4m/cpu-arm-nvfp4-fix`).
- **Utilizzo**: Nessuno sviluppo attivo. Viene utilizzato solo come punto di partenza puro per avere i benchmark della *baseline* (Partial Offloading originale) e per verificare il comportamento del codice upstream originale in caso di dubbi.

### 2. `fionda-scheduler` (Base Implementation)
- **Scopo**: Implementazione della logica di base del Fionda Scheduler (Fasi 1-8).
- **Contenuto**: 
  - Forzatura del caricamento in *pinned memory* per i layer in RAM.
  - Allocazione del VRAM double-buffer.
  - Setup dei CUDA DMA stream e delle primitive di sincronizzazione.
  - Implementazione dell'**Adaptive Placement** (Dynamic Interleaving).
  - Meccanismo di sostituzione dinamica dei puntatori dei tensori (`T->data`) a runtime.
- **Stato**: Questo è il branch principale di lavoro per la prima parte del progetto.

### 3. `fionda-speculative` (Advanced Optimizations)
- **Scopo**: Implementazione dello Speculative Decoding (Fase 9).
- **Creazione**: Verrà staccato (branched off) da `fionda-scheduler` solo *dopo* che lo scheduler dinamico base sarà stato ampiamente testato, profilato e confermato come stabile.
- **Contenuto**:
  - Integrazione di un modello draft ($M_q$) per la generazione di token candidati.
  - Logica di acceptance/rejection sampling.
  - Gestione avanzata del *KV cache rewinding*.
- **Motivazione**: Lo speculative decoding introduce una notevole complessità aggiuntiva. Isolarlo in un branch dedicato garantisce di non "inquinare" o rompere lo scheduler di base.

## Flusso di Lavoro (Workflow)
1. Tutti i commit relativi all'infrastruttura di base e al pipelining PCIe vs Compute avverranno su `fionda-scheduler`.
2. Una volta completati i benchmark positivi su `fionda-scheduler`, la codebase diventerà il nostro nuovo punto di riferimento (un `llama.cpp` ottimizzato per PCIe streaming).
3. Lo sviluppo si sposterà su `fionda-speculative` per la fase finale del progetto universitario. Il test finale misurerà i miglioramenti prestazionali di `fionda-speculative` contro `fionda-scheduler`.
