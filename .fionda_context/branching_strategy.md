# Fionda Branching Strategy

L'utilizzo di branch separati per ogni "macro-upgrade" ci permette di isolare i problemi, semplificare il testing comparativo e avere punti di ripristino stabili.

## `0cc4m/cpu-arm-nvfp4-fix` (Upstream Reference)
Questo è il branch dal quale siamo partiti per sviluppare il progetto Fionda. Il suo scopo è quello di rimanere come riferimento del suorce code di llama.cpp prima delle nostre modifiche.

## `fionda-scheduler` (Base Implementation)
Implementazione della logica di base dello scheduler di Fionda per migliorare le performances in PP
svuluppi realizzati in questo branch: 
- Allocazione del VRAM double-buffer.
- Setup dei CUDA DMA stream e delle primitive di sincronizzazione.
- Implementazione dell'**Adaptive Placement** (Dynamic Interleaving).
- Meccanismo di sostituzione dinamica dei puntatori dei tensori (`T->data`) a runtime.

## `fionda-speculative` (Advanced Optimizations)
Implementazione dello Speculative Decoding per migliorare le performances in TG a partire da `fionda-scheduler`
svuluppi realizzati in questo branch: 
- Integrazione di un modello draft ($M_q$) per la generazione di token candidati.
- Logica di acceptance/rejection sampling.
- Gestione avanzata del *KV cache rewinding*.
