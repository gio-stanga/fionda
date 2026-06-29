# Fionda project

## Project structure

### Modelli su cui testare (devono essere densi)
- gemma 4 (12B parameters)
- llama 3.1 (8B parameters)
- qwen 3 (14B parameters)


### Tipi di quantizzazione da testare
- INT8
- INT4
- INT3 e INT2 (troppo poco accurato??)


### Definire su che parametri fare i benchmark
- rapporto floating op / mem op
- throughput TOKEN/s del modello
- latenza tra prompt e inizio generazione risposta
- TFLOPS e GB/s di picco e medi registrati


### Definire dove e come eseguire i benchmark
- CUDA events per misurazione dei dati
- esecuzione su desktop con RTX 3060 Ti
- valutiamo se riusciamo ad eseguire da altre parti


### Definire la modalità di sviluppo iterativa
Partendo dal caso studio base (niente codice custom, quantizzazione FP16, ecc..), eseguire delle ottimizzazioni incrementali e documentare accuratamente l'incremento
di performance riscontrato sulle varie architetture. Esempio di roadmap:
- benchmark su performance di partenza
- benchmark con aggiunta di Layer-by-Layer Offloading
- benchmark con aggiunta di speculative decoding
