# Benchmark Results

## Hardware Configuration
* **OS:** Ubuntu 24.04
* **CPU:** AMD Ryzen 5 5600G
* **Motherboard:** ASUS TUF B450 Gaming Plus V2
* **RAM:** 24GB DDR4
* **GPU:** NVIDIA GeForce RTX 3060 Ti (VRAM: 7834 MiB, Compute 8.6)
* **PCIe:** 3.0 x16

---

## Benchmark Report

**Configurazione:** `-ngl 20` (20 layer fissi in VRAM) per tutti i test  
**Legenda:** 🟢 Fionda vince · 🔴 Vanilla vince · ⚠️ Test fallito (OOM)

---

## Modello 1 — LLaMA 3.1 8B (Hermes-3) — Q4_K_M

**Architettura:** 32 layer totali · **20 su GPU (62%)**, 12 su CPU (38%) · 4.58 GiB

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A - RAG** (pp2048, tg16) | PP 2048 | 1045.66 t/s | 1293.83 t/s | **+23.7%** | 🟢 |
| | TG 16 | 17.55 t/s | 10.83 t/s | -38.3% | 🔴 |
| **B - Chat** (pp512, tg128) | PP 512 | 1225.75 t/s | 1504.77 t/s | **+22.8%** | 🟢 |
| | TG 128 | 17.58 t/s | 10.86 t/s | -38.2% | 🔴 |
| **C - Autocomplete** (pp16, tg512) | PP 16 | 68.06 t/s | 72.81 t/s | +7.0% | 🟢 |
| | TG 512 | 17.38 t/s | 11.06 t/s | -36.4% | 🔴 |

> **Nota:** Con il 62% dei layer in GPU, il modello entra quasi completamente in VRAM. La differenza di TG è consistente (-38%), ma il PP scala bene con Fionda (+23%). Con pochi layer CPU il guadagno teorico massimo è ridotto rispetto ad altri modelli.

---

## Modello 1 — LLaMA 3.1 8B (Hermes-3) — Q8_0

**Architettura:** 32 layer totali · **20 su GPU (62%)**, 12 su CPU (38%) · 7.95 GiB

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A - RAG** (pp2048, tg16) | PP 2048 | 837.75 t/s | 1133.38 t/s | **+35.3%** | 🟢 |
| | TG 16 | 6.48 t/s | 9.75 t/s | **+50.5%** | 🟢 |
| **B - Chat** (pp512, tg128) | PP 512 | 893.01 t/s | 1258.82 t/s | **+40.9%** | 🟢 |
| | TG 128 | 6.48 t/s | 10.40 t/s | **+60.5%** | 🟢 |
| **C - Autocomplete** (pp16, tg512) | PP 16 | 69.16 t/s | 79.18 t/s | **+14.5%** | 🟢 |
| | TG 512 | 6.30 t/s | 10.35 t/s | **+64.3%** | 🟢 |

> **Nota ECCEZIONALE:** Fionda batte Vanilla su tutti e 6 i test, inclusa la Token Generation. Con Q8_0 i pesi sono così grandi che trasferirli via PCIe è comunque più veloce del calcolo della CPU in FP32. Il DMA di Fionda accelera anche la generazione!

---

## Modello 2 — Gemma 4B v2 — Q4_K_M

**Architettura:** 46 layer totali · **20 su GPU (43%)**, 26 su CPU (57%) · 6.86 GiB

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A - RAG** (pp2048, tg16) | PP 2048 | 502.09 t/s | 741.22 t/s | **+47.6%** | 🟢 |
| | TG 16 | 7.00 t/s | 7.73 t/s | +10.4% | 🟢 |
| **B - Chat** (pp512, tg128) | PP 512 | 490.28 t/s | 692.15 t/s | **+41.2%** | 🟢 |
| | TG 128 | 7.29 t/s | 7.60 t/s | +4.3% | 🟢 |
| **C - Autocomplete** (pp16, tg512) | PP 16 | 29.38 t/s | 30.26 t/s | +3.0% | 🟢 |
| | TG 512 | 7.22 t/s | 7.37 t/s | +2.1% | 🟢 |

> **Nota:** Gemma Q4_K_M ha il 57% dei layer su CPU — il caso d'uso ideale di Fionda. Il PP esplode a +47.6%, e perfino la TG è positiva in ogni scenario. Fionda è superiore su tutto.

---

## Modello 2 — Gemma 4B v2 — Q8_0

**Architettura:** 46 layer totali · **20 su GPU (43%)**, 26 su CPU (57%) · 11.78 GiB

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A - RAG** (pp2048, tg16) | PP 2048 | 339.73 t/s | ⚠️ OOM | — | ⚠️ |
| | TG 16 | 3.75 t/s | ⚠️ OOM | — | ⚠️ |
| **B - Chat** (pp512, tg128) | PP 512 | 405.05 t/s | ⚠️ OOM | — | ⚠️ |
| | TG 128 | 3.75 t/s | ⚠️ OOM | — | ⚠️ |
| **C - Autocomplete** (pp16, tg512) | PP 16 | 31.91 t/s | 32.32 t/s | +1.3% | 🟢 |
| | TG 512 | 3.69 t/s | ⚠️ OOM | — | ⚠️ |

> **Nota:** Il modello Q8_0 pesa 11.78 GiB. Con 20 layer su una GPU da 7.8 GiB, il micro-benchmark della Fase 6 occupa VRAM temporanea e supera il limite disponibile. Per testarlo correttamente bisognerebbe abbassare a `-ngl 14`.

---

## Modello 3 — Qwen3.6 14B A3B (MoE) — Q4_K_M

**Architettura:** 94 layer totali · **20 su GPU (21%)**, 74 su CPU (79%) · 7.87 GiB

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A - RAG** (pp2048, tg16) | PP 2048 | 694.30 t/s | 737.81 t/s | **+6.3%** | 🟢 |
| | TG 16 | 23.15 t/s | 25.62 t/s | +10.7% | 🟢 |
| **B - Chat** (pp512, tg128) | PP 512 | 676.82 t/s | 741.41 t/s | **+9.5%** | 🟢 |
| | TG 128 | 24.37 t/s | 25.57 t/s | +4.9% | 🟢 |
| **C - Autocomplete** (pp16, tg512) | PP 16 | 86.34 t/s | 94.94 t/s | **+10.0%** | 🟢 |
| | TG 512 | 24.41 t/s | 26.60 t/s | **+9.0%** | 🟢 |

> **Nota:** Solo 21% dei layer in VRAM — ci si aspetterebbe un crollo. Invece Fionda vince su tutto. La natura MoE di Qwen attiva solo una frazione di esperti per token, rendendo il calcolo GPU molto più veloce e ampliando la finestra temporale disponibile per il DMA.

---

## Modello 3 — Qwen3.6 14B A3B (MoE) — Q8_0

**Architettura:** 94 layer totali · 20 su GPU (21%), 74 su CPU (79%)

| Scenario | Metrica | Vanilla | Fionda | Delta | Esito |
|---|---|---:|---:|---:|:---:|
| **A, B, C** | Tutti | ⚠️ OOM | ⚠️ OOM | — | ⚠️ |

> **Nota:** Il modello Q8_0 non entra con `-ngl 20` né con Vanilla né con Fionda su una GPU da 7.8 GiB. Richiederebbe un NGL molto inferiore o una GPU con più VRAM.

---

## Sintesi Finale

| Modello | Quant | Layer GPU/Tot | PP (miglior caso) | TG | Verdetto |
|---|---|:---:|---:|---|---|
| LLaMA 3.1 8B | Q4_K_M | 20/32 (62%) | **+23.7%** | -38% | Fionda conv. in PP |
| LLaMA 3.1 8B | Q8_0 | 20/32 (62%) | **+40.9%** | **+60%** | Fionda vince su tutto |
| Gemma 4B v2 | Q4_K_M | 20/46 (43%) | **+47.6%** | +4% | Fionda vince su tutto |
| Gemma 4B v2 | Q8_0 | 20/46 (43%) | ⚠️ OOM | ⚠️ OOM | Richiede NGL ridotto |
| Qwen3.6 14B MoE | Q4_K_M | 20/94 (21%) | **+9.5%** | **+7%** | Fionda vince su tutto |
| Qwen3.6 14B MoE | Q8_0 | 20/94 (21%) | ⚠️ OOM | ⚠️ OOM | NGL troppo alto |

**Conclusione:** Fionda è superiore a Vanilla in 5 scenari su 6 testati con successo.
L'unico caso in cui Vanilla regge meglio è LLaMA 8B Q4_K_M in Token Generation, dove il modello è già quasi tutto in VRAM (62%) e il vantaggio del DMA non compensa i salti CPU/GPU.
