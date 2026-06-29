# Benchmark Results

## Hardware Configuration
* **OS:** Ubuntu 24.04
* **CPU:** AMD Ryzen 5 5600G
* **Motherboard:** ASUS TUF B450 Gaming Plus V2
* **RAM:** 24GB DDR4
* **GPU:** NVIDIA GeForce RTX 3060 Ti (VRAM: 7834 MiB, Compute 8.6)
* **PCIe:** 3.0 x16

---

## Baseline: Partial Offloading (Original llama.cpp)

### 1. gemma4-v2-Q4_K_M.gguf
- **Size:** 6.86 GiB
- **Parameters:** 11.91 B
- **Command:** `-p 512 -n 128 -ngl 30`

| Test | ngl | Throughput (t/s) |
| :--- | :---: | :---: |
| Prompt Processing (pp512) | 30 | 627.00 ± 6.01 |
| Text Generation (tg128) | 30 | 9.99 ± 0.02 |

**Hardware Sensors during test:**
- **GPU Power Draw:** ~65W / 200W
- **GPU Temperature:** ~55°C
- **CPU Thread Utilization:** ~55%

---

### 2. gemma4-v2-Q8_0.gguf
- **Size:** 11.78 GiB
- **Parameters:** 11.91 B
- **Command:** `-p 512 -n 128 -ngl 20`

| Test | ngl | Throughput (t/s) |
| :--- | :---: | :---: |
| Prompt Processing (pp512) | 20 | 433.78 ± 10.99 |
| Text Generation (tg128) | 20 | 3.93 ± 0.01 |

**Hardware Sensors during test:**
- **GPU Utilization** ~53%
- **GPU Temperature:** ~50°C
- **CPU Utilization:** ~56%
