# Benchmark Results

## Hardware Configuration
* **OS:** Ubuntu 24.04
* **CPU:** AMD Ryzen 5 5600G
* **Motherboard:** ASUS TUF B450 Gaming Plus V2
* **RAM:** 24GB DDR4
* **GPU:** NVIDIA GeForce RTX 3060 Ti (VRAM: 7834 MiB, Compute 8.6)
* **PCIe:** 3.0 x16

---

## Note risultati (ngl 12,5%, no speculative)

1. sia vanilla che fionda `--mmap 1`
    pareggio su pp (vanilla 702 contro fionda 691)
    Vanilla stravince su tg (vanilla 8,56 contro fionda 4,82)

2. sia vanilla che fionda `--mmap 0`
    Fionda vince su tg (fionda 1094 contro vanilla 854)
    pareggio su pp (vanilla 4,96 contro fionda 4,84)

3. vanilla  `--mmap 1` e fionda `--mmap 0`
    Vanilla stravince su tg (vanilla 8,57 contro fionda 4,82)
    Fionda stravince su pp (fionda 1093 contro vanilla 705)