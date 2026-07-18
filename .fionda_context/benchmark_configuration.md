# Benchmark Configuration

## Hardware Configuration
* **OS:** Ubuntu 24.04
* **CPU:** AMD Ryzen 5 5600G
* **Motherboard:** ASUS TUF B450 Gaming Plus V2
* **RAM:** 4x8GB DDR4 2666MHz
* **GPU:** NVIDIA GeForce RTX 3060 Ti (VRAM: 7834 MiB, Compute 8.6)
* **PCIe:** 3.0 x16

---

## OS Configuration
Comandi eseguiti prima di lanciare i test:
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

## Test mmap con Llama 3.1 (ngl 12,5%, no speculative)

1. Vanilla `--mmap 0` vs `--mmap 1` 
    - pp passa da 900 t/s a 800 t/s
    - tg rimane costante a 9,54 t/s

2. Fionda `--mmap 0` vs `--mmap 1` 
    - pp passa da 1130 t/s a 850 t/s
    - tg rimane costante a 9,20 t/s

=> meglio settare `--mmap 0` su entrambi
