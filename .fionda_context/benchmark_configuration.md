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
To reduce OS noise during benchmarks, set the CPU and GPU power profile to performance and disable update search and file indexing. On Ubuntu, run these commands.
```bash
# set all cores to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# enable persistence mode on the GPU
sudo nvidia-smi -pm 1

# disable update search and indexing
sudo systemctl stop unattended-upgrades
systemctl --user stop tracker-extract-3 tracker-miner-fs-3
```

---

## mmap test with Llama 3.1 (ngl 12.5%, no speculative)

1. Vanilla `--mmap 0` vs `--mmap 1` 
    - pp goes from 900 t/s to 800 t/s
    - tg remains constant at 9.54 t/s

2. Fionda `--mmap 0` vs `--mmap 1` 
    - pp goes from 1130 t/s to 850 t/s
    - tg remains constant at 9.20 t/s

=> better to set `--mmap 0` on both
