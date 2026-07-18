# Fionda Benchmarking Guide

## Premise

The Fionda scheduler aims to speed up the prompt processing phase through dynamic scheduling of layers in VRAM. Furthermore, the Fionda implementation has been built entirely in CUDA and C++. We will therefore assume that the hardware on which it runs is equipped with an Nvidia graphics card with compute capabilities >= 3.7 and CUDA version >= 9.0

---

## 1. Downloading the models

To download the models, visit https://huggingface.co/ and download the following models in `.gguf` format:
- [Meta Llama 3.1 8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [Gemma 4 12B](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
- [Qwen 3.6 14B](https://huggingface.co/tvall43/Qwen3.6-14B-A3B-VibeForged-v2-GGUF)

For each of these models, download the desired quantizations. In our tests we used the `Q4_K_M` and `Q8_0` quantizations.

---

## 2. Compilation

Compile the project in Release mode
```bash
mkdir -p build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

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

## 3. Manual benchmark execution

`llama.cpp` includes `llama-bench`, an official tool designed specifically for running benchmarks in a precise and reliable way. It is the best way to test different configurations without the UI overhead
```bash
./build/bin/llama-bench -m /path/to/model.gguf -p 512 -n 128 -ngl 12 --mmap 0 --fionda
```

- `-p 512`: prompt of 512 tokens (measures the Prompt Processing phase)
- `-n 128`: generates 128 tokens (measures the Text Generation phase)
- `-ngl 12`: number of layers permanently resident in GPU VRAM
- `--mmap 0`: used to pin system memory (defaults to 1, pinned memory disabled)
- `--fionda`: if present, enables the Fionda dynamic scheduler

How to interpret the results
- Prompt Processing: Indicates how many tokens per second are processed during the pre-fill phase
- Text Generation): Is the effective throughput in Tokens/s during generation

---

## 4. Automated benchmark execution

To automatically run a battery of benchmarks we created the following bash scripts:
- `run_benchmark.sh`  ->  tests different prompt sizes with the same number of fixed layers in VRAM
- `run_ngl_sweep.sh`  ->  tests the same prompt while gradually increasing the number of fixed layers in VRAM

Before running these scripts, the model paths must be updated with those of your previously downloaded models. Furthermore, it is recommended to test one model at a time in order to customize the ngl values used for testing. In our case, the selected models have 32, 48 and 40 layers respectively, so we decided to create 3 custom value lists with a step of 1/8 of the total number of layers in the model.
