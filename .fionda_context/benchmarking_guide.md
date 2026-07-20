# Fionda Benchmarking Guide

## Premise

The Fionda scheduler aims to speed up the prompt processing phase through dynamic scheduling of layers in VRAM. Furthermore, the Fionda implementation has been built entirely in CUDA and C++. We will therefore assume that the hardware on which it runs is equipped with an Nvidia graphics card with compute capabilities >= 3.7 and CUDA version >= 9.0

Fionda also supports speculative decoding via **EAGLE1** (see [README](../README.md) for how it works). This guide covers benchmarks with and without speculation.

---

## 1. Downloading the models

### 1.1 Target models

To download the models, visit https://huggingface.co/ and download [Meta Llama 3.1 8B](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) in `.gguf` format. In our tests we used the `Q4_K_M` and `Q8_0` quantizations.

### 1.2 Draft model (EAGLE1, only needed for speculation)

Speculation needs a **draft model** matched to a specific target. We provide one for **Meta Llama 3.1 8B Instruct** only; the other two models are benchmarked without speculation. It uses ~160 MB of VRAM (its `lm_head` is shared with the target).

Visit [yuhuili/EAGLE-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B/tree/main) and download all files into `models/EAGLE-LLaMA3.1-Instruct-8B`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch gguf numpy sentencepiece

# needs the base GGUF for tokenizer metadata; build the project first (step 2) so llama-quantize exists
python scripts/fionda/convert_eagle1_to_gguf.py \
    --eagle-dir models/EAGLE-LLaMA3.1-Instruct-8B \
    --base-gguf models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
    --outfile   models/eagle1-llama3.1-8b-instruct-nohead.gguf

./build/bin/llama-quantize --allow-requantize --token-embedding-type q4_k \
    models/eagle1-llama3.1-8b-instruct-nohead.gguf \
    models/eagle1-llama3.1-8b-instruct-nohead-q4_k_m.gguf Q4_K_M
```

(`--token-embedding-type q4_k` is required: without an output tensor the quantize recipe would otherwise bump `token_embd` up to Q6_K.)

---

## 2. Compilation

Compile the project in Release mode
```bash
mkdir -p build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

Never benchmark a Debug build. This build also provides `llama-quantize`, used by the draft conversion in step 1.2, so build before converting the draft.

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

To benchmark **with speculation**, add the draft model and its flags:
```bash
./build/bin/llama-bench -m /path/to/target.gguf -p 512 -n 128 -ngl 12 --mmap 0 --fionda \
    -md /path/to/eagle1-draft.gguf --draft-max 4
```

- `-md`: draft model — its presence enables EAGLE1 speculation on the text generation tests
- `--draft-max N`: maximum number of tokens the draft proposes per target pass
- `--draft-p-min`: confidence threshold below which the draft stops proposing (defaults to 0.75)

The draft stays fully resident in VRAM (`--fionda` schedules only the target), so leave ~160 MB for it when choosing `-ngl`.

How to interpret the results
- Prompt Processing: Indicates how many tokens per second are processed during the pre-fill phase
- Text Generation: Is the effective throughput in Tokens/s during generation
- With `-md`, the text generation rows also report `accept%` (share of drafted tokens the target accepted; healthy is ~45–60%) and `dm` (draft-max). Prompt Processing is expected to be *lower* with `-md`.

---

## 4. Automated benchmark execution

To automatically run a battery of benchmarks we created the following bash scripts:
- `run_benchmark.sh`  ->  tests different prompt sizes with the same number of fixed layers in VRAM
- `run_ngl_sweep_eagle.sh`  ->  tests the same prompt while gradually increasing the number of fixed layers in VRAM, producing three blocks per configuration: `>> VANILLA:` (no fionda), `>> FIONDA:` (`--fionda`), and `>> FIONDA+EAGLE:` (`--fionda -md <draft> --draft-max 4`)

Before running these scripts, the model paths must be updated with those of your previously downloaded models. Furthermore, it is recommended to test one model at a time in order to customize the ngl values used for testing. In our case, the selected models have 32, 48 and 40 layers respectively, so we decided to create 3 custom value lists with a step of 1/8 of the total number of layers in the model.

`run_ngl_sweep_eagle.sh` runs only on the Llama 3.1 8B Instruct target (the only one with a matching draft) and keeps all three blocks at the same VRAM budget. As a sanity check, `accept%` should sit around ~45–60%; much lower means the draft and target are mismatched.
