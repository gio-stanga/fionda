# Benchmarking Fionda + EAGLE1 (Llama 3.1 8B) on the CUDA desktop

Complete procedure to reproduce the NGL-sweep report (same format as
[report_ngl_sweep_llama3.1_fair.txt](report_ngl_sweep_llama3.1_fair.txt)) with the third
`>> FIONDA+EAGLE:` block, on the RTX 3060 Ti desktop (8 GB VRAM, PCIe 3.0 x16, Ubuntu 24.04).

Everything runs from the `fionda-speculative` branch. llama-bench has native EAGLE1 support
(`-md`, `--draft-max`, `--draft-p-min`; the tg tests run real greedy speculative generation and
print a `dm` and an `accept%` column).

---

## 1. Get the code

```bash
git clone <your fionda remote> fionda        # or git fetch on an existing clone
cd fionda
git checkout fionda-speculative
```

## 2. Download the models

Put everything in `models/` (gitignored). Total ≈ 13.5 GB.

### 2a. Target: Meta Llama 3.1 8B Instruct, Q4_K_M + Q8_0

Source: [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

```bash
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models   # 4.6 GB
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    Meta-Llama-3.1-8B-Instruct-Q8_0.gguf   --local-dir models   # 8.0 GB
```

> Use the Meta Instruct model, **not** Hermes (used by the old report): the EAGLE draft was
> trained on Meta Llama 3.1 8B *Instruct* hidden states and embeds Meta's lm_head — on a finetune
> the output stays correct but acceptance (and the speedup) drops.

### 2b. Draft: EAGLE1 GGUF (~560 MB VRAM)

Generated from the original checkpoint
([yuhuili/EAGLE-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B), 1.55 GB):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch gguf numpy sentencepiece

hf download yuhuili/EAGLE-LLaMA3.1-Instruct-8B --local-dir models/EAGLE-LLaMA3.1-Instruct-8B

python scripts/fionda/convert_eagle1_to_gguf.py \
    --eagle-dir models/EAGLE-LLaMA3.1-Instruct-8B \
    --base-gguf models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
    --outfile   models/eagle1-llama3.1-8b-instruct.gguf

./build/bin/llama-quantize --allow-requantize \
    models/eagle1-llama3.1-8b-instruct.gguf \
    models/eagle1-llama3.1-8b-instruct-q4_k_m.gguf Q4_K_M
```

(The conversion needs the base GGUF because the draft reuses the target's lm_head and tokenizer.
`llama-quantize` comes from the build in step 3 — build first, then convert.)

## 3. Build (CUDA, Release)

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

Never benchmark a Debug build.

## 4. Run the sweep

```bash
cd .fionda_context
./run_ngl_sweep_eagle.sh | tee report_ngl_sweep_llama3.1_eagle.txt
```

The script produces, per model (Q4_K_M and Q8_0) and per NGL ∈ {4,8,…,32}, three blocks with the
same VRAM budget ("fair" comparison):

| block | ngl used | flags |
|---|---|---|
| `>> VANILLA:` | ngl | — |
| `>> FIONDA:` | ngl − 2 | `--fionda` (2-slot double-buffer ≈ 2 layers) |
| `>> FIONDA+EAGLE:` | ngl − 2 − K | `--fionda -md <draft> --draft-max 4` |

K is the draft's ~560 MB expressed in target layers: **K = 4 for Q4_K_M, K = 2 for Q8_0**
(encoded in the script's `MODELS=("path:offset" ...)` entries). Steps where the eagle ngl would
drop below 1 are skipped with a message.

Manual single run, if needed:

```bash
./build/bin/llama-bench -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -ngl 14 -p 512 -n 128 --fionda \
    -md models/eagle1-llama3.1-8b-instruct-q4_k_m.gguf --draft-max 4
```

## 5. First-run sanity checks (before trusting any numbers)

1. **Acceptance**: `accept%` on tg128 should be **~45–60 %** (validation reference: ~46 % in
   llama-bench, ~53 % in speculative-simple on a real prompt). Much lower ⇒ something is wrong
   on CUDA. Ignore short runs — tg32 reads artificially low (~22 %) because the cold start on the
   random seed prompt dominates.
2. **Draft placement**: run once with `-v` (or watch `nvidia-smi`) and check the draft's ~560 MB
   sits in VRAM and is NOT streamed — `--fionda` applies only to the target; the bench loads the
   draft with full offload.
3. **Correctness** (once, outside llama-bench): greedy output with EAGLE must match fionda-only:
   ```bash
   ./build/bin/llama-speculative-simple -m <target> -md <draft> --fionda \
       --temp 0 -c 2048 -n 256 -p "<prompt>"
   ```
   Tiny wording differences from float rounding are OK; gibberish or looping is a bug.
   (`-c 2048` is required — without it the 128k default context can OOM.)

## 6. What to record per run

- generation t/s (tg128 row) and prompt t/s (pp512 row)
- `accept%` (eagle rows)
- VRAM used (`nvidia-smi`)

Expected outcome (success criteria from [next_steps.md](next_steps.md)): FIONDA+EAGLE beats
FIONDA-only by **~1.5–2.0×** on tg — and the gain must survive the K fewer static layers the
draft costs. Validation measured ~2.0 tokens per target pass; the win only exists where the
target pass is PCIe-bound, so it should grow as ngl shrinks.

## 7. Follow-up sweeps (best configuration only)

Both flags accept comma-separated lists and only multiply the tg tests (pp runs once):

```bash
./build/bin/llama-bench -m <target> -ngl <best> --fionda -md <draft> \
    -p 0 -n 128 --draft-max 2,4,8 --draft-p-min 0.5,0.75,0.9
```

- On Fionda, longer chains are nearly free (the verify pass cost is moving weights, not math), so
  a lower `--draft-p-min` than the 0.75 default may win.
- Draft F16 vs Q4_K_M: swap in `models/eagle1-llama3.1-8b-instruct.gguf` (the unquantized output
  of the conversion, 2.0 GB) once to confirm the Q4 draft loses no acceptance — remember it eats
  ~1 GB VRAM instead of ~560 MB.

## 8. Caveats / known limits

- `-md` is incompatible with `-d/--n-depth` and `-pg` (llama-bench errors out).
- With `-md`, pp512 is legitimately *lower* than plain FIONDA: EAGLE needs the target's hidden
  states, which forces logits+embeddings for every prompt token. That is the real prefill cost of
  running EAGLE. Keep prompts ≤ ~2k tokens (memory fix is a listed follow-up).
- eagle t/s counts the actual generated tokens (the last round can overshoot n_gen by up to
  draft_max−1) — no bias, just don't be surprised by `n_tokens > 128` in JSON output.

## Links

- Target GGUFs: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- EAGLE1 checkpoint: https://huggingface.co/yuhuili/EAGLE-LLaMA3.1-Instruct-8B
- EAGLE paper: https://arxiv.org/abs/2401.15077 (local copy: [eagle1.pdf](eagle1.pdf))
- Design + validation status: [eagle_implementation_plan.md](eagle_implementation_plan.md)
- Benchmark plan + success criteria: [next_steps.md](next_steps.md)
- Sweep script: [run_ngl_sweep_eagle.sh](run_ngl_sweep_eagle.sh)
