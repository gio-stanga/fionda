# FIONDA layer-streaming scheduler

Fionda is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that introduces a dynamic layer scheduler designed to improve prompt processing throughput in partial GPU offloading scenarios.

---

## The problem with the Vanilla scheduler

When a model is too large to fit entirely in VRAM, llama.cpp uses the `-ngl` parameter to decide how many transformer layers to keep on the GPU. The remaining layers stay in system RAM, as one contiguous block. During token generation they are computed by the CPU. During prompt processing (batches of 32 tokens or more), llama.cpp's op-offload instead copies each CPU-resident layer's weights to the GPU over PCIe and computes the layer there.

These copies are serial: the GPU waits for each layer's weights to arrive before computing it, so the GPU and the PCIe bus take turns instead of working at the same time.

---

## Fionda approach

The core idea behind Fionda is to keep the GPU busy for as long as possible by hiding PCIe transfers behind useful computation.

Instead of grouping all CPU layers at the beginning and all GPU layers at the end, Fionda **interleaves** them by spreading CPU-resident (dynamic) layers uniformly among GPU-resident (static) layers using a Bresenham-like algorithm. The result is that after each dynamic layer, if the `-ngl` parameter is high enough, there is always a sequence of static GPU layers to compute while the next dynamic layer is being transferred asynchronously over PCIe.

At model load time, Fionda also runs a small hardware benchmark to estimate the optimal number of GPU layers needed per CPU layer to fully hide the transfer latency. If the current configuration cannot achieve 100% overlap, a warning is printed with a concrete suggestion.

---

## Fionda scheduler implementation

The implementation touches several layers of the stack:

- **Layer interleaving** (`src/llama-model.cpp`): when `--fionda` is active and the model is partially offloaded, layers are assigned to CPU or GPU using a Bresenham distribution instead of the standard contiguous block. The output layer is always kept on the GPU.

- **Async DMA double buffer** (`ggml/src/ggml-backend.cpp`): we extended `ggml_backend_sched` with a VRAM double buffer. Before a GPU split executes, the weights of the next CPU split are already being copied into the alternate VRAM slot over a dedicated `cudaStreamNonBlocking` transfer stream. Synchronization between the transfer stream and the compute stream is handled via CUDA events, so neither stream blocks the other unnecessarily.

- **CUDA resource management** (`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/include/ggml-cuda.h`): a thin API layer exposes VRAM allocation, stream and event creation, async memcpy, and inter-stream synchronization primitives. All resources are created with the correct CUDA device context set explicitly.

- **Profiling** (`include/llama.h`, `src/llama-model.cpp`): Fionda records CUDA timing events around each DMA transfer and each GPU compute block. After inference, `llama_model_get_fionda_profile()` returns average compute time, average DMA time, and the percentage of transfer latency that was successfully hidden. These metrics are also surfaced in `llama-bench` output.

- **Activation gate** (`src/llama-context.cpp`): the async double buffer is only engaged for batches large enough to go through the op-offload path (prompt processing). Token generation, which uses single-token batches, falls back to standard scheduling automatically.

The full implementation diff is at [compare/master...fionda-scheduler](https://github.com/gio-stanga/fionda/compare/master...fionda-scheduler).

---

## Usage

Build with CUDA support:
```bash
mkdir -p build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 12
```

Run inference with Fionda enabled:
```bash
./build/bin/llama-cli -m path/to/model.gguf -ngl 20 --no-mmap --fionda -p "Your prompt here"
```

If you wish to run benchmarks, see [benchmarking.md](benchmarking.md).
