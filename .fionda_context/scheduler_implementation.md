# Scheduler Implementation

## How llama.cpp currently handles scheduling

Understanding llama.cpp's existing scheduling is the prerequisite to knowing what Fionda needs to build. There are two distinct systems: static weight placement at model load time, and a graph dispatcher that runs at inference time.

---

### 1. Static weight placement at load time

When a model is loaded, `load_tensors()` in `src/llama-model.cpp:2612` immediately and permanently decides where each layer's weight tensors will live — CPU RAM or GPU VRAM. The decision is based on the `n_gpu_layers` parameter:

```cpp
// src/llama-model.cpp:2672-2684
const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);
auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        return {cpu_dev, &pimpl->cpu_buft_list};   // → CPU RAM
    }
    const int layer_gpu = ...;
    return {dev, &pimpl->gpu_buft_list.at(dev)};   // → GPU VRAM
};
```

With `n_gpu_layers=20` on a 32-layer model: layers 0–11 are placed in CPU RAM, layers 12–31 in VRAM. This assignment is fixed for the entire session. The weights never move after this point.

---

### 2. The computation graph and what a "split" is

For every forward pass (every token generated), llama.cpp builds a computation graph — a DAG of all tensor operations needed to produce the output. This is done by `model.build_graph()` called from `src/llama-context.cpp:1200`.

Once the graph is built, `ggml_backend_sched_split_graph()` in `ggml/src/ggml-backend.cpp:924` walks the graph and partitions it into **splits**. A split is a contiguous segment of operations that all run on the same backend (CPU or CUDA). The partition is determined by where each operation's input weight tensors live:

```cpp
// ggml/src/ggml-backend.cpp:827-839
if (src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
    int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
    // ...
    return src_backend_id;  // op is assigned to the backend that owns the weight
}
```

**A split is the unit of dispatch** — it is handed to one backend and computed there entirely. When consecutive ops belong to different backends, a split boundary is created and the scheduler inserts an automatic tensor copy to transfer activation data (the intermediate results) across the boundary.

In practice, for a model with all layers in VRAM the graph becomes roughly:
- Split 0 (CPU): input embedding lookup
- Split 1 (CUDA): all transformer layers
- Split 2 (CPU, optional): output normalization if kept on CPU

For a partial offload (e.g. 12 layers on CPU, 20 on GPU), there would be more boundaries and more copies of the activations flowing between backends.

---

### 3. llama.cpp does not move weight data at inference time

This is the most important thing to understand: **llama.cpp never moves layer weight data at inference time**. The weights sit in their buffer (RAM or VRAM) from load time and stay there. The scheduler only copies **activations** (the flowing intermediate data between layers) when they cross a split boundary, never the weights themselves.

The rule is straightforward: if a weight is in VRAM, that layer's computation runs on CUDA. If a weight is in CPU RAM, that layer's computation runs on CPU.

#### The exception: `op_offload`

There is one special case. If weights are placed in **pinned host memory** (a special CPU-side buffer accessible by the GPU via zero-copy DMA, allocated with `ggml_backend_dev_host_buffer_type`) and the `op_offload` flag is enabled, the CUDA backend can claim the operation and run it on the GPU while reading the weights directly from host RAM over PCIe:

```cpp
// ggml/src/ggml-backend.cpp:830-836
if (sched->op_offload && src_backend_id == sched->n_backends - 1
        && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor)
                && ggml_backend_offload_op(sched->backends[b], tensor)) {
            return b;  // assign to GPU even though weight is in host RAM
        }
    }
}
```

The CUDA backend only accepts the offload when the batch size is large enough to justify it (`ggml/src/ggml-cuda/ggml-cuda.cu:5086-5090`).

**This is not what Fionda does, and it is fundamentally worse.** The zero-copy path reads weights from RAM over PCIe inside each matmul kernel call — so for every token the full layer weight data traverses PCIe during the GPU compute kernel itself. It is sequential: PCIe read and GPU compute happen at the same time *within a single op*, but **there is no pipelining between layers**. Layer N+1's weight data does not start moving over PCIe until layer N finishes completely.

---

### 4. The compute loop and where Fionda intercepts

After the graph is split, `ggml_backend_sched_compute_splits()` in `ggml/src/ggml-backend.cpp:1447` runs the splits one by one:

```cpp
// ggml/src/ggml-backend.cpp:1453-1625 (simplified)
for (int split_id = 0; split_id < sched->n_splits; split_id++) {
    ggml_backend_t split_backend = sched->backends[split->backend_id];

    // 1. Copy activation tensors from the previous backend into this one
    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        // ... wait for previous event, then copy activation across backends
        ggml_backend_tensor_copy(input, input_cpy);  // line 1575
    }

    // 2. Dispatch this split's subgraph to its backend, asynchronously
    ggml_backend_graph_compute_async(split_backend, &split->graph);  // line 1582

    // 3. Record a CUDA event to track when this split's copies are safe to overwrite
    ggml_backend_event_record(...);  // line 1623
}
```

The loop is **fully sequential**: it copies activations, fires the compute, records an event, and only then moves to the next split. The async call at step 2 means the GPU is working while the CPU moves to the next iteration, but the next iteration immediately waits for the previous event before doing anything — so in practice there is no overlap between layers.

#### What Fionda needs to change

At a high level, Fionda needs to intercept this loop and add a weight staging step **around each split**:

**Before dispatching split N:**
- Wait for the DMA transfer of layer N's weights into VRAM slot A to be complete
- Point the weight tensors for layer N at slot A (update their `data` pointers)
- Kick off an async DMA transfer of layer N+1's weights into VRAM slot B

**After dispatching split N (async compute is now running on GPU):**
- Nothing yet — slot A is still being read by the GPU, slot B's DMA is running in parallel

**Before dispatching split N+1:**
- Wait for slot B's DMA to complete (it has been running while layer N computed on GPU)
- Point layer N+1's weight tensors at slot B
- Free slot A (no copy back needed — RAM was never modified)
- Kick off DMA of layer N+2 into slot A (now free)

This double-buffer pattern means VRAM only needs space for 2 layers of weights at a time, regardless of model size. RAM holds all weights for the entire session and is never modified.

The core structural change this requires is: **weight tensor `data` pointers, which in llama.cpp are set once at load time and never touched again, must be updated at runtime** before each split executes, to point into the rotating VRAM staging buffer. Everything else in the existing graph machinery (op dispatch, activation copies, event synchronization) can remain largely intact.

---

## 5. Performance Protocol: Fionda Ibrido vs Partial Offload

To systematically decide whether "Hybrid Fionda" (keeping a majority of layers static in VRAM and streaming the rest) is faster than `llama.cpp`'s default CPU + GPU partial offload, we define the following analytical protocol based on hardware bandwidths.

### Parameters
* **$L$**: Total number of layers in the model
* **$S_{layer}$**: Size of a single layer's weights (e.g., in MB)
* **$BW_{pcie}$**: Practical bandwidth of the PCIe bus (e.g., ~12 GB/s for PCIe 3.0 x16, ~24 GB/s for PCIe 4.0 x16)
* **$BW_{ram}$**: Practical bandwidth of system RAM (e.g., ~40 GB/s for DDR4 dual-channel)
* **$BW_{vram}$**: Bandwidth of GPU VRAM (e.g., ~448 GB/s for RTX 3060 Ti)
* **$N_{vram}$**: Number of static layers that fit in VRAM (leaving room for KV cache)
* **$N_{stream}$**: Number of layers to offload/stream ($L - N_{vram}$)

### Base Execution Times per Layer
For batch size 1 (autoregressive generation), execution is overwhelmingly memory-bandwidth bound:
1. **$T_{gpu} \approx S_{layer} / BW_{vram}$** (Time to read/compute a layer directly from VRAM)
2. **$T_{pcie} \approx S_{layer} / BW_{pcie}$** (Time to transfer a layer via PCIe)
3. **$T_{cpu} \approx S_{layer} / BW_{ram} + T_{compute\_overhead}$** (Time for the CPU to read from RAM and perform SIMD compute)

### Model 1: Standard Partial Offload (llama.cpp)
Standard offloading runs sequentially. The GPU computes its layers, stops, and the CPU computes the rest.
$$Time_{partial} = (N_{vram} \times T_{gpu}) + (N_{stream} \times T_{cpu})$$

### Model 2: Hybrid Fionda
Fionda groups the computation to pipeline PCIe transfers. For every layer streamed via PCIe, the GPU computes $R$ static layers from VRAM, where $R = \lfloor N_{vram} / N_{stream} \rfloor$.
The time to compute one such group is determined by whether the PCIe transfer is perfectly hidden by the GPU compute time:
$$Time_{group} = \max(T_{pcie}, R \times T_{gpu}) + T_{gpu}$$
*(The final $+ T_{gpu}$ is the compute time of the newly streamed layer).*

The total time per token is:
$$Time_{fionda} = N_{stream} \times Time_{group} + (N_{vram} \bmod N_{stream}) \times T_{gpu}$$

### The Decision Rule
Hybrid Fionda is strictly better when:
**$Time_{fionda} < Time_{partial}$**

Since $T_{gpu}$ is extremely small (typically ~0.3 ms) compared to $T_{pcie}$ and $T_{cpu}$, the inequality simplifies in the worst-case (low $R$) to comparing the time to stream a layer vs computing it on CPU:
**$T_{pcie} < T_{cpu}$**

#### Scenario Examples:
1. **PCIe 3.0 vs DDR4 (Current Hardware):**
   DDR4 (~40 GB/s) is significantly faster than PCIe 3.0 (~12 GB/s).
   $T_{cpu}$ is ~5 ms, while $T_{pcie}$ is ~11 ms.
   **Verdict:** Fionda is *slower* if $N_{stream}$ is large (e.g., Q8_0 model where $N_{stream} = 25$). Fionda only wins if $N_{stream}$ is very small (e.g., Q4_K_M where $N_{stream} = 4$), because the very high $R$ ratio (44/4 = 11) allows the GPU to mask the slow PCIe transfer, and keeps the entire context execution on the GPU avoiding CPU context-switch latency.
   
2. **PCIe 4.0 vs DDR4:**
   PCIe 4.0 (~24 GB/s) closes the gap. $T_{pcie}$ drops to ~5.5 ms, matching $T_{cpu}$.
   **Verdict:** Fionda breaks even on raw time, but becomes vastly superior because keeping all compute on the GPU natively enables **Speculative Decoding**, multiplying the overall throughput.

### Adaptive Placement (Dynamic Interleaving)
To avoid severe stalls caused by rigid block scheduling (e.g., computing a huge block of static layers and then starving the GPU during consecutive dynamic layers), Fionda will use **Adaptive Placement** at load time.
Instead of simply putting the first $N_{vram}$ layers in VRAM and streaming the rest at the end of the graph, Fionda will strategically choose *which* layers live permanently in VRAM and which are streamed based on the target gap $G = \lceil T_{pcie} / T_{gpu} \rceil$. 
For example, if loading a layer takes the equivalent time of computing 7 layers ($G=7$), Fionda will allocate layers 0-6 in VRAM, make layer 7 dynamic, allocate 8-14 in VRAM, make layer 15 dynamic, and so on. 
Since Transformer inference is strictly sequential, this interleaving ensures the model naturally provides exactly the required compute time (the static layers) between dynamic layers to perfectly mask the PCIe latency for the next streamed layer.

---

## Implementation Plan

The plan is structured in phases. Each phase builds on the previous and can be tested independently before proceeding. All changes are confined to the Fionda fork — nothing is sent upstream.

The target scenario throughout: dense model, GGUF format, NVIDIA GPU, Q4 or Q8 quantization, single GPU.

---

### Phase 1 — Load all layer weights into pinned host memory

**Goal:** make all transformer layer weights live in pinned RAM at load time, regardless of `n_gpu_layers`.

**Why pinned (page-locked) memory:** regular `malloc` memory cannot be the source of an async `cudaMemcpyAsync`. The OS may move pages at any time, which would corrupt an in-flight DMA. Pinned memory fixes physical addresses so the copy engine can safely read from them without CPU involvement.

**What to change:**

In `src/llama-model.cpp:2674`, the lambda `get_layer_buft_list` currently returns the GPU buffer list for any layer index above `i_gpu_start`. Change it so that for Fionda mode it always returns the **host (pinned) buffer list** — which already exists as `pimpl->cpu_buft_list` and already includes the pinned host buffer type via `make_cpu_buft_list()` at line 2626. The pinned buffer type comes from `ggml_backend_dev_host_buffer_type(dev)` (wired in `src/llama-context.cpp:299`).

Add a flag (e.g. `fionda_mode`) to `llama_model_params` (`src/llama-model.h`) to gate this behaviour so the rest of llama.cpp is unaffected.

After this phase: the model loads with all layer weights in pinned RAM. Inference will be slow (CPU compute) but correct — a useful baseline to diff against.

---

### Phase 2 — Allocate the VRAM double-buffer

**Goal:** allocate a fixed VRAM region that can hold exactly 2 layers' worth of weights.

**What to build:** a new struct `FiondaVramBuffer` (new file: `src/fionda-scheduler.h`):

```cpp
struct FiondaVramBuffer {
    void * slot[2];        // two VRAM pointers, each of size max_layer_bytes
    size_t max_layer_bytes;
    int    cur_slot;       // which slot is currently being used for compute
};
```

**Where to compute `max_layer_bytes`:** after `load_tensors()` completes, iterate over `pimpl->dev_layer` and sum up the `ggml_nbytes` of every weight tensor per layer. Take the maximum across all layers. This gives the size of the largest single layer — both slots must be this size to be able to hold any layer.

**Where to allocate:** in `llama_context` init (`src/llama-context.cpp`, after the backends are initialized). Call `cudaMalloc(&slot[0], max_layer_bytes)` and `cudaMalloc(&slot[1], max_layer_bytes)`. Store `FiondaVramBuffer` in `llama_context`.

After this phase: VRAM double-buffer exists but is not yet used.

---

### Phase 3 — Create the CUDA copy stream and synchronization events

**Goal:** set up the CUDA primitives needed to pipeline DMA with compute.

**What to create**, added to `FiondaVramBuffer` or a companion `FiondaSyncPrimitives` struct:

```cpp
cudaStream_t copy_stream;          // dedicated to cudaMemcpyAsync transfers
cudaEvent_t  dma_done[2];          // fires when DMA into slot[i] completes
cudaEvent_t  compute_done[2];      // fires when GPU compute on slot[i] finishes
```

**Where to init:** in `llama_context` init, after the VRAM buffer is allocated:

```cpp
cudaStreamCreate(&copy_stream);
cudaEventCreate(&dma_done[0]);
cudaEventCreate(&dma_done[1]);
cudaEventCreate(&compute_done[0]);
cudaEventCreate(&compute_done[1]);
```

The `compute_stream` already exists inside the CUDA backend context (`ggml_backend_cuda_context::stream()` in `ggml/src/ggml-cuda/ggml-cuda.cu`). We do not create a new one — we reference the existing one.

After this phase: the CUDA primitives exist but are not yet wired into the compute path.

---

### Phase 4 — Build the per-layer node map from the compute graph

**Goal:** given a fully-built `ggml_cgraph`, know exactly which graph nodes belong to which transformer layer.

**How layer indices are embedded in tensor names:** the graph callback in `src/llama-context.cpp:2199` names every tensor with its layer index:

```cpp
ggml_format_name(cur, "%s-%d", name, il);  // e.g. "attn_q-3" for layer 3
```

So a graph node named `"attn_q-3"` belongs to layer 3. Nodes with no `-N` suffix belong to the input or output sections.

**What to build:** a function `fionda_build_layer_map(ggml_cgraph * gf)` (in `src/fionda-scheduler.cpp`) that walks `gf->nodes`, parses the trailing `-N` from each name, and returns:

```cpp
struct FiondaLayerMap {
    // for each layer index: the range [node_start, node_end) in gf->nodes
    std::vector<std::pair<int,int>> layer_node_ranges;
};
```

This map is built once per graph (or reused if the graph is reused). It does not change between tokens for the same model.

---

### Phase 5 — Implement `fionda_graph_compute()`

**Goal:** replace the call to `ggml_backend_sched_graph_compute_async()` in `src/llama-context.cpp:2187` with a new function that pipelines DMA with compute at layer granularity.

**Intercept point:** `src/llama-context.cpp:2187`:

```cpp
// current code:
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);

// replace with:
auto status = fionda_graph_compute(sched.get(), gf, &fionda_ctx);
```

**What `fionda_graph_compute()` does (pseudocode):**

```
// Pre-load: DMA layer 0 into slot[0] before the loop starts
cudaMemcpyAsync(slot[0], layer[0].pinned_src, layer[0].size,
                cudaMemcpyHostToDevice, copy_stream)
cudaEventRecord(dma_done[0], copy_stream)

cur = 0,  nxt = 1

for each transformer layer N (0 to num_layers-1):

    // --- BEFORE COMPUTE ---
    // Wait for layer N's DMA to be done before pointing kernels at it
    cudaStreamWaitEvent(compute_stream, dma_done[cur], 0)

    // Redirect weight tensor data pointers for layer N to slot[cur]
    fionda_remap_layer_tensors(layer[N], slot[cur])

    // Kick off DMA for layer N+1 into slot[nxt]
    // (only after slot[nxt]'s previous compute is confirmed done)
    if N+1 < num_layers:
        cudaStreamWaitEvent(copy_stream, compute_done[nxt], 0)
        cudaMemcpyAsync(slot[nxt], layer[N+1].pinned_src, layer[N+1].size,
                        cudaMemcpyHostToDevice, copy_stream)
        cudaEventRecord(dma_done[nxt], copy_stream)

    // --- COMPUTE ---
    // Dispatch the subgraph for layer N (the range from the layer map)
    ggml_backend_graph_compute_async(cuda_backend, &layer_N_subgraph)

    // Record that slot[cur] is done — it becomes free for the next DMA
    cudaEventRecord(compute_done[cur], compute_stream)

    swap(cur, nxt)

// Run the non-layer parts of the graph (input embedding, output head)
// via the normal sched path — they are small and stay on CPU or in VRAM permanently
```

The non-layer parts of the graph (input embedding lookup, final norm, logit projection) are small and not streamed. They run normally through the existing `ggml_backend_sched_compute_splits()` path, which handles them in the splits before and after the transformer body.

---

### Phase 6 — `fionda_remap_layer_tensors()`

**Goal:** update the `data` pointer of every weight tensor in a layer to point into the correct VRAM slot before that layer's kernels are dispatched.

**What it does:** for each weight tensor T belonging to layer N, set:

```cpp
T->data = (char*)slot[cur] + offset_of_T_within_layer;
```

Where `offset_of_T_within_layer` is a fixed offset computed once at model load time, based on the layout we choose when packing the layer's weights into the slot.

**The packing:** at load time, for each layer, record an ordered list of (tensor pointer, byte offset within the slot). The offsets are laid out sequentially with alignment. This packing is done once in a `fionda_build_layer_layout()` function and stored alongside the `FiondaVramBuffer`.

**The conflict with CUDA Graph capture:** `ggml_backend_cuda_graph_compute()` in `ggml/src/ggml-cuda/ggml-cuda.cu:4105` optionally captures a CUDA Graph — a replay-able recording of kernel launches with fixed pointer values. Dynamic `data` pointers break this. The compatibility check at line 2930 (`ggml_cuda_graph_check_compability`) already disables capture under certain conditions. We extend it to also return `false` when any node's weight tensor belongs to the Fionda double-buffer — this disables capture for the streaming layers while leaving it active for any static tensors (e.g. the embedding table).

---

### Phase 7 — End-to-end correctness testing

Before measuring any performance, verify correctness:

1. Run with a **small model that fits entirely in VRAM** (e.g. Llama 3.2 1B in Q4). Compare token-by-token output between unmodified llama.cpp and Fionda. Output must be bit-identical.
2. Run with a **model that exceeds VRAM** to verify the streaming path actually executes without segfaults or CUDA errors.
3. Run `cuda-memcheck` or `compute-sanitizer` on both cases.

See `testing_protocol.md` for the full testing procedure.

---

### Phase 8 — Performance measurement and tuning

Once correctness is confirmed:

1. Measure tokens/sec with and without Fionda on an oversized model.
2. Profile with Nsight Systems to visualize the overlap between the copy stream and the compute stream — confirm DMA and compute are actually concurrent.
3. Tune `max_layer_bytes` alignment (cache line and CUDA memory alignment requirements) to maximise DMA throughput.
4. Experiment with Q4 vs Q8 to validate the expected linear speedup in transfer time.

---

### Future: Speculative Decoding (Phase 9)

Speculative decoding is a separate phase built on top of the working dynamic scheduler. It requires:

- A second small model M_q that fits fully in VRAM
- Running M_q autoregressively γ times to generate candidate tokens
- Running M_p once in batched mode (γ+1 tokens) through the streaming layers
- Applying the acceptance/rejection sampling from Algorithm 1 of the paper

The direct benefit for Fionda: M_p processing γ+1 tokens per layer means each layer stays resident in VRAM ~(γ+1)× longer, giving that much more time for the PCIe DMA of the next layer — directly relaxing the bandwidth bottleneck that limits the base scheduler.

KV cache rewinding (discarding cache entries for rejected tokens) is the main implementation complexity and is where a custom CUDA operation may become necessary.
