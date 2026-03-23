# SuperLlama Research: Compiled Findings

Research conducted 2026-03-23. Sources: Ollama GitHub issues, llama.cpp documentation, HuggingFace model pages, community benchmarks.

---

## 1. WHY OLLAMA IS SLOWER THAN RAW LLAMA.CPP

### 1.1 Architecture Overview

Ollama is a Go application with **five layers of indirection** between the user and the actual inference:

```
HTTP Client
  -> [Layer 1] Gin HTTP Router (server/routes.go)
  -> [Layer 2] Scheduler (server/sched.go)
  -> [Layer 3] LLM Server Manager (llm/server.go) -- spawns subprocess
  -> [Layer 4] Runner Subprocess (runner/ollamarunner/runner.go) -- internal HTTP server
  -> [Layer 5] llama.cpp via CGO (llama/llama.go)
```

Raw llama.cpp has one layer: a single process that serves tokens directly over HTTP.

### 1.2 Top Performance Killers (Ordered by Impact)

#### #1: Conservative VRAM Estimation Causing Silent CPU Fallback

This is the **single largest performance killer**. Ollama's memory estimator frequently underestimates available GPU capacity, causing layers to silently spill to CPU. Even one layer on CPU can cause 2-10x slowdowns.

- Ollama contributor @pdevine confirmed: "The reason for being '1/18th' the speed is almost certainly due to the memory calculation... if even a single layer gets put into system memory instead of on the GPU, the performance will be dramatically impacted." (Issue #11259)
- Qwen3.5-27B has a hybrid Mamba+Attention architecture (48 Mamba layers + 16 Attention layers). Ollama incorrectly treated all layers as full attention layers, creating ~1.5GB of phantom KV cache allocation, preventing full GPU offload. Result: 5-6x slower than llama.cpp. (Issue #14579)
- GPU-to-CPU fallback happens silently with no warning (Issue #14258).
- On Apple Silicon M1: GPU acceleration fell back to CPU between Ollama v0.12.5 and v0.12.9. One user found `num_gpu=1` embedded, getting 7.65 t/s versus 53 t/s unrestricted. (Issue #12976)

#### #2: Ollama's Own Inference Engine Is Less Optimized Than llama.cpp

Ollama is reimplementing models in its own Go-native engine rather than using llama.cpp. New models launch with unoptimized implementations:

- @pdevine (Ollama contributor): "Ollama doesn't use llama-cpp for either gemma3n or gemma3. Most new models are implemented directly inside of Ollama."
- Qwen3.5-35B on Ollama engine: **89.27 t/s** vs llama.cpp engine: **218.86 t/s** -- a **2.45x performance gap** from the engine alone. (Issue #14861, @rick-github)
- On M3 Max: 35 t/s (Ollama engine) vs ~96 t/s (after llama.cpp engine fix) -- 2.7x gap.
- This is a recurring pattern: "gpt-oss had similar performance issues when initially released on the ollama engine" -- @pdevine

#### #3: Two-Process HTTP Architecture (Per-Token Overhead)

Every token traverses: GPU -> C++ runner -> JSON serialize -> HTTP chunked response -> Go HTTP client -> JSON deserialize -> Go processing -> JSON re-serialize -> HTTP to external client.

Raw llama.cpp has a single process with one HTTP hop. Ollama has **two layers of HTTP + JSON serialization** that raw llama.cpp does not have.

The runner subprocess is spawned via: `exec.Command(os.Executable(), "runner", "--ollama-engine", "--model", path, "--port", port)` and communicates over localhost TCP on an ephemeral port (49152-65535).

#### #4: Platform-Specific Build Deficiencies

- ARM64 builds in Ollama 0.14.x did not use NEON SIMD extensions: **10x regression** (10.42 t/s to 0.93 t/s). Fixed in 0.15.1. (Issue #13860)
- Windows build lacked CUDA 12 runners and missed FMA instruction support (`fma=0` vs llama.cpp's `fma=1`). (Issue #6338)
- Thread detection: Ollama detected 8/16 threads while llama.cpp correctly detected 16/16 on the same CPU. (Issue #6338)
- Ollama ships pre-built binaries that may not be optimized for specific hardware, whereas compiling llama.cpp from source gets architecture-specific optimizations (AVX-512, specific CUDA compute capabilities).

#### #5: Conservative Default Parameters

| Parameter | Ollama Default | Problem |
|-----------|---------------|---------|
| Context length | Model's training context (often 32K+) | No auto-sizing to VRAM. If too large, layers silently spill to CPU |
| Flash Attention | OFF (`OLLAMA_FLASH_ATTENTION=false`) | Misses memory savings and speed improvements |
| KV Cache Type | f16 | No quantized KV cache option unless flash attention is on |
| Parallel Requests | 1 (`OLLAMA_NUM_PARALLEL=1`) | Certain architectures force-limited to 1 |
| Thread Count | Auto-detected (sometimes wrong) | Issue #6338: detected 8/16 instead of 16/16 |

### 1.3 Community Benchmarks: Ollama vs llama.cpp

| Source | Model | Ollama | llama.cpp | Gap | Root Cause |
|--------|-------|--------|-----------|-----|------------|
| Issue #6338 | Gemma 2 2B Q4_K_M | ~80 t/s | ~130 t/s | 38% slower | Thread detection, missing FMA, CUDA 12 |
| Issue #14579 | Qwen3.5-27B Q4 | 15-20 t/s | 100 t/s | 5-6x slower | Phantom KV cache, VRAM overflow to CPU |
| Issue #14861 | Qwen3.5-35B | 89 t/s (engine) | 219 t/s (llama.cpp) | 2.45x slower | Unoptimized Ollama engine |
| Issue #14861 | Qwen3.5-35B (M3 Max) | 35 t/s | ~96 t/s | 2.7x slower | Engine optimization needed |
| Issue #13860 | Any model (ARM64) | 0.93 t/s | 10.42 t/s | 10x slower | Missing NEON SIMD |
| Issue #12976 | qwen2.5:32b (M1) | "Unusable" | Full GPU speed | Severe | Metal scheduling bug |
| Issue #14579 | Qwen3.5-27B (RTX 3090) | 12 t/s | 35 t/s | ~3x slower | Context/VRAM miscalculation |

### 1.4 What Ollama Does That Is Unnecessary for Single-Model Use

| Component | Why Unnecessary |
|-----------|----------------|
| Model Registry and Pull System | Model is already on disk |
| Multi-Model Scheduler | No model switching, eviction, or queuing needed |
| Modelfile Parsing | Single pre-configured model; use CLI flags |
| Template System with AST Walking | One template, hardcoded |
| Model Format Detection/Conversion | Direct path to model |
| Two-Process Architecture | Can run inference engine directly |
| API Compatibility Layers | Only need one API format |
| Automatic GPU Layer Decision | Explicit `-ngl` flag when hardware is known |
| VRAM Recovery Monitoring | No model swapping, no `waitForVRAMRecovery()` |
| Reference Counting | Single model always loaded |
| Cloud Proxy | No remote inference |
| Authentication | No registry auth |
| Quantization Support | Model pre-quantized |
| Gin Framework Overhead | `net/http` with 2-3 routes suffices |

---

## 2. LLAMA.CPP OPTIMAL CONFIGURATION

### 2.1 Key Runtime Flags

#### GPU Offloading: `-ngl, --gpu-layers`
- Controls how many transformer layers are offloaded to GPU (Metal on macOS).
- **Apple Silicon recommendation:** Use `auto` or `all`. Unified memory means all layers should be on Metal.
- Benchmark proof: 13.45 t/s at 10 layers vs 131.66 t/s at full 35-layer offload.

#### Flash Attention: `-fa, --flash-attn`
- Default: `auto`. Set to `on` for best results.
- Metal has two kernel variants: half8x8 (default, larger batches) and half4x4 (small batches, `ne01 < 20`).
- Supported head dimensions: 32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 256, 320, 576.
- Reduces memory usage and generally improves performance, especially with larger contexts.
- M2 Ultra official benchmarks all used flash attention enabled.

#### KV Cache Quantization: `-ctk` / `-ctv`
- Default: `f16` for both keys and values.
- Supported types: `f32`, `f16`, `bf16`, `q8_0`, and other quantized formats.
- Use `q8_0` to reduce memory pressure with large contexts. Lower quants save memory but may degrade quality.

#### Batch Sizes: `-b` / `-ub`
- `-b` (logical batch, default 2048): Max tokens per prompt evaluation pass.
- `-ub` (micro-batch, default 512): Actual GPU chunk size.
- Benchmark: 1,436 t/s at batch 128 vs 2,498 t/s at batch 1024 for prompt processing.
- Defaults are well-tuned. For prompt-heavy workloads, increase `-b` to 4096 and `-ub` to 1024.

#### Context Size: `-c, --ctx-size`
- Default: 0 (loads from model metadata).
- **Critical:** Set to the minimum your application needs. Do NOT use model maximum (often 128K+).
- Impact: GPT-OSS 20B on M2 Ultra: 2,713 t/s prompt at 2K context vs 1,557 t/s at 32K; generation drops from 130 to 109 t/s.
- For a coding assistant: 8192-16384 is a reasonable starting point.

#### Threading: `-t` / `-tb`
- Default: -1 (auto-detect).
- On Apple Silicon, most work is done by Metal GPU, so CPU threading matters less.
- Additional flags: `-C` (CPU affinity mask), `--prio` (process priority 0-3), `--numa`.

#### Memory Management
- `--mlock`: Keeps model in RAM, prevents OS swapping. Use for persistent server.
- `--mmap` (default: on): Maps model file directly, faster loading, OS manages paging.
- `--no-mmap --mlock` together: Guarantees no page faults during inference.

#### Additional Performance Flags

| Flag | Description | Default | Recommendation |
|------|-------------|---------|----------------|
| `-np, --parallel` | Parallel decode slots (server) | -1 (auto) | Set to expected concurrent users |
| `-kvo, --kv-offload` | Offload KV cache to GPU | enabled | Keep enabled |
| `--op-offload` | Offload host tensor ops to device | true | Keep enabled |
| `--repack` | Weight repacking for faster inference | enabled | Keep enabled |
| `-fit` | Auto-adjust args to fit device memory | on | Keep enabled |
| `--cont-batching` | Continuous batching (server) | enabled | Always keep enabled in server mode |
| `--cache-prompt` | Reuse KV cache across requests | enabled | Keep enabled |
| `-cmoe, --cpu-moe` | Keep MoE weights on CPU | off | Use for MoE models exceeding GPU memory |

### 2.2 Build-Time Optimizations

#### macOS Apple Silicon (Metal enabled by default)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(sysctl -n hw.ncpu)
```
No extra flags needed. Metal and Accelerate framework are automatic.

Or install via Homebrew:
```bash
brew install llama.cpp
```

#### CUDA (NVIDIA GPUs)
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"
```
Additional CUDA flags:
- `-DGGML_CUDA_FORCE_MMQ=ON`: Custom quantized kernels (lower VRAM, slower for large batches)
- `-DGGML_CUDA_FA_ALL_QUANTS=ON`: Flash attention for all KV cache quant types
- `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128`: NVLink peer access threshold

#### General Build Flags
- `-DCMAKE_BUILD_TYPE=Release`: Always use for production.
- `-DGGML_NATIVE=ON` (default): CPU-native optimizations.
- `-DBUILD_SHARED_LIBS=OFF`: Static linking for deployment.

### 2.3 Server vs CLI Mode

**llama-server is definitively better for wrapping:**

| Aspect | llama-server | llama-cli |
|--------|-------------|-----------|
| Startup cost | One-time model load, persistent | Model loaded every invocation |
| Concurrency | Multiple parallel slots via `-np` | Single session only |
| Batching | Continuous batching across requests | Single-request only |
| Prompt caching | Across requests (default) | Within session only |
| Latency (first token) | Low (model already loaded) | High (model load every call) |
| API | OpenAI-compatible REST | Direct stdio |
| Monitoring | Prometheus `/metrics`, `/health`, `/slots` | None |

### 2.4 Recommended Server Launch Command

```bash
./llama-server \
  -m model.gguf \
  -ngl all \
  -fa on \
  -c 8192 \
  -np 1 \
  --mlock \
  --cache-prompt \
  -b 2048 \
  -ub 512 \
  --host 127.0.0.1 \
  --port 8080
```

Adjust `-c` based on context needs. Add `-ctk q8_0 -ctv q8_0` if memory-constrained.

### 2.5 Apple Silicon Specific Details

#### Metal Backend Capabilities by Chip

| Feature | M1/A14+ | M3+ (Metal3) | M5+ (Metal4) |
|---------|---------|--------------|--------------|
| SIMD group reduction | Yes | Yes | Yes |
| SIMD group matrix multiply | Yes | Yes | Yes |
| BFloat16 | No | Yes | Yes |
| Tensor API | No | No | Yes (default on M5+/A19+) |

#### Metal Environment Variables

| Variable | Purpose |
|----------|---------|
| `GGML_METAL_BF16_DISABLE` | Disable bfloat16 on Metal3 if issues |
| `GGML_METAL_TENSOR_ENABLE` | Force-enable tensor API on pre-M5 |
| `GGML_METAL_NO_RESIDENCY` | Disable residency sets |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | Residency duration (default: 180s) |

#### Unified Memory Architecture
- CPU and GPU share the same physical memory.
- Default: shared buffers (`MTLResourceStorageModeShared`), avoiding CPU-GPU copies.
- Residency sets (macOS 15.0+) help keep model buffers GPU-resident.

### 2.6 M2 Ultra Reference Benchmarks

All with flash attention enabled, 16 threads, all layers on GPU:

| Model | Size | Prompt 2048 (t/s) | Generation 32 (t/s) |
|-------|------|--------------------|---------------------|
| GPT-OSS 20B (MXFP4 MoE) | 11.27 GiB | 2,713 | 130 |
| GPT-OSS 120B (MXFP4 MoE) | 59.02 GiB | 1,649 | 86 |
| Qwen3-Coder 30B (Q8_0) | 30.25 GiB | 2,453 | 79 |
| Qwen2.5-Coder 7B (Q8_0) | 7.54 GiB | 1,566 | 80 |
| Gemma-3 4B (Q4_0) | 2.35 GiB | 2,924 | 134 |

---

## 3. QWEN 3.5 MODEL DETAILS

### 3.1 Model Family Overview

Qwen 3.5 was released early March 2026. Licensed under **Apache 2.0**. Supersedes Qwen3 (May 2025) and Qwen2.5 (Sep 2024).

#### Available Sizes

| Model | Total Params | Active Params | Type | Notes |
|-------|-------------|---------------|------|-------|
| Qwen3.5-0.8B | 0.9B | 0.9B | Dense | Mobile-class |
| Qwen3.5-2B | 2B | 2B | Dense | Small |
| Qwen3.5-4B | 5B | 5B | Dense | Compact |
| Qwen3.5-9B | 10B | 10B | Dense | Mid-range |
| Qwen3.5-27B | 28B | 28B | Dense | Large |
| Qwen3.5-35B-A3B | 36B | 3B | MoE (256 experts, 8 routed + 1 shared) | 35B quality at 3B speed |
| Qwen3.5-122B-A10B | 125B | 10B | MoE | Flagship MoE |
| Qwen3.5-397B-A17B | 403B | 17B | MoE | Datacenter-class |

### 3.2 Architecture: Hybrid Gated DeltaNet + Attention

Qwen3.5 is **NOT a standard transformer**:

- Layout: `N x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`
- Uses **Gated Delta Networks** (linear attention) for 75% of layers and standard **Gated Attention** for 25%
- MoE variants: 256 experts, 8 routed + 1 shared active per token
- Trained with Multi-Token Prediction (MTP)
- Multimodal capable (vision encoder built-in)

### 3.3 Context Window

- **Native:** 262,144 tokens (256K)
- **Extended:** Up to 1,010,000 tokens (~1M) via YaRN RoPE scaling
- RoPE config: `rope_type: "yarn"`, `rope_theta: 10000000`, `partial_rotary_factor: 0.25`, `factor: 4.0`

### 3.4 Chat Template (ChatML Format)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
<think>
...reasoning...
</think>

Response here<|im_end|>
```

**Key tokens:**

| Token | ID | Role |
|-------|-----|------|
| `<|im_end|>` | 248046 | EOS token |
| `<|endoftext|>` | 248044 | Pad token |
| BOS | None | Not used (`add_bos_token: false`) |

- **Tokenizer class:** Qwen2Tokenizer
- **Thinking mode:** On by default. Model outputs `<think>...</think>` blocks. Disable via `chat_template_kwargs: {"enable_thinking": false}`.
- llama.cpp auto-detects ChatML format via `<|im_start|>` in embedded template.

**Tool calling tokens:** `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`

### 3.5 Recommended Sampling Parameters

Thinking mode (general tasks):
```
temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5
```

Non-thinking mode (instruct):
```
temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5
```

### 3.6 GGUF Quantization Sizes

All GGUFs provided by **unsloth** on HuggingFace. "UD-" prefix = Unsloth Dynamic 2.0 (claims superior accuracy).

#### Qwen3.5-9B (Dense)

| Quant | Size | Quality |
|-------|------|---------|
| Q3_K_M | 4.67 GB | Acceptable |
| **Q4_K_M** | **5.68 GB** | **Good (recommended)** |
| Q5_K_M | 6.58 GB | Very good |
| Q6_K | 7.46 GB | Excellent |
| Q8_0 | 9.53 GB | Near-original |
| BF16 | 17.9 GB | Original |

#### Qwen3.5-27B (Dense)

| Quant | Size | Quality |
|-------|------|---------|
| Q3_K_M | 13.5 GB | Acceptable |
| **Q4_K_M** | **16.7 GB** | **Good (recommended)** |
| Q5_K_M | 19.6 GB | Very good |
| Q6_K | 22.5 GB | Excellent |
| Q8_0 | 28.6 GB | Near-original |
| BF16 | 53.8 GB | Original |

#### Qwen3.5-35B-A3B (MoE)

| Quant | Size | Quality |
|-------|------|---------|
| Q3_K_M | 16.4 GB | Acceptable |
| **Q4_K_M** | **22 GB** | **Good (recommended)** |
| Q5_K_M | 26.2 GB | Very good |
| Q8_0 | 36.9 GB | Near-original |
| BF16 | 69.4 GB | Original |

### 3.7 Model Recommendations by Apple Silicon RAM

| Unified Memory | Best Model | Quant | File Size | Notes |
|---------------|------------|-------|-----------|-------|
| 8 GB | Qwen3.5-9B | Q4_K_M | 5.68 GB | Tight fit, ~2GB for context |
| 16 GB | Qwen3.5-9B | Q8_0 | 9.53 GB | Comfortable, great quality |
| 24 GB | Qwen3.5-27B | Q4_K_M | 16.7 GB | Best quality/size sweet spot |
| 32 GB | Qwen3.5-27B | Q6_K | 22.5 GB | Excellent quality, room for context |
| 36 GB | Qwen3.5-35B-A3B | Q4_K_M | 22 GB | MoE: 35B quality at 3B speed |
| 48 GB | Qwen3.5-35B-A3B | Q8_0 | 36.9 GB | MoE near-lossless |
| 64 GB | Qwen3.5-27B | BF16 | 53.8 GB | Full precision, or 35B Q8 + huge context |

### 3.8 Sweet Spot Picks

**Primary: Qwen3.5-27B at Q4_K_M (16.7 GB)** -- 27B dense parameters, strong benchmarks across coding/reasoning/general tasks, fits on 24GB+ Mac. Dense architecture = simpler, more predictable performance.

**Runner-up: Qwen3.5-35B-A3B at Q4_K_M (22 GB)** -- MoE with only 3B active params = extremely fast inference despite 35B quality. MMLU-Pro 85.3, SWE-bench 69.2. Needs ~22GB RAM. **Caveat:** All 35B params must be in memory even though only 3B are active.

**Budget: Qwen3.5-9B at Q4_K_M (5.68 GB)** -- Fits on 8GB Macs.

### 3.9 llama.cpp Support Status (March 2026)

- **Gated DeltaNet op:** Merged March 7, 2026 (PR #19504)
- **Vulkan backend:** DeltaNet support merged March 12, 2026 (PR #20334)
- **Metal backend:** Being worked on, may need chunked processing for older Apple GPUs (PR #20244)
- **MTP (Multi-Token Prediction):** In progress (PR #20700), not yet merged

**Known issues:**
- Thinking mode + tool calling conflict: `<think>` tags interfere with tool call parsing (#20837)
- Default thinking behavior: model outputs `<think>` blocks by default; may need stripping (#20833)
- Interrupted completion bug: resuming can inject spurious `<think>` tags (#20768)

### 3.10 Download URLs

| Model | URL |
|-------|-----|
| Qwen3.5-9B GGUF | `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF` |
| Qwen3.5-27B GGUF | `https://huggingface.co/unsloth/Qwen3.5-27B-GGUF` |
| Qwen3.5-35B-A3B GGUF | `https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF` |
| Qwen3.5-4B GGUF | `https://huggingface.co/unsloth/Qwen3.5-4B-GGUF` |
| Qwen3.5-0.8B GGUF | `https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF` |
| Qwen3.5-122B-A10B GGUF | `https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF` |
| Official collection | `https://huggingface.co/collections/Qwen/qwen3.5` |

---

## 4. OLLAMA ARCHITECTURE BLOAT -- WHAT CAN BE ELIMINATED

### 4.1 Full Request Flow in Ollama

```
HTTP Client
  |
  v
[Layer 1] Gin HTTP Router (server/routes.go)
  - allowedHostsMiddleware, CORS
  - ChatHandler/GenerateHandler parses JSON
  - Validates model reference
  - Cloud model proxy check
  |
  v
[Layer 2] Scheduler (server/sched.go)
  - scheduleRunner(): validates model, loads config, merges options
  - GetRunner() checks loaded map
  - If not loaded: queues to pendingReqCh, estimates memory, may evict other models
  - Calls Load() -> GPU layer assignment -> StartRunner()
  |
  v
[Layer 3] LLM Server Manager (llm/server.go)
  - Validates NumCtx against training context
  - Spawns SUBPROCESS: exec.Command(os.Executable(), "runner", "--ollama-engine", "--model", path, "--port", port)
  - Subprocess binds to localhost ephemeral port (49152-65535)
  - Parent monitors subprocess via goroutine
  |
  v
[Layer 4] Runner Subprocess (runner/ollamarunner/runner.go)
  - HTTP server: /api/load, /api/completion, /api/embeddings, /api/health
  - Manages KV cache, batch scheduling, parallel sequences
  - Calls model.Forward() for inference
  |
  v
[Layer 5] llama.cpp via CGO (llama/llama.go)
  - C.llama_model_load_from_file(), C.llama_decode(), C.llama_tokenize()
  - Vendored llama.cpp with patches in llama/patches/
```

### 4.2 Ollama Server Directory: 40 Files

Core components: routes.go, model.go, model_resolver.go, create.go, download.go, images.go, sched.go, prompt.go, logprob.go, quantization.go, auth.go, cloud_proxy.go, inference_request_log.go, upload.go, plus test files and platform-specific code.

### 4.3 Components Unnecessary for Single-Model Use

| Component | Files/Packages | Why Unnecessary |
|-----------|---------------|----------------|
| Multi-model scheduler | server/sched.go | No model switching, eviction, queuing |
| Model registry/pulling | server/download.go, manifests | Model is local |
| Model creation | server/create.go, Modelfile parser | Single pre-configured model |
| Cloud proxy | server/cloud_proxy.go | No remote inference |
| Authentication | server/auth.go | No registry auth |
| Push/copy/delete handlers | Routes in routes.go | No model management |
| Memory estimation | Multi-GPU layout logic | Fixed single-model config |
| VRAM recovery monitoring | waitForVRAMRecovery() | No model swapping |
| Reference counting | runnerRef.refCount | Single model always loaded |
| Eviction policy | findRunnerToUnload() | Never unload |
| Model resolution | server/model_resolver.go | Direct path to model |
| Template auto-detection | detectChatTemplate() | Hardcode the template |
| Gin framework | Full web framework | net/http with 2-3 routes suffices |
| Multiple runner engines | llamarunner, mlxrunner, imagegen | Only need one |
| Subprocess architecture | Re-exec as child process | Link directly via CGO |
| Image/blob handlers | Blob upload/download routes | Not needed |
| Quantization support | server/quantization.go | Model pre-quantized |
| Upload system | server/upload.go | No uploads |

### 4.4 Ollama Default Parameters

From `DefaultOptions()` in `api/types.go`:

| Parameter | Default Value |
|-----------|--------------|
| NumPredict | -1 (unlimited) |
| NumKeep | 4 |
| Temperature | 0.8 |
| TopK | 40 |
| TopP | 0.9 |
| MinP | 0.0 |
| RepeatLastN | 64 |
| RepeatPenalty | 1.1 |
| NumCtx | Model default (often 32K+) |
| NumBatch | 512 |
| NumGPU | -1 (auto) |

From `envconfig/config.go`:

| Parameter | Default |
|-----------|---------|
| OLLAMA_KEEP_ALIVE | 5 minutes |
| OLLAMA_NUM_PARALLEL | 1 |
| OLLAMA_MAX_LOADED_MODELS | 0 (unlimited) |
| OLLAMA_MAX_QUEUE | 512 |
| OLLAMA_FLASH_ATTENTION | false |
| OLLAMA_LOAD_TIMEOUT | 5 minutes |

### 4.5 Key Insight: llama.cpp Is Vendored and Patched

Ollama does NOT use `llama-server` (the standalone llama.cpp HTTP server). Instead it vendored, patched, and compiled llama.cpp into the Ollama binary itself via CGO. The Ollama binary re-executes itself in "runner" mode as a subprocess.

---

## 5. MINIMAL WRAPPER APPROACH -- BEST STRATEGY

### 5.1 Critical Finding: llama-server Already Does 90% of What Ollama Does

llama.cpp's built-in server (`llama-server`) already provides:

- **OpenAI-compatible API:** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
- **Anthropic-compatible API:** `/v1/messages`
- **Additional endpoints:** `/completion`, `/embedding`, `/reranking`, `/infill`, `/tokenize`, `/detokenize`, `/health`, `/slots`, `/metrics` (Prometheus)
- **Built-in web UI** at `http://localhost:8080`
- **Streaming** with continuous/dynamic batching
- **Concurrent requests** via `-np N`
- **Chat template support** including custom Jinja templates
- **Auto-download from HuggingFace:** `llama-server -hf <user>/<model>[:quant]`

### 5.2 What Ollama Actually Adds Over llama-server

1. A model registry/catalog with short names (`ollama run llama3`)
2. A persistent background daemon
3. Modelfile format (parameters + system prompt + model reference)
4. Model management (list, pull, delete, copy)
5. Its own API
6. A Go-based abstraction layer

### 5.3 Existing Minimal Alternatives

#### llamafile -- Single-File Executable
- Combines llama.cpp with Cosmopolitan Libc for cross-platform polyglot binaries.
- Near-native performance on Apple Silicon (~55 tok/s for LLaVA 7B on M2).
- **Limitations:** Lags behind upstream llama.cpp, bundling weights into multi-GB executable is awkward.
- **Verdict:** Interesting distribution model but overkill for single-platform use.

#### llama-cpp-python -- Python Bindings
- Uses ctypes (direct C FFI) -- thinnest possible binding layer.
- Near-zero overhead for inference (C code runs natively).
- Includes OpenAI-compatible API server.
- **Verdict:** Good if Python is already in the stack but adds a Python dependency.

#### Other Projects

| Project | Language | Notes |
|---------|----------|-------|
| node-llama-cpp | TypeScript/C++ | N-API bindings, auto GPU detect. Heavy Node.js dependency. |
| mistral.rs | Rust | Full Rust inference engine (not a wrapper). Impressive but a whole new engine. |
| rustformers/llm | Rust | ARCHIVED. Dead project. |
| HuggingFaceModelDownloader | Go | Fast parallel downloader (16 connections/file). Could serve as download component. |

### 5.4 The Minimum Viable Wrapper

The simplest path to "Ollama UX with zero overhead":

1. **A model alias file** (~20 lines of config mapping short names to HuggingFace GGUF repos)
2. **A wrapper script** (~100-200 lines) that:
   - Reads the alias file
   - Detects hardware (GPU type, VRAM, CPU cores)
   - Computes optimal flags (`-ngl`, `-c`, `-t`, `-b`)
   - Execs `llama-server` or `llama-cli` (which handle downloading, caching, inference, and serving)
3. **A pre-installed llama.cpp** (`brew install llama.cpp` or prebuilt binary)

```
superllama run qwen3.5
  -> resolves "qwen3.5" to HF repo via config
  -> exec llama-server -hf unsloth/Qwen3.5-27B-GGUF:Q4_K_M -ngl 99 -c 8192 -fa on --mlock
```

### 5.5 Language Choice

| Language | Pros | Cons |
|----------|------|------|
| Shell script | Zero dependencies, instant startup, ~100 lines | No Windows, limited model registry |
| Python | Easy model registry, HF integration, llama-cpp-python | Adds Python dependency, ~200ms startup |
| Go | Single binary, cross-platform, fast startup | Heavier build, more code than needed |
| Rust | Single binary, zero-cost abstractions | Overkill for a thin wrapper |

**Recommendation:** Shell script for MVP, potentially graduating to Go or Rust binary for cross-platform support later.

### 5.6 The Core Insight

**Do not wrap the inference engine at all. Just call it with the right flags.**

Everything -- the inference engine, OpenAI-compatible API, web UI, model downloading, GGUF parsing, GPU acceleration, chat templates -- is already built into llama.cpp. Ollama's ~50K lines of Go mostly reimplements or wraps functionality that `llama-server` already provides natively.

The path of least resistance to maximum performance is: a thin layer that resolves model names to HuggingFace repos, detects hardware, computes optimal flags, and execs `llama-server`.

---

## REFERENCE LINKS

### Ollama Source Code
- Runner management: https://github.com/ollama/ollama/blob/main/llm/server.go
- Scheduler: https://github.com/ollama/ollama/blob/main/server/sched.go
- HTTP routes: https://github.com/ollama/ollama/blob/main/server/routes.go
- Inference loop: https://github.com/ollama/ollama/blob/main/runner/ollamarunner/runner.go
- Environment config: https://github.com/ollama/ollama/blob/main/envconfig/config.go

### Ollama GitHub Issues
- Issue #14861 -- Qwen3.5 2.45x gap: https://github.com/ollama/ollama/issues/14861
- Issue #14579 -- Qwen3.5 5-6x gap, phantom KV cache: https://github.com/ollama/ollama/issues/14579
- Issue #6338 -- Gemma 2 38% gap: https://github.com/ollama/ollama/issues/6338
- Issue #11259 -- Developer confirms memory estimation root cause: https://github.com/ollama/ollama/issues/11259
- Issue #12976 -- Apple Silicon GPU fallback: https://github.com/ollama/ollama/issues/12976
- Issue #13860 -- ARM64 10x regression: https://github.com/ollama/ollama/issues/13860
- Issue #12353 -- No auto-sizing context to VRAM: https://github.com/ollama/ollama/issues/12353

### llama.cpp Documentation
- README: https://github.com/ggml-org/llama.cpp/blob/master/README.md
- Build guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- CLI flags: llama.cpp/tools/cli/README.md
- Server docs: llama.cpp/tools/server/README.md
- Server architecture: llama.cpp/tools/server/README-dev.md
- Benchmarking: llama.cpp/tools/llama-bench/README.md
- Metal shaders: llama.cpp/ggml/src/ggml-metal/
- M2 Ultra benchmarks: llama.cpp/benches/mac-m2-ultra/

### Qwen3.5 Model Downloads
- https://huggingface.co/unsloth/Qwen3.5-27B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- https://huggingface.co/collections/Qwen/qwen3.5

---

## 6. CLAUDE CODE INTEGRATION WITH LOCAL MODELS

### 6.1 How It Works

Claude Code communicates exclusively via the Anthropic Messages API (`POST /v1/messages`).
llama-server implements this endpoint with full compatibility, converting Anthropic format
to OpenAI format internally via `convert_anthropic_to_oai()` and converting responses back
via `to_json_anthropic()` / `to_json_anthropic_stream()`.

### 6.2 llama-server's /v1/messages Implementation

**Source files:**
- `tools/server/server-common.cpp` — `convert_anthropic_to_oai()` (request parsing)
- `tools/server/server-task.cpp` — `to_json_anthropic()`, `to_json_anthropic_stream()` (response formatting)
- `tools/server/server.cpp` — route registration

**Supported features:**
- Full SSE streaming: `message_start`, `content_block_start`, `content_block_delta` (text_delta, input_json_delta, thinking_delta, signature_delta), `content_block_stop`, `message_delta`, `message_stop`
- Tool use: `tool_use` and `tool_result` content blocks with proper IDs
- Thinking/extended thinking blocks with `thinking_budget_tokens`
- System prompts (string and array format)
- Image input (base64 and URL)
- Token counting: `POST /v1/messages/count_tokens`
- All sampling parameters: temperature, top_p, top_k, stop_sequences, max_tokens

**Not supported / differences:**
- Anthropic-style prompt caching (`cache_control` blocks) — not implemented
- `tool_choice` — accepted but may not match Anthropic's exact behavior
- Signature field in thinking blocks sends empty string
- Token counts are approximate (model tokenizer, not Anthropic's)

### 6.3 Required Environment Variables

From Ollama's `cmd/launch/claude.go` source and Anthropic docs:

**Minimum required:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8080   # llama-server URL
ANTHROPIC_API_KEY=""                        # empty (no real key needed)
ANTHROPIC_AUTH_TOKEN=superllama             # any non-empty value
```

**Model routing (set all to same model for local):**
```bash
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.5
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.5
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.5
CLAUDE_CODE_SUBAGENT_MODEL=qwen3.5
```

**Recommended for local use:**
```bash
CLAUDE_CODE_AUTO_COMPACT_WINDOW=65536         # match your actual context size
CLAUDE_CODE_ATTRIBUTION_HEADER=0              # not needed locally
DISABLE_PROMPT_CACHING=1                      # llama-server doesn't support it
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1    # no telemetry
```

**Optional tuning:**
```bash
CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1       # disable adaptive reasoning
MAX_THINKING_TOKENS=4096                      # cap thinking budget
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80            # compact at 80% of context
CLAUDE_CODE_SIMPLE=1                          # minimal system prompt, fewer tools
CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192            # cap output per request
```

### 6.4 Context Window Requirements

Claude Code recommends **64K+ tokens**. The system prompt + tool definitions alone consume
significant tokens (~4-8K depending on tools enabled).

**Auto-compaction:** Claude Code auto-compacts at ~95% of context window (configurable via
`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`). With smaller windows, compaction happens frequently,
losing conversation history but remaining functional.

### 6.5 Known Issues (llama.cpp + Claude Code)

| Issue | Status | Impact |
|-------|--------|--------|
| Thinking blocks dropped during conversion | Fixed (PR #20120) | Was breaking thinking mode |
| "Cannot pass both content and thinking" crash | Fixed (PR #20500) | Was crashing server |
| Only last token returned in streaming | Fixed (issue #18613) | Was breaking Claude Code CLI |
| System messages with `x-anthropic-` prefix slow | Open (#20623) | Performance degradation |
| Prompt cache invalidation on long conversations | Open (#19794) | Full reprocessing on cache miss |
| tool_choice parameter handling differs | Unclear | May affect tool selection |

### 6.6 What `ollama launch claude` Does (for reference)

From `cmd/launch/claude.go`:
1. Finds `claude` binary (PATH or `~/.claude/local/claude`)
2. Sets all environment variables above
3. For cloud models, sets `CLAUDE_CODE_AUTO_COMPACT_WINDOW` based on model's known context limit
4. Execs `claude --model <model>` with passthrough args

### 6.7 Model Requirements for Claude Code

The model must reliably:
1. Output well-formed JSON for tool_use arguments (Claude Code sends ~20+ tools)
2. Handle large system prompts (4-8K tokens of tool definitions)
3. Follow instructions about when to use tools vs respond directly
4. Handle multi-turn conversations with interleaved tool results

Qwen 3.5 supports native tool calling (`<tool_call>` tokens) and has strong instruction following,
making it one of the better local models for Claude Code use.
