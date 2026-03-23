# SuperLlama — Build Plan v3

## Goal

One pip install. One command to run. Cross-platform (macOS, Linux, Windows). 2-3x faster than Ollama.

```bash
# Install (any platform)
pip install superllama

# Run as API server + web UI
superllama serve

# Run as Claude Code — full agentic coding, locally, free
superllama code

# Interactive chat
superllama chat
```

## Implementation: Python package (stdlib only, ~300 lines)

```
superllama/
├── pyproject.toml          # packaging + CLI entry point
├── superllama/
│   ├── __init__.py         # version
│   └── cli.py              # entire CLI (~300 lines)
├── plan.md
└── RESEARCH.md
```

Python is ONLY the launcher (~100ms startup). Inference path is 100% C++:
```
Python (start llama-server) → exits
Claude Code → HTTP → llama-server (C++) → GPU
```

---

## Key Discovery: llama-server has a complete Anthropic API

llama-server implements `/v1/messages` with full Anthropic Messages API compatibility:

- **Streaming SSE events**: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`
- **Tool use**: `tool_use` and `tool_result` content blocks (Claude Code's core mechanism)
- **Thinking/extended thinking**: `thinking` content blocks with `thinking_budget_tokens`
- **Image input**: base64 and URL sources
- **System prompts**: string and array formats
- **Token counting**: `/v1/messages/count_tokens`

This means Claude Code can connect directly to llama-server. No Ollama needed.

```
Ollama path:      Claude Code → Ollama (5 layers of overhead) → llama.cpp → GPU
SuperLlama path:  Claude Code → llama-server (direct) → GPU
```

Source: `tools/server/server-common.cpp` (`convert_anthropic_to_oai()`),
`tools/server/server-task.cpp` (`to_json_anthropic_stream()`)

---

## Architecture (Two Layers)

```
┌──────────────────────────────────────────────────────────┐
│  Layer 2: SMART CLIENT (Claude Code)                     │
│  ─────────────────────────────────────                   │
│  Claude Code handles:                                    │
│  • Tool use (read/edit files, run commands, grep, glob)  │
│  • Context window management (auto-compress old msgs)    │
│  • Agentic loops (plan → execute → verify → iterate)     │
│  • Codebase awareness (project structure, file search)   │
│  • Session memory                                        │
│  • Git integration                                       │
│                                                          │
│  Connects via: ANTHROPIC_BASE_URL=http://localhost:8080  │
└──────────────────────────┬───────────────────────────────┘
                           │ Anthropic Messages API
                           │ /v1/messages (SSE streaming)
                           │ tool_use / tool_result blocks
┌──────────────────────────┴───────────────────────────────┐
│  Layer 1: MODEL SERVER (SuperLlama → llama-server)       │
│  ─────────────────────────────────────────────────        │
│  Python CLI (pip install superllama) handles:            │
│  • Cross-platform: macOS, Linux, Windows                 │
│  • Hardware detection (chip, RAM, OS, GPU)               │
│  • Auto-install llama-server binary from GitHub releases │
│  • Optimal flag computation (-ngl all, -fa on, etc.)     │
│  • Model selection (Qwen 3.5 sizes)                      │
│  • Starts llama-server as subprocess (zero overhead)     │
│                                                          │
│  llama-server provides:                                  │
│  • Anthropic-compatible API (/v1/messages)               │
│  • OpenAI-compatible API (/v1/chat/completions)          │
│  • Built-in web UI (localhost:8080)                      │
│  • Auto-download from HuggingFace (-hf flag)             │
│  • Metal/CUDA GPU acceleration                           │
│  • Streaming, continuous batching, prompt caching         │
└──────────────────────────────────────────────────────────┘
```

**Layer 1** is what we build (~350 lines of bash).
**Layer 2** is Claude Code — already built, we just connect to it.

---

## What Ships

```
superllama/
├── pyproject.toml        # pip package config
├── superllama/
│   ├── __init__.py       # version
│   └── cli.py            # entire CLI (~300 lines, stdlib only)
├── plan.md               # this file
└── RESEARCH.md           # background research
```

---

## Install Flow (`install.sh`)

### Step 1: Detect Platform

```
OS:   macOS / Linux
Arch: arm64 / x86_64
RAM:  via sysctl (macOS) or /proc/meminfo (Linux)
Chip: Apple Silicon model (M1/M2/M3/M4/M5) via sysctl
```

### Step 2: Install llama.cpp

**macOS (preferred):**
```bash
brew install llama.cpp    # Metal enabled by default
```
Fallback (no brew): download prebuilt from GitHub releases:
```
https://github.com/ggml-org/llama.cpp/releases/latest → llama-b{VER}-bin-macos-arm64.tar.gz
```

**Linux:**
```
https://github.com/ggml-org/llama.cpp/releases/latest → llama-b{VER}-bin-ubuntu-x64.tar.gz
```

Extract `llama-server` + `llama-cli` to `~/.local/bin/` or `/usr/local/bin/`.

### Step 3: Check for Claude Code (optional)

```bash
if command -v claude &>/dev/null; then
  echo "✓ Claude Code detected — 'superllama code' will be available"
  HAS_CLAUDE=true
else
  echo "ℹ Claude Code not found. Install it for agentic coding:"
  echo "  npm install -g @anthropic-ai/claude-code"
  HAS_CLAUDE=false
fi
```

### Step 4: Present Model Choice

Auto-detect RAM, show what fits. For `superllama code` (Claude Code mode), recommend
models that can handle 64K context.

```
SuperLlama Setup
────────────────
Detected: Apple M2 Pro, 32 GB RAM

Pick your model:

  SIZE       MODEL               QUANT     DISK     SPEED    CLAUDE CODE
  ────────────────────────────────────────────────────────────────────────
  small      Qwen3.5-4B          Q4_K_M    3.1 GB   ~120 t/s   limited (8K ctx max)
  medium     Qwen3.5-9B          Q4_K_M    5.7 GB    ~80 t/s   basic (16K ctx)
  large      Qwen3.5-27B         Q4_K_M   16.7 GB    ~40 t/s   good (32K ctx)      ★
  large+     Qwen3.5-27B         Q8_0     28.6 GB    ~35 t/s   limited (16K ctx)
  turbo      Qwen3.5-35B-A3B     Q4_K_M   22.0 GB    ~60 t/s   good (32K ctx)

Choice [small/medium/large/large+/turbo] (default: large):
```

**Context budget per model** (RAM = model weight + KV cache):
- KV cache for 64K ctx ≈ 2-4 GB (varies by model + quant)
- KV cache for 32K ctx ≈ 1-2 GB
- KV cache for 8K ctx ≈ 0.3-0.5 GB

Auto-compute max context: `MAX_CTX = (AVAILABLE_RAM - MODEL_SIZE - 4GB_OS_HEADROOM) / KV_PER_TOKEN`

### Step 5: Write Config + Install CLI

```bash
mkdir -p ~/.superllama

cat > ~/.superllama/config << EOF
MODEL_SIZE=large
HF_REPO=unsloth/Qwen3.5-27B-GGUF
QUANT=Q4_K_M
CTX_SIZE=32768
PORT=8080
HOST=127.0.0.1
EOF

install -m 755 superllama /usr/local/bin/superllama
```

### Step 6: Pre-download Model (prompted)

```
Download model now? (16.7 GB) [Y/n]:
```

Uses `llama-cli -hf <repo>:<quant> --no-display-prompt -p "test" -n 1` to trigger download
then immediately exit. Fast, no server startup.

### Step 7: Done

```
✓ llama.cpp installed (llama-server, llama-cli)
✓ superllama installed to /usr/local/bin/superllama
✓ Model: Qwen3.5-27B (Q4_K_M, 16.7 GB)
✓ Config: ~/.superllama/config

Usage:
  superllama              Start API server + web UI on localhost:8080
  superllama chat         Interactive terminal chat
  superllama code         Agentic coding with Claude Code (free, local)
  superllama --help       See all options
```

---

## CLI Design (`superllama`)

### Commands

```bash
superllama                     # start server (Anthropic + OpenAI API + web UI)
superllama chat                # interactive terminal chat via llama-cli
superllama code                # launch Claude Code connected to local Qwen 3.5
superllama bench               # benchmark: show actual tok/s on your hardware
superllama config              # show current config
superllama config set KEY VAL  # change a setting
superllama switch <size>       # switch model size (tiny/small/medium/large/large+/turbo)
superllama stop                # kill running superllama server
superllama status              # show server status, loaded model, memory usage
superllama update              # update llama.cpp to latest
superllama --help              # help
```

### `superllama` (server mode)

```bash
#!/bin/bash
source ~/.superllama/config

FLAGS=(
  -hf "${HF_REPO}:${QUANT}"
  -ngl all                    # all layers on GPU
  -fa on                      # flash attention
  -c "$CTX_SIZE"              # context window
  -np 1                       # single user
  --mlock                     # pin model in RAM
  --cache-prompt              # reuse KV cache
  -b 2048                     # batch size
  -ub 512                     # micro-batch
  --host "$HOST"
  --port "$PORT"
)

echo "SuperLlama starting Qwen3.5 on http://${HOST}:${PORT}"
echo "Web UI:     http://${HOST}:${PORT}"
echo "OpenAI API: http://${HOST}:${PORT}/v1/chat/completions"
echo "Claude API: http://${HOST}:${PORT}/v1/messages"
echo ""
echo "Press Ctrl+C to stop"

exec llama-server "${FLAGS[@]}"
```

### `superllama chat` (interactive)

```bash
exec llama-cli \
  -hf "${HF_REPO}:${QUANT}" \
  -ngl all -fa on -c "$CTX_SIZE" --mlock \
  -cnv \
  --chat-template chatml
```

### `superllama code` (Claude Code integration) — THE KEY FEATURE

```bash
# Check Claude Code is installed
if ! command -v claude &>/dev/null; then
  echo "Claude Code not found. Install it:"
  echo "  npm install -g @anthropic-ai/claude-code"
  exit 1
fi

# Start llama-server in background if not already running
if ! curl -s "http://${HOST}:${PORT}/health" | grep -q '"status":"ok"'; then
  echo "Starting SuperLlama server..."

  # For Claude Code mode, maximize context (need 64K+ ideally)
  CODE_CTX=$(compute_max_context)  # auto-size based on available RAM

  llama-server \
    -hf "${HF_REPO}:${QUANT}" \
    -ngl all -fa on -c "$CODE_CTX" --mlock \
    --cache-prompt -b 2048 -ub 512 \
    --host "$HOST" --port "$PORT" \
    > ~/.superllama/server.log 2>&1 &
  SERVER_PID=$!
  echo $SERVER_PID > ~/.superllama/server.pid

  # Wait for server to be ready
  echo "Waiting for model to load..."
  while ! curl -s "http://${HOST}:${PORT}/health" | grep -q '"status":"ok"'; do
    sleep 1
  done
  echo "Server ready."
fi

# Launch Claude Code pointed at local SuperLlama
# These env vars match what `ollama launch claude` sets (from ollama source: cmd/launch/claude.go)
echo "Starting Claude Code with local Qwen 3.5..."
echo "(All inference is local — no API costs)"
echo ""

MODEL_NAME="qwen3.5"

ANTHROPIC_BASE_URL="http://${HOST}:${PORT}" \
ANTHROPIC_API_KEY="" \
ANTHROPIC_AUTH_TOKEN="superllama" \
ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_NAME" \
ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_NAME" \
ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_NAME" \
CLAUDE_CODE_SUBAGENT_MODEL="$MODEL_NAME" \
CLAUDE_CODE_AUTO_COMPACT_WINDOW="$CODE_CTX" \
CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
DISABLE_PROMPT_CACHING="1" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1" \
  exec claude --model "$MODEL_NAME" "$@"
```

**Environment variables explained:**

| Variable | Value | Why |
|----------|-------|-----|
| `ANTHROPIC_BASE_URL` | `http://localhost:8080` | Point Claude Code at our llama-server |
| `ANTHROPIC_API_KEY` | `""` (empty) | No real API key needed for local |
| `ANTHROPIC_AUTH_TOKEN` | `"superllama"` | Auth header value (can be anything) |
| `ANTHROPIC_DEFAULT_*_MODEL` | `"qwen3.5"` | Route ALL model tiers (opus/sonnet/haiku) to same local model |
| `CLAUDE_CODE_SUBAGENT_MODEL` | `"qwen3.5"` | Sub-agents also use local model |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | context size | Tell Claude Code our actual context limit so it compacts properly |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `"0"` | Disable attribution header (not needed locally) |
| `DISABLE_PROMPT_CACHING` | `"1"` | llama-server doesn't support Anthropic-style cache_control blocks |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `"1"` | No telemetry/autoupdater (we're local) |

**What the user gets:**
- Full Claude Code UI in their terminal
- Tool use: file reading, editing, shell commands, grep, glob — all work
- Context management: Claude Code auto-compacts at ~95% of context window
- Agentic loops: plan → execute → verify → iterate
- Git integration: commits, diffs, PRs
- All powered by local Qwen 3.5, zero cost

**Known issues to watch for (from llama.cpp GitHub):**
- Thinking blocks had regressions (dropped blocks, crashes) — fixed in recent builds, but fragile
- System messages starting with `x-anthropic-` can cause slowdowns (issue #20623, open)
- Prompt cache invalidation may cause full reprocessing on long conversations
- `tool_choice` parameter is accepted but behavior may differ from Anthropic's
- Signatures in thinking blocks send empty string (Anthropic SDK requires the field to exist)

### `superllama bench`

```bash
exec llama-bench \
  -m "$(get_cached_model_path)" \
  -ngl 99 -fa 1 \
  -p 512,2048 -n 128 -r 3
```

### `superllama stop`

```bash
if [ -f ~/.superllama/server.pid ]; then
  kill "$(cat ~/.superllama/server.pid)" 2>/dev/null
  rm ~/.superllama/server.pid
  echo "Server stopped."
else
  echo "No server running."
fi
```

### `superllama status`

```bash
if curl -s "http://${HOST}:${PORT}/health" | grep -q '"status":"ok"'; then
  echo "Server: running on http://${HOST}:${PORT}"
  curl -s "http://${HOST}:${PORT}/slots" | python3 -m json.tool
else
  echo "Server: not running"
fi
echo "Model: ${MODEL_SIZE} (${HF_REPO}:${QUANT})"
echo "Context: ${CTX_SIZE} tokens"
echo "Config: ~/.superllama/config"
```

---

## Model Map

```bash
declare -A MODELS
MODELS[tiny]="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"     #  0.6 GB
MODELS[small]="unsloth/Qwen3.5-4B-GGUF:Q4_K_M"       #  3.1 GB
MODELS[medium]="unsloth/Qwen3.5-9B-GGUF:Q4_K_M"      #  5.7 GB
MODELS[large]="unsloth/Qwen3.5-27B-GGUF:Q4_K_M"      # 16.7 GB
MODELS[large+]="unsloth/Qwen3.5-27B-GGUF:Q8_0"       # 28.6 GB
MODELS[turbo]="unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"  # 22.0 GB
```

---

## Context Window Strategy

Claude Code recommends 64K+ context. Here's how we handle it:

| Model | RAM | Max Safe Context | Claude Code Viability |
|-------|-----|------------------|-----------------------|
| Qwen3.5-4B Q4_K_M | 8 GB | ~8K | Marginal — basic edits only |
| Qwen3.5-9B Q4_K_M | 16 GB | ~32K | Workable — most tasks fine |
| Qwen3.5-27B Q4_K_M | 24 GB | ~16K | Workable — context will compress |
| Qwen3.5-27B Q4_K_M | 32 GB | ~48K | Good — most Claude Code tasks |
| Qwen3.5-27B Q4_K_M | 36 GB+ | ~64K | Full Claude Code experience |
| Qwen3.5-35B-A3B Q4_K_M | 36 GB | ~32K | Workable — fast + smart |
| Qwen3.5-35B-A3B Q4_K_M | 48 GB+ | ~64K | Best: fast inference + full context |

**`superllama code` auto-computes context**: It calculates the largest context window
that fits in RAM after the model is loaded, capped at 65536.

```bash
# Pseudocode for context calculation
AVAILABLE_RAM=$(get_ram_gb)
MODEL_RAM=$(get_model_size_gb)
OS_HEADROOM=4  # GB reserved for OS
REMAINING=$((AVAILABLE_RAM - MODEL_RAM - OS_HEADROOM))

# ~0.5 GB per 8K tokens of KV cache (approximate, varies by model)
MAX_CTX=$((REMAINING * 8192 / 500))
CTX_SIZE=$(min $MAX_CTX 65536)
```

---

## Why This Beats Ollama

| Aspect | Ollama | SuperLlama |
|--------|--------|------------|
| Layers to GPU | 5 (HTTP→Gin→Scheduler→Subprocess→CGO→llama.cpp) | 1 (llama-server direct) |
| Per-token overhead | Double HTTP + JSON serialization | Single HTTP stream |
| GPU offloading | Conservative VRAM estimator, silent CPU fallback | `-ngl all` — explicit |
| Flash attention | OFF by default | ON by default |
| Context | Model default (32K+), overflows silently | Auto-sized to fit RAM |
| Inference engine | Ollama's Go engine (2.45x slower on Qwen3.5) | Raw llama.cpp |
| Anthropic API | Built into Ollama server (extra layer) | Built into llama-server (native) |
| Binary | 200MB+ Go binary | ~5MB llama-server |
| Code | ~50K lines Go | ~350 lines bash |
| Claude Code setup | `ollama launch claude` (through Ollama's overhead) | `superllama code` (direct to llama-server) |

---

## Performance Expectations

| Model | Ollama (reported) | SuperLlama (expected) | Gain |
|-------|-------------------|-----------------------|------|
| Qwen3.5-27B Q4_K_M | 15-20 tok/s | 35-45 tok/s | 2-3x |
| Qwen3.5-35B-A3B Q4_K_M | 35 tok/s | 90-100 tok/s | 2.5-3x |
| Qwen3.5-9B Q4_K_M | ~60 tok/s | ~80-100 tok/s | 1.5-2x |

---

## Config

`~/.superllama/config`:

```bash
MODEL_SIZE=large          # tiny/small/medium/large/large+/turbo
CTX_SIZE=32768            # auto-computed during install, user can override
PORT=8080                 # server port
HOST=127.0.0.1            # bind address
```

Pass extra llama-server flags:
```bash
superllama -- -np 4 -ctk q8_0 -ctv q8_0
superllama code -- --verbose    # extra flags to claude
```

---

## Build Order + Test Plan

### Phase 1: Core (build first, test first)

```
1. [ ] Write `superllama` script
       - Hardware detection
       - Config read/write
       - `superllama` → exec llama-server
       - `superllama chat` → exec llama-cli
       - `superllama code` → start server + launch claude
       - `superllama stop/status`
       - `superllama bench`
       - `superllama switch <size>`
       - `superllama --help`
       - Passthrough flags via `--`

2. [ ] Write `install.sh`
       - Platform detection
       - llama.cpp install (brew / prebuilt)
       - Claude Code detection
       - Model selection
       - Context auto-sizing
       - Config write
       - Install superllama to PATH
       - Optional model pre-download
```

### Phase 1 Tests

```
TEST 1: Fresh install on macOS Apple Silicon
  $ curl -fsSL .../install.sh | sh
  Expected: llama.cpp installed, model selected, config written

TEST 2: Server mode
  $ superllama
  Expected: llama-server starts, web UI accessible at :8080
  Verify:   curl http://localhost:8080/health → {"status":"ok"}

TEST 3: Chat mode
  $ superllama chat
  Expected: interactive chat in terminal, model responds

TEST 4: Claude Code integration (THE CRITICAL TEST)
  $ superllama code
  Expected:
    - Server starts in background
    - Claude Code launches
    - Can read files, edit code, run commands
    - Tool use works (tool_use/tool_result blocks flow correctly)
    - Streaming works (tokens appear incrementally)
    - Context management works (long sessions don't crash)
  Verify:
    - Ask Claude Code to read a file → it uses tool_use → gets result
    - Ask Claude Code to edit a file → it uses tool_use → file changes
    - Ask Claude Code to run a test → it executes shell command
    - Have a long conversation → context compresses, doesn't OOM

TEST 5: API compatibility
  $ curl http://localhost:8080/v1/messages \
      -H "Content-Type: application/json" \
      -H "x-api-key: test" \
      -d '{
        "model": "qwen3.5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}]
      }'
  Expected: valid Anthropic Messages API response

TEST 6: Streaming
  $ curl http://localhost:8080/v1/messages \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen3.5",
        "max_tokens": 100,
        "stream": true,
        "messages": [{"role": "user", "content": "Hello"}]
      }'
  Expected: SSE stream with message_start, content_block_start, etc.

TEST 7: Tool use round-trip
  $ curl http://localhost:8080/v1/messages \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen3.5",
        "max_tokens": 1024,
        "tools": [{"name": "read_file", "description": "Read a file",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": "Read the file at /tmp/test.txt"}]
      }'
  Expected: response contains tool_use content block with name="read_file"

TEST 8: Performance benchmark
  $ superllama bench
  Expected: shows tok/s, compare against Ollama running same model

TEST 9: Switch model
  $ superllama switch medium
  $ superllama config
  Expected: config updated to Qwen3.5-9B Q4_K_M

TEST 10: Stop/status
  $ superllama status   → shows running/not running
  $ superllama stop     → kills server
  $ superllama status   → shows not running
```

### Phase 2: Polish

```
3. [ ] superllama update (fetch latest llama.cpp release)
4. [ ] superllama logs (tail ~/.superllama/server.log)
5. [ ] Graceful shutdown (trap SIGINT, clean up PID file)
6. [ ] Auto-restart server if model changes
7. [ ] Handle "server already running on different model" case
8. [ ] Colorized output (green for success, yellow for warnings)
```

### Phase 3: Distribution

```
9.  [ ] GitHub repo with README
10. [ ] Host install.sh on GitHub raw (or custom domain)
11. [ ] Homebrew tap (brew install superllama)
12. [ ] Linux testing (Ubuntu, Fedora)
13. [ ] NVIDIA GPU testing (CUDA path)
```

---

## Key Design Decisions

1. **Shell script, not compiled binary.** Transparent, auditable. `cat $(which superllama)` shows everything.

2. **`exec` for server/chat.** Replaces shell process with llama-server. Zero overhead. `ps` shows `llama-server`.

3. **Background server for `code` mode.** `superllama code` starts the server in background, then exec's Claude Code in foreground. Server persists after Claude Code exits (reusable). `superllama stop` kills it.

4. **No daemon.** No launchd/systemd service. Server runs in foreground (`superllama`) or background (`superllama code`). Simple process, simple cleanup.

5. **`-hf` for model management.** llama.cpp handles download, cache, resume. We never touch model files.

6. **`-ngl all` always.** No VRAM guessing. Apple Silicon = unified memory, offload everything. Discrete GPU = offload what fits, llama.cpp handles overflow better than Ollama.

7. **Flash attention ON.** Ollama: off. Us: on. Free performance.

8. **Auto-sized context.** Compute max context from available RAM. Don't blindly use model default (128K+). For `superllama code`, target 64K if RAM allows.

9. **Anthropic API is native.** llama-server's `/v1/messages` does full Anthropic format conversion (including tool_use streaming). We don't need a proxy or translation layer. Claude Code talks directly to llama-server.

10. **Claude Code is Layer 2, not us.** We don't build agentic features, context management, or tool dispatch. Claude Code already does all of that. We just serve the model fast and let Claude Code do its thing.

---

## Realistic Expectations: Local Qwen 3.5 vs Cloud Claude

Be honest with users:

| | Local Qwen 3.5 (SuperLlama) | Cloud Claude (Anthropic API) |
|---|---|---|
| Cost | Free | ~$3-15/hr heavy use |
| Privacy | 100% local | Data to Anthropic |
| Offline | Yes | No |
| Speed (tok/s) | 35-100 depending on model | 80-100 |
| Intelligence | Good but noticeably weaker | State of the art |
| Complex multi-file edits | Sometimes fails | Usually succeeds |
| Tool use reliability | Good with Qwen 3.5, occasional errors | Very reliable |
| Context understanding | Good up to 32-64K | Excellent up to 200K |

SuperLlama with Claude Code is best for:
- **Cost-free daily coding** — quick edits, explanations, test generation
- **Offline work** — flights, no internet
- **Privacy-sensitive code** — proprietary codebases
- **Learning/experimenting** — unlimited usage, no bill anxiety
- **Supplement to cloud Claude** — use local for simple tasks, cloud for hard ones
