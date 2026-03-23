# SuperLlama

Run Qwen 3.5 locally at raw llama.cpp speed. Faster than Ollama.

## Install

```bash
pip install superllama
```

First run auto-installs llama.cpp and picks the best model for your hardware.

## Usage

```bash
superllama serve     # API server + web UI on localhost:8080
superllama chat      # interactive terminal chat
superllama aider     # coding agent (recommended)
superllama code      # Claude Code with local model
```

### Coding with Aider (recommended)

```bash
pip install aider-chat           # or: pipx install aider-chat --python python3.11
cd your-project/
superllama aider
```

Starts llama-server in background, launches [Aider](https://aider.chat) pointed at it. Aider can read/edit files, run commands, auto-commit — all powered by local Qwen 3.5.

### API Server

```bash
superllama serve
```

Serves OpenAI-compatible API (`/v1/chat/completions`) and Anthropic-compatible API (`/v1/messages`) on `http://localhost:8080`. Built-in web UI included. Any tool that speaks OpenAI or Anthropic API works — Open WebUI, Continue.dev, Cline, etc.

### Other commands

```bash
superllama bench              # benchmark tok/s
superllama config             # show config
superllama config set K V     # change a setting
superllama switch <size>      # switch model (tiny/small/medium/large/large+/turbo)
superllama stop               # stop background server
superllama status             # check server status
superllama setup              # re-run setup
```

## Models

Auto-selected based on your RAM:

| Size | Model | Quant | Disk | Min RAM |
|------|-------|-------|------|---------|
| tiny | Qwen3.5-0.8B | Q4_K_M | 0.6 GB | 4 GB |
| small | Qwen3.5-4B | Q4_K_M | 3.1 GB | 6 GB |
| medium | Qwen3.5-9B | Q4_K_M | 5.7 GB | 10 GB |
| large | Qwen3.5-27B | Q4_K_M | 16.7 GB | 22 GB |
| large+ | Qwen3.5-27B | Q8_0 | 28.6 GB | 34 GB |
| turbo | Qwen3.5-35B-A3B | Q4_K_M | 22.0 GB | 28 GB |

## How it works

SuperLlama is a thin Python wrapper (~400 lines, stdlib only) around [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. It:

1. Auto-installs llama-server from GitHub releases
2. Detects your hardware (OS, RAM, GPU)
3. Picks optimal flags (`-ngl all`, `-fa on`, `--mlock`, etc.)
4. Starts llama-server — that's it

Python is only the launcher (~100ms). The inference path is 100% C++: `llama-server → Metal/CUDA GPU`.

## Why not Ollama?

Ollama adds 5 layers between you and the GPU (HTTP router → scheduler → subprocess → CGO → llama.cpp). SuperLlama has 1 layer (llama-server direct). On Qwen 3.5, that's 2-3x faster inference.

See [RESEARCH.md](RESEARCH.md) for the full analysis.

## Cross-platform

Works on macOS (Metal), Linux (CUDA/CPU), and Windows (CUDA/CPU).

## License

MIT
