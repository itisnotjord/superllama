# SuperLlama + Aider — Plan

## Why Aider over Claude Code for local models

| | Claude Code | Aider |
|---|---|---|
| System prompt size | ~15-20K tokens (even with SIMPLE=1) | ~2-4K tokens |
| API format | Anthropic Messages API (translated) | OpenAI API (native in llama-server) |
| Local model support | Bolted on, fragile | First-class, well-tested |
| Edit format | Complex tool_use blocks | Simple: `whole` file or `diff` |
| Context overhead | Tool definitions eat ~10K tokens | Minimal, `use_repo_map: false` saves more |
| "hi" response time | ~2 min on 16GB Mac | ~5-10 sec on 16GB Mac |
| Install | `npm install -g @anthropic-ai/claude-code` | `pip install aider-chat` (same ecosystem) |

**The core problem**: Claude Code was designed for 200K context Claude models. Its system prompt alone consumes most of a local model's context. Aider was designed to work with many models, including local/small ones.

---

## Architecture

```
superllama code
  │
  ├── Starts llama-server in background (if not running)
  │     └── OpenAI API at http://localhost:8080/v1/chat/completions
  │
  └── Launches Aider pointed at llama-server
        └── OPENAI_API_BASE=http://localhost:8080/v1
        └── --model openai/qwen3.5
        └── --edit-format whole
        └── --no-show-model-warnings
```

No Anthropic API translation. No bloated system prompt. Direct OpenAI API → llama-server → GPU.

---

## How Aider works with llama-server

```bash
# llama-server already serves OpenAI-compatible API:
#   POST /v1/chat/completions
#   POST /v1/completions
#   POST /v1/models

# Aider connects via:
OPENAI_API_BASE=http://localhost:8080/v1 \
OPENAI_API_KEY=superllama \
  aider --model openai/qwen3.5
```

That's it. No Anthropic env vars, no model routing, no prompt caching workarounds.

---

## Optimal Aider settings for local Qwen 3.5

Create `~/.superllama/aider.model.settings.yml`:

```yaml
- name: openai/qwen3.5
  edit_format: whole          # simpler than diff, works better with local models
  use_repo_map: false         # saves context — repo map can eat 5-10K tokens
  use_system_prompt: true
  streaming: true
  use_temperature: true
  extra_params:
    max_tokens: 4096          # cap output length
    num_ctx: 16384            # match our server context
```

### Edit format choice

| Format | Tokens used | Reliability | Best for |
|--------|-------------|-------------|----------|
| `whole` | More (sends full file) | Very reliable | Small/medium models, small files |
| `diff` | Less (only changes) | Needs strong model | Large models (27B+), large files |
| `udiff` | Less | Moderate | Mid-range models |

**Default `whole` for models under 27B. Switch to `diff` for 27B+.**

---

## Updated `superllama code` command

```python
def cmd_code(config, extra):
    # 1. Check aider is installed
    aider_path = shutil.which("aider")
    if not aider_path:
        print("Aider not found. Install: pip install aider-chat")
        sys.exit(1)

    # 2. Start server if not running
    if not server_health(config):
        start_server(config)
        wait_for_server(config)

    # 3. Write optimized model settings
    model_settings = CONFIG_DIR / "aider.model.settings.yml"
    model = MODELS[config["model_size"]]
    edit_fmt = "diff" if model["size_gb"] >= 16 else "whole"
    write_aider_settings(model_settings, config, edit_fmt)

    # 4. Launch aider
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = f"http://{config['host']}:{config['port']}/v1"
    env["OPENAI_API_KEY"] = "superllama"

    cmd = [
        aider_path,
        "--model", "openai/qwen3.5",
        "--model-settings-file", str(model_settings),
        "--no-show-model-warnings",
        "--no-auto-lint",
    ] + extra

    os.execvpe(aider_path, cmd, env)
```

---

## Context budget comparison

On 16GB Mac with Qwen3.5-9B Q4_K_M (ctx=16384):

```
Claude Code:
  System prompt + tools:  ~15,000 tokens
  Available for chat:      ~1,384 tokens  ← basically nothing, instant compact

Aider (whole mode, no repo map):
  System prompt:           ~2,000 tokens
  Available for chat:     ~14,384 tokens  ← 7x more room
```

---

## What Aider gives you

- **File editing**: reads and writes files in your project
- **Git integration**: auto-commits changes, shows diffs
- **Multi-file edits**: can edit multiple files in one go
- **Undo**: `aider /undo` reverts last change
- **Add files**: `/add file.py` to add files to context
- **Voice**: `/voice` for voice input
- **Web**: `/web url` to fetch and discuss web pages
- **Lint/test**: auto-run linter and tests after edits (disable with --no-auto-lint for speed)
- **Cost tracking**: shows token usage per message

---

## Commands update

```
superllama serve              # start llama-server (API + web UI)
superllama chat               # quick terminal chat (llama-cli)
superllama code [-- args]     # start server + launch Aider
superllama bench              # benchmark tok/s
superllama config             # show config
superllama switch <size>      # switch model
superllama stop               # stop server
superllama status             # server status
```

---

## Install flow update

```bash
pip install superllama aider-chat
superllama setup
superllama code
```

Or the install script checks for aider:
```python
if not shutil.which("aider"):
    print("Installing aider...")
    subprocess.run([sys.executable, "-m", "pip", "install", "aider-chat"])
```

---

## Performance expectations on 16GB Mac

| Model | Context | Edit format | First response | Edits |
|-------|---------|-------------|----------------|-------|
| tiny (0.8B) | 16K | whole | ~2 sec | Fast but low quality |
| small (4B) | 16K | whole | ~5 sec | Decent for small edits |
| medium (9B) | 16K | whole | ~8-15 sec | Good for most tasks |
| large (27B) | 16K | diff | ~20-30 sec | Best quality, needs 24GB+ |

vs Claude Code on same hardware: ~2 min per response due to context bloat.

---

## Migration: what changes in cli.py

1. `cmd_code()`: Replace Claude Code launcher with Aider launcher
2. Remove all `ANTHROPIC_*` env vars from code mode
3. Add `write_aider_settings()` helper to generate YAML config
4. Add Aider detection + auto-install
5. Remove `CLAUDE_CODE_*` env vars
6. Use OpenAI API format instead of Anthropic

Net change: ~30 lines replaced, code gets simpler.
