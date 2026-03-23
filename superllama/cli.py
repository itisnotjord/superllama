#!/usr/bin/env python3
"""SuperLlama CLI -- run Qwen 3.5 locally via llama.cpp. Faster than Ollama."""

import argparse
import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Model map
# ---------------------------------------------------------------------------

MODELS = {
    "tiny":   {"repo": "unsloth/Qwen3.5-0.8B-GGUF",    "quant": "Q4_K_M", "size_gb": 0.6,  "ram_min": 4},
    "small":  {"repo": "unsloth/Qwen3.5-4B-GGUF",       "quant": "Q4_K_M", "size_gb": 3.1,  "ram_min": 6},
    "medium": {"repo": "unsloth/Qwen3.5-9B-GGUF",       "quant": "Q4_K_M", "size_gb": 5.7,  "ram_min": 10},
    "large":  {"repo": "unsloth/Qwen3.5-27B-GGUF",      "quant": "Q4_K_M", "size_gb": 16.7, "ram_min": 22},
    "large+": {"repo": "unsloth/Qwen3.5-27B-GGUF",      "quant": "Q8_0",   "size_gb": 28.6, "ram_min": 34},
    "turbo":  {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF",  "quant": "Q4_K_M", "size_gb": 22.0, "ram_min": 28},
}

CONFIG_DIR = pathlib.Path.home() / ".superllama"
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"
BIN_DIR = CONFIG_DIR / "bin"

DEFAULT_CONFIG = {
    "model_size": "large",
    "ctx_size": 32768,
    "port": 8080,
    "host": "127.0.0.1",
}

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_system_info():
    """Return dict with os, arch, ram_gb, gpu_type."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "ram_gb": 0,
        "gpu_type": "cpu",
    }

    # --- RAM ---
    sysname = info["os"]
    if sysname == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            info["ram_gb"] = int(out.strip()) / (1024 ** 3)
        except Exception:
            info["ram_gb"] = 8
    elif sysname == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = kb / (1024 ** 2)
                        break
        except Exception:
            info["ram_gb"] = 8
    elif sysname == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                text=True,
            )
            for line in out.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    info["ram_gb"] = int(line) / (1024 ** 3)
                    break
        except Exception:
            info["ram_gb"] = 8
    else:
        info["ram_gb"] = 8

    info["ram_gb"] = round(info["ram_gb"], 1)

    # --- GPU ---
    if sysname == "Darwin" and info["arch"] == "arm64":
        info["gpu_type"] = "metal"
    else:
        if shutil.which("nvidia-smi"):
            try:
                subprocess.check_call(
                    ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                info["gpu_type"] = "cuda"
            except Exception:
                info["gpu_type"] = "cpu"
        else:
            info["gpu_type"] = "cpu"

    return info

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    """Load config from disk. Returns None if not found."""
    if not CONFIG_FILE.exists():
        return None
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def recommend_model(ram_gb):
    """Return the largest model size whose ram_min <= ram_gb - 4."""
    available = ram_gb - 4
    best = "tiny"
    for name, m in MODELS.items():
        if m["ram_min"] <= available:
            best = name
    return best


def compute_max_context(config):
    """Estimate max context tokens for the chosen model given system RAM."""
    info = get_system_info()
    return compute_max_context_for_model(config["model_size"], info["ram_gb"])


def compute_max_context_for_model(size, ram_gb):
    model = MODELS[size]
    # Reserve: model weight + 5GB OS/compute headroom
    headroom = ram_gb - model["size_gb"] - 5
    if headroom <= 0:
        return 4096
    # KV cache: ~1GB per 8K tokens (conservative estimate)
    ctx = int(headroom * 8192)
    # Cap based on RAM tier — bigger context = slower prompt processing
    # 16GB Mac: 8-16K is the sweet spot for responsiveness
    # 32GB+: can afford 32-64K
    if ram_gb <= 18:
        cap = 16384
    elif ram_gb <= 36:
        cap = 32768
    else:
        cap = 65536
    return max(4096, min(ctx, cap))

# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------

def find_binary(name):
    """Find a llama.cpp binary by name, checking PATH then ~/.superllama/bin/."""
    found = shutil.which(name)
    if found:
        return found
    candidate = BIN_DIR / name
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    # macOS brew typical location
    brew_path = pathlib.Path("/opt/homebrew/bin") / name
    if brew_path.exists():
        return str(brew_path)
    return None


def find_llama_server():
    return find_binary("llama-server")


def _get_latest_llama_release_tag():
    url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["tag_name"]


def _release_asset_name(tag, info):
    ver = tag.lstrip("b").lstrip("v")
    sysname = info["os"]
    arch = info["arch"]
    if sysname == "Darwin":
        suffix = "arm64" if arch == "arm64" else "x64"
        return f"llama-b{ver}-bin-macos-{suffix}.tar.gz"
    elif sysname == "Linux":
        return f"llama-b{ver}-bin-ubuntu-x64.tar.gz"
    elif sysname == "Windows":
        return f"llama-b{ver}-bin-win-x64.zip"
    else:
        raise RuntimeError(f"Unsupported OS: {sysname}")


def install_llama_server(info=None):
    """Download pre-built llama.cpp binaries from GitHub releases."""
    if info is None:
        info = get_system_info()

    print("Fetching latest llama.cpp release...")
    tag = _get_latest_llama_release_tag()
    asset = _release_asset_name(tag, info)
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    dl_path = CONFIG_DIR / asset
    print(f"Downloading {asset}...")
    urllib.request.urlretrieve(url, str(dl_path))

    print("Extracting...")
    if asset.endswith(".tar.gz"):
        with tarfile.open(dl_path, "r:gz") as tar:
            tar.extractall(path=str(CONFIG_DIR / "_extract"))
    elif asset.endswith(".zip"):
        with zipfile.ZipFile(dl_path, "r") as zf:
            zf.extractall(path=str(CONFIG_DIR / "_extract"))

    # Find and move binaries + shared libraries into BIN_DIR
    extract_root = CONFIG_DIR / "_extract"
    wanted_bins = {"llama-server", "llama-cli", "llama-bench"}
    if info["os"] == "Windows":
        wanted_bins = {n + ".exe" for n in wanted_bins}
        lib_exts = (".dll",)
    elif info["os"] == "Darwin":
        lib_exts = (".dylib",)
    else:
        lib_exts = (".so",)

    for root, _dirs, files in os.walk(extract_root):
        for fn in files:
            is_wanted_bin = fn in wanted_bins
            is_lib = any(fn.endswith(ext) or ext + "." in fn for ext in lib_exts)
            if is_wanted_bin or is_lib:
                src = pathlib.Path(root) / fn
                dst = BIN_DIR / fn
                shutil.copy2(src, dst)
                if info["os"] != "Windows":
                    os.chmod(dst, 0o755)

    # Cleanup
    shutil.rmtree(extract_root, ignore_errors=True)
    dl_path.unlink(missing_ok=True)

    path = find_llama_server()
    if path:
        print(f"Installed llama-server -> {path}")
    else:
        print("Warning: installation finished but llama-server not found on PATH.")
    return path

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _read_pid():
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def server_health(config):
    """Return True if the server responds with status ok."""
    url = f"http://{config['host']}:{config['port']}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


def wait_for_server(config, timeout=300):
    deadline = time.time() + timeout
    dots = 0
    last_log_size = 0
    while time.time() < deadline:
        if server_health(config):
            print()
            return True
        # Show progress by tailing the log for key events
        if LOG_FILE.exists():
            try:
                current_size = LOG_FILE.stat().st_size
                if current_size > last_log_size:
                    with open(LOG_FILE) as f:
                        f.seek(last_log_size)
                        new_lines = f.read()
                    last_log_size = current_size
                    for line in new_lines.splitlines():
                        if "downloading" in line.lower() or "download" in line.lower():
                            print(f"\r  Downloading model...{' ' * 20}", end="", flush=True)
                        elif "loading model" in line.lower() or "load_model" in line.lower():
                            print(f"\r  Loading model into GPU...{' ' * 20}", end="", flush=True)
                        elif "warming up" in line.lower():
                            print(f"\r  Warming up...{' ' * 20}", end="", flush=True)
                        elif "listening on" in line.lower():
                            print(f"\r  Server ready!{' ' * 20}", end="", flush=True)
            except OSError:
                pass
        dots = (dots + 1) % 4
        print(f"\r  Waiting{'.' * (dots + 1)}{' ' * (4 - dots)}", end="", flush=True)
        time.sleep(1)
    print()
    print("Error: server did not become healthy within timeout.")
    print(f"Check logs: cat {LOG_FILE}")
    sys.exit(1)


def start_server(config, ctx_override=None, extra_flags=None):
    """Start llama-server as a background process."""
    llama = find_llama_server()
    if not llama:
        print("llama-server not found. Run: superllama setup")
        sys.exit(1)

    model = MODELS[config["model_size"]]
    ctx = ctx_override or config["ctx_size"]

    cmd = [
        llama,
        "-hf", f"{model['repo']}:{model['quant']}",
        "-ngl", "all",
        "-fa", "on",
        "-c", str(ctx),
        "-np", "1",
        "--mlock",
        "--cache-prompt",
        "-b", "2048",
        "-ub", "512",
        "--host", config["host"],
        "--port", str(config["port"]),
    ] + (extra_flags or [])

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    PID_FILE.write_text(str(proc.pid))
    print(f"llama-server started (pid {proc.pid}), logging to {LOG_FILE}")
    return proc


def stop_server():
    pid = _read_pid()
    if pid is None:
        print("No server PID file found.")
        return
    if _pid_alive(pid):
        print(f"Stopping llama-server (pid {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            print(f"Could not kill pid {pid}: {e}")
    else:
        print(f"Server (pid {pid}) is not running.")
    PID_FILE.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# llama.cpp cache directory helper
# ---------------------------------------------------------------------------

def get_llama_cache_dir():
    """Return the llama.cpp HuggingFace model cache directory."""
    # llama.cpp caches HF downloads in ~/.cache/llama.cpp
    return pathlib.Path.home() / ".cache" / "llama.cpp"


def find_cached_model(config):
    """Attempt to find the locally cached GGUF file for the active model."""
    model = MODELS[config["model_size"]]
    cache = get_llama_cache_dir()
    # The HF cache structure uses the repo id with slashes replaced
    repo_part = model["repo"].replace("/", "_")
    quant = model["quant"]

    # Walk the cache dir looking for a matching gguf
    if cache.exists():
        for root, _dirs, files in os.walk(cache):
            for fn in files:
                if fn.endswith(".gguf") and quant in fn:
                    full = pathlib.Path(root) / fn
                    if repo_part.split("_")[-1].lower() in fn.lower():
                        return str(full)
                    # Fall back: any gguf with the right quant in a path
                    # that contains the repo name fragment
                    if model["repo"].split("/")[-1].lower().replace("-", "") in root.lower().replace("-", ""):
                        return str(full)
    return None

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_serve(config, extra):
    llama = find_llama_server()
    if not llama:
        print("llama-server not found. Run: superllama setup")
        sys.exit(1)

    model = MODELS[config["model_size"]]
    cmd = [
        llama,
        "-hf", f"{model['repo']}:{model['quant']}",
        "-ngl", "all",
        "-fa", "on",
        "-c", str(config["ctx_size"]),
        "-np", "1",
        "--mlock",
        "--cache-prompt",
        "-b", "2048",
        "-ub", "512",
        "--host", config["host"],
        "--port", str(config["port"]),
    ] + extra

    print(f"Starting llama-server ({config['model_size']}) on "
          f"http://{config['host']}:{config['port']}")
    print(f"Model : {model['repo']}:{model['quant']}")
    print(f"Context: {config['ctx_size']} tokens")
    print()

    if platform.system() == "Windows":
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    else:
        os.execvp(cmd[0], cmd)


def cmd_chat(config, extra):
    cli = find_binary("llama-cli")
    if not cli:
        print("llama-cli not found. Run: superllama setup")
        sys.exit(1)

    model = MODELS[config["model_size"]]
    cmd = [
        cli,
        "-hf", f"{model['repo']}:{model['quant']}",
        "-ngl", "all",
        "-fa", "on",
        "-c", str(config["ctx_size"]),
        "--mlock",
        "-cnv",
    ] + extra

    print(f"Starting chat ({config['model_size']})...")
    print()

    if platform.system() == "Windows":
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    else:
        os.execvp(cmd[0], cmd)


def cmd_code(config, extra):
    claude_path = shutil.which("claude")
    if not claude_path:
        print("Claude Code not found.")
        print("Install it with:  npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    # Start server in background if not already running
    model = MODELS[config["model_size"]]
    code_ctx = compute_max_context(config)

    if not server_health(config):
        print(f"Model:   {model['repo']}:{model['quant']}")
        print(f"Context: {code_ctx} tokens")
        print(f"Server:  http://{config['host']}:{config['port']}")
        print()
        print("Starting llama-server (first run downloads the model, may take a few minutes)...")
        # Use KV cache quantization to fit more context in limited RAM
        start_server(config, ctx_override=code_ctx, extra_flags=["-ctk", "q8_0", "-ctv", "q8_0"])
        wait_for_server(config, timeout=600)  # 10 min for large model downloads
    else:
        print(f"Server already running on http://{config['host']}:{config['port']}")

    model_name = "qwen3.5"
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": f"http://{config['host']}:{config['port']}",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "superllama",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_name,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_name,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_name,
        "CLAUDE_CODE_SUBAGENT_MODEL": model_name,
        # Tell Claude Code our ACTUAL context size so it compacts at the right time
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(code_ctx),
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "DISABLE_PROMPT_CACHING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # Minimal system prompt + fewer tools — saves ~10K tokens of context
        "CLAUDE_CODE_SIMPLE": "1",
    })

    cmd = [claude_path, "--model", model_name] + extra
    print(f"Launching Claude Code with local {model_name} (simple mode)...")

    if platform.system() == "Windows":
        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    else:
        os.execvpe(claude_path, cmd, env)


def _write_aider_model_settings(config):
    """Write optimized Aider model settings for local Qwen 3.5."""
    model = MODELS[config["model_size"]]
    # Use 'whole' edit format for smaller models, 'diff' for 27B+
    edit_fmt = "diff" if model["size_gb"] >= 16 else "whole"
    settings_path = CONFIG_DIR / "aider.model.settings.yml"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        f"- name: openai/qwen3.5\n"
        f"  edit_format: {edit_fmt}\n"
        f"  use_repo_map: false\n"
        f"  streaming: true\n"
        f"  use_temperature: true\n"
        f"  extra_params:\n"
        f"    max_tokens: 4096\n"
    )
    return str(settings_path), edit_fmt


def cmd_aider(config, extra):
    aider_path = shutil.which("aider")
    if not aider_path:
        print("Aider not found.")
        print("Install it with:  pip install aider-chat")
        sys.exit(1)

    # Start server in background if not running
    if not server_health(config):
        model = MODELS[config["model_size"]]
        ctx = config["ctx_size"]
        print(f"Model:   {model['repo']}:{model['quant']}")
        print(f"Context: {ctx} tokens")
        print(f"Server:  http://{config['host']}:{config['port']}")
        print()
        print("Starting llama-server (first run downloads the model, may take a few minutes)...")
        start_server(config)
        wait_for_server(config, timeout=600)
    else:
        print(f"Server already running on http://{config['host']}:{config['port']}")

    # Write optimized model settings
    settings_path, edit_fmt = _write_aider_model_settings(config)

    env = os.environ.copy()
    env["OPENAI_API_BASE"] = f"http://{config['host']}:{config['port']}/v1"
    env["OPENAI_API_KEY"] = "superllama"

    cmd = [
        aider_path,
        "--model", "openai/qwen3.5",
        "--model-settings-file", settings_path,
        "--no-show-model-warnings",
        "--no-auto-lint",
        "--no-auto-test",
    ] + extra

    print(f"Launching Aider (edit format: {edit_fmt})...")
    print()

    if platform.system() == "Windows":
        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    else:
        os.execvpe(aider_path, cmd, env)


def cmd_bench(config):
    bench = find_binary("llama-bench")
    if not bench:
        print("llama-bench not found. Run: superllama setup")
        sys.exit(1)

    model_path = find_cached_model(config)
    if not model_path:
        print("Cached model file not found.")
        print("Run 'superllama serve' once first so the model gets downloaded,")
        print("then try 'superllama bench' again.")
        sys.exit(1)

    cmd = [
        bench,
        "-m", model_path,
        "-ngl", "99",
        "-fa", "1",
        "-p", "512,2048",
        "-n", "128",
        "-r", "3",
    ]

    print(f"Benchmarking: {model_path}")
    print()

    if platform.system() == "Windows":
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    else:
        os.execvp(cmd[0], cmd)


def cmd_setup():
    info = get_system_info()
    print(f"Detected: {info['os']} {info['arch']}, {info['ram_gb']} GB RAM, GPU: {info['gpu_type']}")

    # Show models that fit
    print("\nAvailable models:\n")
    header = f"  {'SIZE':<10} {'MODEL':<35} {'DISK':<10} {'MIN RAM':<10}"
    print(header)
    print(f"  {'─' * 10} {'─' * 35} {'─' * 10} {'─' * 10}")
    for name, m in MODELS.items():
        fits = m["ram_min"] <= info["ram_gb"]
        marker = "" if fits else " (insufficient RAM)"
        label = m["repo"].split("/")[1]
        print(f"  {name:<10} {label:<35} {m['size_gb']:<10.1f} {m['ram_min']:<10}{marker}")

    default = recommend_model(info["ram_gb"])
    try:
        choice = input(f"\nModel size [{default}]: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print()
        choice = default

    if choice not in MODELS:
        print(f"Unknown size '{choice}'. Using '{default}'.")
        choice = default

    ctx = compute_max_context_for_model(choice, info["ram_gb"])
    ctx = min(ctx, 65536)
    ctx = max(ctx, 4096)

    # Check llama-server
    if not find_llama_server():
        print("\nllama-server not found. Installing from GitHub releases...")
        install_llama_server(info)
    else:
        print(f"\nllama-server found: {find_llama_server()}")

    cfg = {
        "model_size": choice,
        "ctx_size": ctx,
        "port": 8080,
        "host": "127.0.0.1",
    }
    save_config(cfg)
    print(f"\nConfig saved to {CONFIG_FILE}")
    print(f"  model_size : {choice}")
    print(f"  ctx_size   : {ctx}")
    print(f"  port       : 8080")
    print()
    print("Setup complete! Try one of:")
    print("  superllama serve   -- start API server + web UI")
    print("  superllama chat    -- interactive terminal chat")
    print("  superllama code    -- launch Claude Code with local Qwen 3.5")


def cmd_stop(config):
    stop_server()


def cmd_status(config):
    pid = _read_pid()
    if pid and _pid_alive(pid):
        healthy = server_health(config) if config else False
        status = "healthy" if healthy else "running (not healthy yet)"
        print(f"llama-server is {status} (pid {pid})")
        print(f"  endpoint: http://{config['host']}:{config['port']}")
    else:
        print("llama-server is not running.")
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)


def cmd_config(config, args):
    if args.action == "set":
        if not args.key or args.value is None:
            print("Usage: superllama config set <key> <value>")
            sys.exit(1)
        key = args.key
        value = args.value
        if config is None:
            config = dict(DEFAULT_CONFIG)

        # Type coercion for known keys
        if key in ("ctx_size", "port"):
            try:
                value = int(value)
            except ValueError:
                print(f"Error: {key} must be an integer.")
                sys.exit(1)
        elif key == "model_size":
            if value not in MODELS:
                print(f"Error: unknown model size '{value}'. Choose from: {', '.join(MODELS)}")
                sys.exit(1)

        config[key] = value
        save_config(config)
        print(f"Set {key} = {value}")
    else:
        # Show config
        if config is None:
            print("No config found. Run: superllama setup")
            sys.exit(1)
        print(json.dumps(config, indent=2))


def cmd_switch(config, size):
    if config is None:
        config = dict(DEFAULT_CONFIG)
    old = config.get("model_size", "?")
    config["model_size"] = size

    info = get_system_info()
    ctx = compute_max_context_for_model(size, info["ram_gb"])
    config["ctx_size"] = min(ctx, 65536)

    save_config(config)
    print(f"Switched model: {old} -> {size}")
    print(f"Context size adjusted to {config['ctx_size']} tokens")

    # If server is running, remind to restart
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print("Note: restart the server for changes to take effect (superllama stop && superllama serve)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="superllama",
        description="Run Qwen 3.5 locally via llama.cpp. Faster than Ollama.",
    )
    parser.add_argument("--version", action="version", version=f"superllama {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Start API server + web UI")
    subparsers.add_parser("chat", help="Interactive terminal chat")
    subparsers.add_parser("code", help="Launch Claude Code with local Qwen 3.5")
    subparsers.add_parser("aider", help="Launch Aider coding agent (recommended for local models)")
    subparsers.add_parser("bench", help="Benchmark your setup (tok/s)")
    subparsers.add_parser("setup", help="Interactive first-time setup")
    subparsers.add_parser("stop", help="Stop the running server")
    subparsers.add_parser("status", help="Show server status")

    config_parser = subparsers.add_parser("config", help="Show or set config values")
    config_parser.add_argument("action", nargs="?", choices=["set"], default=None)
    config_parser.add_argument("key", nargs="?")
    config_parser.add_argument("value", nargs="?")

    switch_parser = subparsers.add_parser("switch", help="Switch model size")
    switch_parser.add_argument("size", choices=list(MODELS.keys()))

    args, extra = parser.parse_known_args()

    # Auto-setup on first run
    config = load_config()
    if config is None and args.command not in ("setup", None):
        print("First run detected -- starting interactive setup.\n")
        cmd_setup()
        config = load_config()
        if config is None:
            print("Setup did not complete. Exiting.")
            sys.exit(1)
        print()

    if args.command == "serve":
        cmd_serve(config, extra)
    elif args.command == "chat":
        cmd_chat(config, extra)
    elif args.command == "code":
        cmd_code(config, extra)
    elif args.command == "aider":
        cmd_aider(config, extra)
    elif args.command == "bench":
        cmd_bench(config)
    elif args.command == "setup":
        cmd_setup()
    elif args.command == "stop":
        cmd_stop(config)
    elif args.command == "status":
        cmd_status(config)
    elif args.command == "config":
        cmd_config(config, args)
    elif args.command == "switch":
        cmd_switch(config, args.size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
