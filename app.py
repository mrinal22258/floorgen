"""
FloorGen Interactive Web Demonstration Platform.
Unified master entry point interfacing directly with the FloorGen generative pipeline,
retrieval-augmented exemplar grounding, CP-SAT constraint satisfaction, and BIM exporter.

Automatically prepares the runtime environment, starts local background services (Ollama Qwen),
verifies checkpoints, and launches the web studio in a single command:
    python app.py
"""

import sys
import os
import time
import shutil
import subprocess
import urllib.request
from pathlib import Path

# Safe terminal encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from floorgen.rag.llm_brief_parser import ensure_ollama_service


def setup_environment():
    """Validates directories, environment file, and background services."""
    print("=" * 70)
    print("  [FloorGen] Initializing All-in-One Autonomous Architecture Engine...")
    print("=" * 70)

    # 1. Verify workspace directories
    dirs_to_create = [
        "checkpoints/sota",
        "data/processed",
        "data/raw",
        "outputs/run_demo",
        "static/images",
        "static/css",
        "static/js"
    ]
    for d in dirs_to_create:
        p = REPO_ROOT / d
        p.mkdir(parents=True, exist_ok=True)
    print("  [1/4] Workspace directory structure verified.")

    # 2. Ensure .env exists
    env_file = REPO_ROOT / ".env"
    env_example = REPO_ROOT / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("  [2/4] Initialized .env configuration.")
    else:
        print("  [2/4] Configuration environment verified.")

    # 3. Ensure local Ollama background service is running
    print("  [3/4] Ensuring Ollama AI service is active...")
    ollama_ok = ensure_ollama_service(timeout=20.0)
    if ollama_ok:
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                import json
                models_data = json.loads(resp.read().decode())
                model_names = [m.get("name") for m in models_data.get("models", [])]
                active_str = ", ".join(model_names) if model_names else "none"
                print(f"        ✓ Ollama service online. Active reasoning models: {active_str}")
        except Exception:
            print("        ✓ Ollama service online.")
    else:
        print("        ! Ollama not reachable; automatic fallback to rule-based parser active.")


    # 4. Verify model checkpoints
    ckpt_path = REPO_ROOT / "checkpoints/rag_diffusion.pt"
    if not ckpt_path.exists():
        print("  [4/4] Neural checkpoints missing. Initializing model weights...")
        try:
            from floorgen.models.train_sota import train_sota_models
            train_sota_models(epochs=1, batch_size=8, device="cpu")
            print("        ✓ Neural checkpoints initialized.")
        except Exception as e:
            print(f"        ! Checkpoint init notice: {e}")
    else:
        print("  [4/4] Neural diffusion and graph checkpoints verified.")

    print("-" * 70)


if __name__ == "__main__":
    # 1. Run all auto-setup tasks first
    setup_environment()

    # 2. Import demo and theme
    from floorgen.demo.app import create_demo
    from floorgen.demo.theme import get_rose_quartz_theme, ROSE_QUARTZ_HEAD_SCRIPT, ROSE_QUARTZ_CSS

    print("  [FloorGen] Building Studio Interface...")
    demo = create_demo()

    print("\n" + "=" * 70)
    print("  ★ FloorGen Architectural Studio is Ready!")
    print("  Local URL:   http://localhost:7860  or  http://127.0.0.1:7860")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        theme=get_rose_quartz_theme(),
        css=ROSE_QUARTZ_CSS,
        head=ROSE_QUARTZ_HEAD_SCRIPT
    )
