"""
Architectural Brief Parser with Local Ollama Qwen Integration.
Leverages local Ollama Qwen2.5:3b (zero cloud cost) with automated fallback
to the heuristic rule-based brief parser.
"""

import os
import sys
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from floorgen.logging_config import get_logger

log = get_logger("llm_brief_parser")

_OLLAMA_PROC = None

def ensure_ollama_service(timeout: float = 20.0) -> bool:
    """
    Ensures that local Ollama background service is active.
    If not running, attempts to automatically launch 'ollama serve' in the background.
    """
    global _OLLAMA_PROC

    # 1. Check if already responding
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    # 2. Locate binary
    ollama_bin = shutil.which("ollama")
    ollama_app = None
    if sys.platform == "win32":
        local_app = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama app.exe"
        if local_app.exists():
            ollama_app = str(local_app)

    if not ollama_bin:
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
            Path(os.path.expanduser("~")) / "AppData/Local/Programs/Ollama/ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ]
        for c in candidates:
            if c.exists():
                ollama_bin = str(c)
                break

    if not ollama_bin and not ollama_app:
        log.warning("Ollama executable not found on system. Running with heuristic fallback.")
        return False

    launch_cmd = [ollama_app] if (ollama_app and not ollama_bin) else [str(ollama_bin), "serve"]
    log.info(f"Auto-starting local Ollama service via {launch_cmd[0]}...")
    try:
        creationflags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        env = os.environ.copy()
        _OLLAMA_PROC = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags
        )
        t_start = time.time()
        while time.time() - t_start < timeout:
            time.sleep(0.8)
            try:
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        log.info("Local Ollama service successfully initialized and responding.")
                        return True
            except Exception:
                continue
    except Exception as exc:
        log.warning(f"Could not automatically launch Ollama service: {exc}")

    return False


SYSTEM_PROMPT = """You are an expert architectural brief parser. Convert the user's natural language request into a strictly structured JSON floorplan specification.

Output ONLY valid, parseable JSON with NO markdown formatting, NO backticks, and NO conversational text:
{
  "rooms": ["living_room", "master_bedroom", "second_bedroom", "bathroom", "kitchen", "balcony"],
  "affinities": [["kitchen", "living_room"], ["master_bedroom", "bathroom"]],
  "style": "open_plan",
  "area_sqft": 1100,
  "constraints": ["morning_light_kitchen", "split_bedrooms"]
}

Allowed room categories:
- living_room
- master_bedroom
- second_bedroom
- bathroom
- kitchen
- balcony
- entrance
- dining_room
- study
- storage
"""

VALID_ROOM_CATEGORIES = {
    "living_room", "master_bedroom", "second_bedroom", "bathroom",
    "kitchen", "balcony", "entrance", "dining_room", "study", "storage"
}


def parse_brief_with_ollama(
    brief: str,
    model: str = "qwen2.5:3b",
    timeout_sec: float = 8.0
) -> Dict[str, Any]:
    """
    Parses an architectural brief using local Ollama Qwen2.5:3b.
    Falls back gracefully to keyword heuristic parsing if Ollama is unreachable.
    """
    if not brief or not brief.strip():
        return {
            "rooms": ["living_room", "master_bedroom", "bathroom", "kitchen"],
            "affinities": [],
            "style": "standard",
            "area_sqft": None,
            "constraints": []
        }

    try:
        ensure_ollama_service()
        import ollama
        log.info(f"Parsing brief via Ollama model '{model}'...")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse: {brief.strip()}"}
            ],
            format="json",
            options={"temperature": 0.1, "num_predict": 384}
        )
        content = response["message"]["content"].strip()
        # Clean any accidental wrapping markdown
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)

        # Validate and sanitize rooms
        raw_rooms = data.get("rooms", [])
        sanitized_rooms = [r for r in raw_rooms if r in VALID_ROOM_CATEGORIES]
        if not sanitized_rooms:
            sanitized_rooms = ["living_room", "master_bedroom", "bathroom", "kitchen"]

        data["rooms"] = sanitized_rooms
        data["source"] = f"ollama_{model}"
        log.info(f"Ollama brief parsed successfully: {sanitized_rooms}")
        return data

    except Exception as e:
        log.warning(f"Ollama brief parsing failed ({e}). Falling back to rule-based parser.")
        from floorgen.rag.brief_parser import parse_architectural_brief
        heuristic = parse_architectural_brief(brief)
        return {
            "rooms": heuristic.detected_rooms if heuristic.detected_rooms else ["living_room", "master_bedroom", "bathroom", "kitchen"],
            "affinities": [list(p) for p in heuristic.affinity_pairs],
            "style": heuristic.to_dict().get("style_preference", "standard"),
            "area_sqft": heuristic.estimated_sqft,
            "constraints": heuristic.style_keywords,
            "source": "heuristic_fallback"
        }
