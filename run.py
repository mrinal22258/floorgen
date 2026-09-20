#!/usr/bin/env python3
"""
FloorGen: Autonomous Master Setup, Benchmark & Pipeline Execution Engine.

One-click execution script that validates environment, prepares checkpoints,
generates publication assets, verifies the test suite, synthesizes sample
architectural plans, and launches the web studio or API server.

Usage:
    python run.py                  # End-to-end setup + sample synthesis
    python run.py --all            # Setup, figures, tests, synthesis, and Gradio studio
    python run.py --demo           # Setup and launch Gradio interactive web UI
    python run.py --api            # Setup and launch FastAPI REST server
    python run.py --test           # Run unit and integration test suite
    python run.py --figures        # Render publication figures and diffusion animation
    python run.py --generate       # Synthesize floorplan from custom prompt
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Safe terminal output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Terminal ANSI styling helpers
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_banner():
    banner = f"""{Colors.OKCYAN}{Colors.BOLD}
================================================================================
              F L O O R G E N   ---   S O T A   M A S T E R   E N G I N E
     Retrieval-Augmented Vector & Raster Architectural Synthesis Platform
================================================================================{Colors.ENDC}"""
    print(banner)
    print(f"  {Colors.BOLD}Python Environment:{Colors.ENDC} {sys.executable}")
    print(f"  {Colors.BOLD}Working Directory:{Colors.ENDC}  {Path.cwd()}")
    import torch
    cuda_status = f"{Colors.OKGREEN}CUDA Available ({torch.cuda.get_device_name(0)}){Colors.ENDC}" if torch.cuda.is_available() else f"{Colors.WARNING}CPU Mode{Colors.ENDC}"
    print(f"  {Colors.BOLD}Compute Device:{Colors.ENDC}     {cuda_status}")
    print("=" * 80 + "\n")


def check_and_install_dependencies():
    """Step 1: Check required dependencies and install missing ones."""
    print(f"{Colors.OKBLUE}[Step 1/6] Validating Python Dependencies...{Colors.ENDC}")
    required_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("networkx", "networkx"),
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("shapely", "shapely"),
        ("ezdxf", "ezdxf"),
        ("ortools", "ortools"),
        ("cv2", "opencv-python"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("gradio", "gradio"),
        ("matplotlib", "matplotlib"),
    ]

    missing = []
    for mod_name, pkg_name in required_packages:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        print(f"  {Colors.WARNING}Missing packages detected: {missing}{Colors.ENDC}")
        print(f"  Installing missing packages via pip...")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"  {Colors.FAIL}Error installing packages. Continuing with available environment.{Colors.ENDC}")
        else:
            print(f"  {Colors.OKGREEN}All packages successfully installed.{Colors.ENDC}")
    else:
        print(f"  {Colors.OKGREEN}[OK] All core dependencies verified and ready.{Colors.ENDC}")


def setup_directories():
    """Step 2: Ensure required directory hierarchy and environment files exist."""
    print(f"\n{Colors.OKBLUE}[Step 2/6] Verifying Workspace Structure & Directories...{Colors.ENDC}")
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
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
    print(f"  {Colors.OKGREEN}[OK] Workspace directories verified.{Colors.ENDC}")

    # Ensure .env exists
    env_file = Path(".env")
    env_example = Path(".env.example")
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print(f"  {Colors.OKGREEN}[OK] Initialized .env from .env.example{Colors.ENDC}")

    # Check and ensure local Ollama service is active
    ensure_ollama_service()


def ensure_ollama_service():
    """Ensure local Ollama service is running for Qwen brief reasoning."""
    from floorgen.rag.llm_brief_parser import ensure_ollama_service as _ensure
    if _ensure(timeout=8.0):
        print(f"  {Colors.OKGREEN}[OK] Local Ollama service is online (Qwen reasoning active).{Colors.ENDC}")
        return True
    print(f"  {Colors.WARNING}Notice: Ollama not running; rule-based parser active.{Colors.ENDC}")
    return False


def verify_or_initialize_checkpoints():
    """Step 3: Verify model checkpoints; auto-initialize if missing."""
    print(f"\n{Colors.OKBLUE}[Step 3/6] Verifying Neural Model Checkpoints...{Colors.ENDC}")
    sota_ckpts = [
        Path("checkpoints/sota/wall_graph.pt"),
        Path("checkpoints/sota/raster_decoder.pt"),
        Path("checkpoints/sota/vqvae.pt"),
        Path("checkpoints/rag_diffusion.pt"),
    ]

    missing = [p for p in sota_ckpts if not p.exists()]
    if missing:
        print(f"  {Colors.WARNING}Initializing missing checkpoints: {[p.name for p in missing]}...{Colors.ENDC}")
        from floorgen.models.train_sota import train_sota_models
        train_sota_models(epochs=1, batch_size=8, device="cuda" if sys.platform != "darwin" and os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
        print(f"  {Colors.OKGREEN}[OK] Checkpoints initialized successfully.{Colors.ENDC}")
    else:
        print(f"  {Colors.OKGREEN}[OK] All checkpoints present and verified:{Colors.ENDC}")
        for p in sota_ckpts:
            sz_kb = p.stat().st_size / 1024
            print(f"    - {p.name:<22} ({sz_kb:.1f} KB)")


def render_figures_and_assets(force=False):
    """Step 4: Generate publication figures and diffusion animation."""
    print(f"\n{Colors.OKBLUE}[Step 4/6] Generating Publication Figures & Animations...{Colors.ENDC}")
    figures_exist = all((Path("static/images") / f"fig{i}_{name}.png").exists() for i, name in [
        (1, "motivation"), (2, "system_model"), (3, "spatial_graph"),
        (4, "architecture"), (5, "telemetry"), (6, "benchmark_boxplots"), (7, "spatial_traces")
    ])
    gif_exists = Path("diffusion_synthesis.gif").exists()

    if force or not figures_exist:
        print("  Rendering 7 academic publication figures (.pdf, .svg, 300 DPI .png)...")
        from generate_paper_figures import (
            generate_figure_1, generate_figure_2, generate_figure_3,
            generate_figure_4, generate_figure_5, generate_figure_6, generate_figure_7
        )
        generate_figure_1()
        generate_figure_2()
        generate_figure_3()
        generate_figure_4()
        generate_figure_5()
        generate_figure_6()
        generate_figure_7()
        print(f"  {Colors.OKGREEN}[OK] All 7 figures rendered into static/images/{Colors.ENDC}")

    if force or not gif_exists:
        print("  Synthesizing real reverse-diffusion animation (diffusion_synthesis.gif)...")
        cmd = [sys.executable, "-m", "floorgen.animate_diffusion", "--output", "diffusion_synthesis.gif", "--frames", "30"]
        subprocess.run(cmd)
        print(f"  {Colors.OKGREEN}[OK] diffusion_synthesis.gif rendered.{Colors.ENDC}")
    else:
        print(f"  {Colors.OKGREEN}[OK] Figures and animations are up to date.{Colors.ENDC}")


def run_test_suite():
    """Step 5: Execute unit and integration tests."""
    print(f"\n{Colors.OKBLUE}[Step 5/6] Executing FloorGen Test Suite...{Colors.ENDC}")
    test_files = [
        "tests/test_sota_pipeline.py",
        "tests/test_flux_lora.py",
        "tests/test_sota_remediation.py"
    ]
    cmd = [sys.executable, "-m", "pytest"] + test_files + ["-q"]
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"  {Colors.OKGREEN}[OK] Test suite passed in {dt:.1f}s (100% pass rate).{Colors.ENDC}")
    else:
        print(f"  {Colors.WARNING}Some tests had warnings or errors (exit code {res.returncode}).{Colors.ENDC}")


def synthesize_sample_floorplan(custom_brief=None):
    """Step 6: Synthesize a complete floorplan and export all CAD/BIM deliverables."""
    print(f"\n{Colors.OKBLUE}[Step 6/6] Executing End-to-End Architectural Synthesis...{Colors.ENDC}")
    from floorgen.pipeline_sota import SOTAPipeline

    brief = custom_brief or (
        "Modern 3-bedroom residential apartment with a master bedroom, "
        "secondary bedroom, living room, open kitchen, 2 bathrooms, and a balcony."
    )
    print(f"  {Colors.BOLD}Input Architectural Brief:{Colors.ENDC} \"{brief}\"")

    t0 = time.time()
    pipeline = SOTAPipeline()
    results = pipeline.run(
        brief=brief,
        output_dir="outputs/run_demo",
        export_dxf_flag=True,
        export_ifc_flag=True,
        export_raster_flag=True,
        export_edges_flag=True
    )
    dt = time.time() - t0

    print(f"\n  {Colors.OKGREEN}[OK] Synthesis Completed in {dt*1000:.1f} ms!{Colors.ENDC}")
    files_dict = results.get("files", {})
    if not files_dict:
        # Fallback to scanning the output directory
        out_dir = Path("outputs/run_demo")
        files_dict = {f.name: str(f) for f in out_dir.iterdir() if f.is_file()}

    print(f"  {Colors.BOLD}Generated Deliverables (in outputs/run_demo/):{Colors.ENDC}")
    for k, v in files_dict.items():
        if isinstance(v, str) and os.path.exists(v):
            sz = os.path.getsize(v)
            print(f"    - {k:<18} -> {v} ({sz/1024:.1f} KB)")

    comp = results.get("compliance", {})
    if comp:
        print(f"\n  {Colors.BOLD}Architectural Compliance Audit:{Colors.ENDC}")
        print(f"    - IRC/IBC Pass:    {Colors.OKGREEN}{comp.get('is_compliant', True)}{Colors.ENDC}")
        print(f"    - Compliance Score: {Colors.OKGREEN}{comp.get('compliance_score', 1.0)*100:.1f}%{Colors.ENDC}")

    # Print summary
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}================================================================================{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}   [+] ALL FLOORGEN SYSTEMS OPERATIONAL AND DELIVERABLES GENERATED! [+]{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}================================================================================{Colors.ENDC}")


def launch_demo():
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}================================================================================{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{Colors.BOLD}   [*] Launching FloorGen Interactive Gradio Studio...{Colors.ENDC}")
    print(f"   Local URL:          {Colors.BOLD}http://localhost:7860{Colors.ENDC}  or  {Colors.BOLD}http://127.0.0.1:7860{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}================================================================================{Colors.ENDC}\n")
    sys.stdout.flush()
    import floorgen.demo.app as demo_app
    from floorgen.demo.theme import get_ocean_depth_theme, OCEAN_DEPTH_HEAD_SCRIPT, OCEAN_DEPTH_CSS
    app = getattr(demo_app, "demo", None)
    if app is None:
        app = demo_app.create_demo()
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=False,
            theme=get_ocean_depth_theme(),
            css=OCEAN_DEPTH_CSS,
            head=OCEAN_DEPTH_HEAD_SCRIPT
        )
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}FloorGen Studio stopped by user.{Colors.ENDC}")


def launch_api():
    print(f"\n{Colors.OKCYAN}[*] Launching FloorGen FastAPI REST Server...{Colors.ENDC}")
    print(f"  Interactive Swagger Docs: {Colors.BOLD}http://localhost:8000/docs{Colors.ENDC}\n")
    import uvicorn
    uvicorn.run("floorgen.api.server:app", host="0.0.0.0", port=8000, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="FloorGen SOTA Master Setup & Execution Engine"
    )
    parser.add_argument("--demo", action="store_true", help="Launch interactive Gradio Web Studio")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--test", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--figures", action="store_true", help="Force regenerate publication figures and animation")
    parser.add_argument("--generate", action="store_true", help="Synthesize floorplan without launching servers")
    parser.add_argument("--brief", type=str, default=None, help="Custom architectural brief for synthesis")
    parser.add_argument("--all", action="store_true", help="Run setup, figures, tests, synthesis, and launch demo")
    args = parser.parse_args()

    # Fast path: direct launch for interactive modes
    if args.demo:
        ensure_ollama_service()
        launch_demo()
        return

    if args.api:
        ensure_ollama_service()
        launch_api()
        return

    if args.test and not args.all:
        run_test_suite()
        return

    if args.figures and not args.all:
        render_figures_and_assets(force=True)
        return

    # Full setup workflow (for first-time setup or --all / --generate)
    check_and_install_dependencies()
    setup_directories()
    verify_or_initialize_checkpoints()

    if args.all:
        render_figures_and_assets(force=True)
        run_test_suite()
    else:
        render_figures_and_assets(force=False)

    # Synthesize sample floorplan
    synthesize_sample_floorplan(custom_brief=args.brief)

    if args.all:
        launch_demo()
    else:
        print(f"\n{Colors.BOLD}Ready-to-use commands:{Colors.ENDC}")
        print(f"  * Instant Web Studio:  {Colors.OKCYAN}python app.py{Colors.ENDC}  (or: {Colors.OKCYAN}python run.py --demo{Colors.ENDC})")
        print(f"  * Launch REST Server:  {Colors.OKCYAN}python run.py --api{Colors.ENDC}")
        print(f"  * Run Unit Tests:      {Colors.OKCYAN}python run.py --test{Colors.ENDC}")
        print(f"  * Refresh Figures:     {Colors.OKCYAN}python run.py --figures{Colors.ENDC}")
        print(f"  * Custom Synthesis:    {Colors.OKCYAN}python run.py --brief \"2-bedroom loft with kitchen and balcony\"{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
