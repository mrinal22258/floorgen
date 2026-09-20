"""
FloorGen Command-Line Interface (CLI).
High-performance synthesis using pre-warmed FloorGenPipeline singleton.
"""

import os
import argparse
import json
import torch

from floorgen import __version__
from floorgen.config import FloorGenConfig, default_config
from floorgen.pipeline import FloorGenPipeline
from floorgen.postprocess.vectorize import export_json_vector


def main():
    parser = argparse.ArgumentParser(description="FloorGen: Generative Floorplan Synthesis CLI")
    parser.add_argument("--rooms", type=str, default="living_room,master_bedroom,bathroom,kitchen,balcony",
                        help="Comma-separated list of desired room categories")
    parser.add_argument("--brief", type=str, default=None,
                        help="Natural-language floorplan description brief")
    parser.add_argument("--rag", action="store_true", default=True,
                        help="Enable RAG exemplar retrieval conditioning")
    parser.add_argument("--no_rag", dest="rag", action="store_false")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of nearest exemplars to retrieve from RAG index")
    parser.add_argument("--output", type=str, default="generated_floorplan.svg",
                        help="Output path for rendered SVG")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Path to floorplan corpus directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rag_diffusion.pt",
                        help="Path to trained model weights checkpoint")
    parser.add_argument("--solver", action="store_true", default=True,
                        help="Enable Google OR-Tools CP-SAT architectural constraint solver")
    parser.add_argument("--no_solver", dest="solver", action="store_false")
    parser.add_argument("--export_dxf", action="store_true", default=True,
                        help="Export professional AutoCAD DXF deliverable")
    parser.add_argument("--export_ifc", action="store_true", default=True,
                        help="Export ISO-16739 compliant IFC BIM model deliverable")
    parser.add_argument("--sampling", type=str, choices=["ddim", "ddpm"], default="ddim",
                        help="Sampling method (fast DDIM or stochastic DDPM)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of reverse diffusion sampling steps")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of layout variants to sample (selects highest realism score)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cuda or cpu)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"         FloorGen v{__version__}: Generative Floorplan Synthesis")
    print("=" * 65)

    # If rooms specified explicitly, use them; if only brief given, let pipeline infer via Ollama
    room_list = None
    if "--rooms" in os.sys.argv:
        room_list = [r.strip() for r in args.rooms.split(",") if r.strip()]
        print(f"Target Room List: {room_list}")
    elif not args.brief:
        room_list = [r.strip() for r in args.rooms.split(",") if r.strip()]
        print(f"Default Room List: {room_list}")

    if args.brief:
        print(f"User Brief: '{args.brief}'")

    # Configure pipeline overrides
    cfg = FloorGenConfig(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        top_k=args.top_k,
        default_sampling=args.sampling,
        default_steps=args.steps,
        enable_solver=args.solver
    )

    pipeline = FloorGenPipeline.get_instance(cfg)
    device_str = str(pipeline.device)
    print(f"[FloorGen] Running Vector Diffusion Generative Core on device: {device_str}")
    if device_str == "cuda":
        print(f"[FloorGen] GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"[FloorGen] Loaded trained checkpoint from '{cfg.checkpoint_path}' (Epoch {pipeline.checkpoint_epoch})")

    if args.rag and pipeline.retriever:
        print(f"[FloorGen RAG] Searching top-{args.top_k} nearest real floorplan exemplars (MMR Diverse)...")

    print(f"[FloorGen] Synthesizing {max(1, args.batch)} layout candidate(s) via {args.sampling.upper()} ({args.steps} steps)...")
    res = pipeline.generate(
        rooms=room_list,
        brief=args.brief,
        top_k=args.top_k,
        batch=args.batch,
        sampling=args.sampling,
        steps=args.steps,
        solver=args.solver,
        export_dxf_flag=args.export_dxf,
        export_ifc_flag=args.export_ifc
    )

    if res.exemplars:
        print(f"[FloorGen RAG] Retrieved {len(res.exemplars)} matching exemplars:")
        for r in res.exemplars:
            p = r.get("plan", {})
            print(f"   - [{r.get('plan_id')}] (Score: {r.get('score', 0):.3f}) - {p.get('archetype', 'real_rplan')}")

    print(f"[FloorGen Quality] Realism Score: {res.realism_score}% | Aspect Ratio: {res.aspect_ratio:.3f} | Circulation: {res.circulation}")
    if res.compliance:
        pass_status = "PASSED" if res.compliance.get("passed") else "WARNINGS"
        print(f"[FloorGen Compliance] Building Code Audit (IRC/IBC/ADA): {pass_status} ({res.compliance.get('compliance_score', 1.0)*100:.1f}%)")

    # Export Deliverables (SVG, JSON, DXF, IFC)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(res.svg)

    json_output = os.path.splitext(args.output)[0] + ".json"
    export_json_vector(res.json_spec, json_output)

    dxf_output = None
    if args.export_dxf and res.dxf_content:
        dxf_output = os.path.splitext(args.output)[0] + ".dxf"
        with open(dxf_output, "w", encoding="utf-8") as f:
            f.write(res.dxf_content)

    ifc_output = None
    if args.export_ifc and res.ifc_content:
        ifc_output = os.path.splitext(args.output)[0] + ".ifc"
        with open(ifc_output, "w", encoding="utf-8") as f:
            f.write(res.ifc_content)

    print(f"[FloorGen] Successfully synthesized floorplan deliverables:")
    print(f"   * SVG Vector:   {args.output}")
    print(f"   * JSON Spec:    {json_output}")
    if dxf_output:
        print(f"   * AutoCAD DXF:  {dxf_output}")
    if ifc_output:
        print(f"   * BIM Model:    {ifc_output}")



if __name__ == "__main__":
    main()
