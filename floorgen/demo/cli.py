"""
FloorGen Command-Line Interface (CLI).
Allows synthesizing floorplans from constraints, querying RAG exemplars, and exporting SVGs.

Usage examples:
  python -m floorgen.demo.cli --rooms living_room,master_bedroom,bathroom,kitchen --top_k 5 --output output.svg
  python -m floorgen.demo.cli --brief "Spacious 3-bedroom family apartment with large living room" --output result.svg
"""

import os
import argparse
import json
import torch
import numpy as np

from floorgen.data.scripts.parse_rplan import load_plans_from_dir
from floorgen.data.scripts.generate_sample_data import generate_dataset
from floorgen.rag.retriever import FloorplanRAGRetriever
from floorgen.models.diffusion_core.house_diffusion import DDPMScheduler
from floorgen.models.diffusion_core.rag_diffusion import RAGFloorplanDiffusion
from floorgen.postprocess.vectorize import regularize_floorplan, export_svg, export_json_vector
from floorgen.eval.llm_judge import evaluate_structural_realism
from floorgen.data.dataset import ROOM_TYPE_TO_ID


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
    args = parser.parse_args()

    print("=" * 65)
    print("         FloorGen: Generative Floorplan Synthesis")
    print("=" * 65)

    room_list = [r.strip() for r in args.rooms.split(",") if r.strip()]
    print(f"Target Room List: {room_list}")
    if args.brief:
        print(f"User Brief: '{args.brief}'")

    # 1. Ingestion & RAG Indexing
    plans = load_plans_from_dir(args.data_dir)
    if len(plans) == 0:
        print(f"[FloorGen] No indexed plans found in '{args.data_dir}'. Generating dataset...")
        plans = generate_dataset(num_samples=100, output_dir=args.data_dir)

    retriever = FloorplanRAGRetriever()
    retriever.index_corpus(plans)

    # 2. Retrieve top-k exemplars
    exemplar_boxes = None
    if args.rag:
        print(f"[FloorGen RAG] Searching top-{args.top_k} nearest real floorplan exemplars...")
        retrieved = retriever.retrieve(room_list=room_list, text_brief=args.brief, top_k=args.top_k)
        print(f"[FloorGen RAG] Retrieved {len(retrieved)} matching exemplars:")
        for r in retrieved:
            print(f"   • [{r['plan_id']}] (Score: {r['score']:.3f}) - {r['plan'].get('archetype', 'plan')}")

        cond = retriever.format_conditioning_context(retrieved)
        if cond["exemplar_boxes"] is not None:
            exemplar_boxes = torch.tensor(cond["exemplar_boxes"], dtype=torch.float32).unsqueeze(0) # (1, K, N, 4)

    # 3. Model Generation
    print("[FloorGen] Running Vector Diffusion Generative Core...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDPMScheduler(num_timesteps=200).to(torch.device(device))
    model = RAGFloorplanDiffusion(hidden_dim=128, num_layers=4).to(device)

    # Prepare inputs
    max_rooms = max(len(room_list), 12)
    room_types_t = torch.zeros((1, max_rooms), dtype=torch.long, device=device)
    room_mask_t = torch.zeros((1, max_rooms), dtype=torch.bool, device=device)
    adj_matrix_t = torch.zeros((1, max_rooms, max_rooms), dtype=torch.float32, device=device)

    for i, r in enumerate(room_list[:max_rooms]):
        room_types_t[0, i] = ROOM_TYPE_TO_ID.get(r, 0)
        room_mask_t[0, i] = True
        # Default chain/star connectivity if unspecified
        if i > 0:
            adj_matrix_t[0, 0, i] = 1.0
            adj_matrix_t[0, i, 0] = 1.0

    if exemplar_boxes is not None:
        exemplar_boxes = exemplar_boxes.to(device)

    # Diffusion Reverse Denoising Sampling
    pred_boxes = model.sample(
        scheduler=scheduler,
        room_types=room_types_t,
        adj_matrix=adj_matrix_t,
        room_mask=room_mask_t,
        retrieved_exemplars=exemplar_boxes,
        num_inference_steps=30
    )

    raw_boxes_np = pred_boxes[0].cpu().numpy()
    mask_np = room_mask_t[0].cpu().numpy()

    # 4. Post-processing & Vectorization
    print("[FloorGen] Vectorizing & regularizing room geometry...")
    vector_plan = regularize_floorplan(raw_boxes_np, room_list, mask_np)

    # 5. Structural Realism Evaluation
    eval_stats = evaluate_structural_realism(vector_plan)
    print(f"[FloorGen Quality] Realism Score: {eval_stats['overall_realism']}% | Aspect Ratio: {eval_stats['aspect_ratio_score']} | Circulation: {eval_stats['circulation_score']}")

    # 6. Export SVG
    svg_code = export_svg(vector_plan)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(svg_code)

    json_output = os.path.splitext(args.output)[0] + ".json"
    export_json_vector(vector_plan, json_output)

    print(f"✨ Successfully generated floorplan!")
    print(f"   • SVG Vector: {args.output}")
    print(f"   • JSON Spec:  {json_output}")


if __name__ == "__main__":
    main()
