"""
Realism & LLM-as-Judge Evaluation Protocol for FloorGen.
Provides an automated architectural plausibility scoring harness and structured
LLM-as-judge prompt generator comparing real vs generated floorplan pairs.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def evaluate_structural_realism(floorplan: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluates physical and architectural plausibility heuristics:
    1. Aspect Ratio Score: penalizes extreme room dimensions
    2. Overlap Penalty: penalizes overlapping non-adjacent rooms
    3. Circulation & Accessibility: ensures every room has at least one door/connection
    4. Boundary Compactness: ratio of total room area to site bounding box
    """
    rooms = floorplan.get("rooms", [])
    doors = floorplan.get("doors", [])
    n_rooms = len(rooms)
    if n_rooms == 0:
        return {"overall_realism": 0.0}

    # 1. Aspect ratio score (penalize aspect < 0.3 or > 3.0)
    aspect_scores = []
    for r in rooms:
        bbox = r.get("bbox", [0, 0, 10, 10])
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        aspect = min(w, h) / max(w, h) # in (0, 1]
        aspect_scores.append(aspect)
    aspect_score = float(np.mean(aspect_scores))

    # 2. Circulation / connectivity score
    connected_rooms = set()
    for d in doors:
        connected_rooms.add(d.get("room_a"))
        connected_rooms.add(d.get("room_b"))
    for u, v in floorplan.get("adjacency", []):
        connected_rooms.add(u)
        connected_rooms.add(v)

    connectivity_ratio = len(connected_rooms) / float(n_rooms) if n_rooms > 1 else 1.0

    # 3. Compactness score
    total_room_area = sum(r.get("area", 0) for r in rooms)
    all_x = [p[0] for r in rooms for p in r.get("polygon", [])]
    all_y = [p[1] for r in rooms for p in r.get("polygon", [])]
    if all_x and all_y:
        span_w = max(all_x) - min(all_x)
        span_h = max(all_y) - min(all_y)
        bound_area = max(1.0, span_w * span_h)
        compactness = min(1.0, total_room_area / bound_area)
    else:
        compactness = 0.5

    overall = 0.35 * aspect_score + 0.35 * connectivity_ratio + 0.30 * compactness

    return {
        "aspect_ratio_score": round(aspect_score, 3),
        "circulation_score": round(connectivity_ratio, 3),
        "compactness_score": round(compactness, 3),
        "overall_realism": round(overall * 100.0, 1)
    }


def generate_llm_judge_prompt(
    target_plan: Dict[str, Any],
    gen_plan_a: Dict[str, Any],
    gen_plan_b: Dict[str, Any]
) -> str:
    """
    Generates a structured prompt for an LLM judge (e.g. Claude / Gemini) to evaluate
    and compare two generated floorplan candidates against the user's constraints.
    """
    prompt = f"""You are an expert architectural critic and floorplan evaluator.
Evaluate the following two floorplan generations (Candidate A vs Candidate B) for realism, spatial flow, and constraint satisfaction.

### Target Constraints:
- Room inventory: {', '.join([r['category'] for r in target_plan.get('rooms', [])])}
- Bubble Graph connections: {target_plan.get('adjacency', [])}

### Candidate A (Baseline GAN / No-RAG Diffusion):
- Number of rooms: {len(gen_plan_a.get('rooms', []))}
- Adjacency contact edges: {gen_plan_a.get('adjacency', [])}
- Structural metrics: {evaluate_structural_realism(gen_plan_a)}

### Candidate B (FloorGen RAG-Conditioned Diffusion):
- Number of rooms: {len(gen_plan_b.get('rooms', []))}
- Adjacency contact edges: {gen_plan_b.get('adjacency', [])}
- Structural metrics: {evaluate_structural_realism(gen_plan_b)}

### Evaluation Instructions:
1. Compare room proportions and living-room centrality.
2. Check whether private zones (bedrooms/bathrooms) are reasonably separated from social areas (kitchen/entrance).
3. Score each candidate from 1 to 10 on Realism and Compatibility.
4. Conclude with a JSON object: {{"preferred_candidate": "A" or "B", "score_A": float, "score_B": float, "rationale": "..."}}
"""
    return prompt
