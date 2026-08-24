"""
Compatibility & Graph Edit Distance (GED) Evaluation for FloorGen.
Measures whether the generated floorplan's physical room adjacency matches
the input bubble diagram constraint.
"""

from typing import List, Dict, Any, Tuple
import networkx as nx
import numpy as np


def build_graph_from_plan(plan: Dict[str, Any]) -> nx.Graph:
    """Builds a NetworkX graph from a floorplan dictionary."""
    G = nx.Graph()
    for r in plan.get("rooms", []):
        G.add_node(r["id"], category=r.get("category", "room"))
    for u, v in plan.get("adjacency", []):
        G.add_edge(u, v)
    return G


def compute_compatibility_ged(
    target_graph: nx.Graph,
    generated_graph: nx.Graph
) -> Tuple[float, float, float]:
    """
    Computes modified Graph Edit Distance and Edge Precision/Recall between
    input bubble diagram and reconstructed floorplan graph.
    Returns: (ged_distance, edge_precision, edge_recall)
    """
    target_edges = set([tuple(sorted(e)) for e in target_graph.edges()])
    gen_edges = set([tuple(sorted(e)) for e in generated_graph.edges()])

    if len(target_edges) == 0 and len(gen_edges) == 0:
        return 0.0, 1.0, 1.0

    # Insertions, deletions, substitutions
    missing_edges = target_edges - gen_edges
    spurious_edges = gen_edges - target_edges
    
    ged = len(missing_edges) + len(spurious_edges)
    
    # Precision & Recall
    precision = len(target_edges & gen_edges) / max(1, len(gen_edges))
    recall = len(target_edges & gen_edges) / max(1, len(target_edges))

    return float(ged), float(precision), float(recall)


def evaluate_batch_compatibility(
    input_plans: List[Dict[str, Any]],
    generated_plans: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Computes average GED and compatibility metrics over a test split."""
    geds = []
    precisions = []
    recalls = []

    for inp, gen in zip(input_plans, generated_plans):
        g_in = build_graph_from_plan(inp)
        g_out = build_graph_from_plan(gen)

        ged, p, r = compute_compatibility_ged(g_in, g_out)
        geds.append(ged)
        precisions.append(p)
        recalls.append(r)

    return {
        "mean_GED": round(float(np.mean(geds)), 3),
        "edge_precision": round(float(np.mean(precisions)), 3),
        "edge_recall": round(float(np.mean(recalls)), 3),
        "compatibility_score": round(float(1.0 / (1.0 + np.mean(geds))), 3)
    }
