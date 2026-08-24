"""
Plan-to-Text Templating & Multi-Modal Vector Embedding.
Encodes floorplan bubble diagrams, room inventories, and spatial constraints into dense vector representations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import math


def plan_to_text_template(plan: Dict[str, Any]) -> str:
    """
    Serializes a floorplan dictionary into a structured descriptive text prompt
    capturing room counts, spatial adjacencies, and layout structure.
    """
    rooms = plan.get("rooms", [])
    num_rooms = len(rooms)
    
    # 1. Count room types
    type_counts: Dict[str, int] = {}
    room_id_to_type: Dict[int, str] = {}
    for r in rooms:
        cat = r.get("category", "room")
        type_counts[cat] = type_counts.get(cat, 0) + 1
        room_id_to_type[r["id"]] = cat

    summary_list = [f"{count} {cat}" for cat, count in sorted(type_counts.items())]
    room_summary = ", ".join(summary_list)

    # 2. Extract topological adjacencies
    adj_sentences = []
    for u, v in plan.get("adjacency", []):
        t_u = room_id_to_type.get(u, f"room_{u}")
        t_v = room_id_to_type.get(v, f"room_{v}")
        adj_sentences.append(f"{t_u} is adjacent to {t_v}")
    
    adj_text = ". ".join(adj_sentences) if adj_sentences else "Open floor configuration"

    # 3. Overall description
    archetype = plan.get("archetype", "residential floorplan")
    desc = plan.get("description", "")

    template = (
        f"Floorplan with {num_rooms} rooms: {room_summary}. "
        f"Connectivity layout: {adj_text}. "
        f"Archetype style: {archetype}. {desc}"
    ).strip()

    return template


class FloorplanEmbedder:
    """
    Vector embedder for floorplan queries and database plans.
    Supports SentenceTransformers when available, with a fast, deterministic
    geometric-topological vectorizer fallback requiring zero GPU/API cost.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.model = None
        self._try_load_model(model_name)

    def _try_load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"[FloorGen Embedder] Loaded SentenceTransformer '{model_name}' successfully.")
        except Exception:
            # Fallback to local topological feature embedder
            self.model = None

    def _fallback_embed(self, text_or_plan: Any) -> np.ndarray:
        """
        Deterministic topological & geometric feature embedding (384 dimensions).
        Encodes room histogram, graph degree sequence, and text hash embeddings.
        Runs entirely on CPU with zero network dependencies.
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        
        # If passed a plan dict directly
        if isinstance(text_or_plan, dict):
            text = plan_to_text_template(text_or_plan)
            rooms = text_or_plan.get("rooms", [])
            for r in rooms:
                cat_id = r.get("category_id", 0) % 16
                vec[cat_id] += 1.0
                vec[16 + cat_id] += r.get("area", 100.0) / 1000.0
            for u, v in text_or_plan.get("adjacency", []):
                vec[32 + (u % 16)] += 0.5
                vec[32 + (v % 16)] += 0.5
        else:
            text = str(text_or_plan)

        # Hash text tokens into remaining vector dimensions
        words = text.lower().replace(",", " ").replace(".", " ").split()
        for w in words:
            h = hash(w)
            idx = 64 + (abs(h) % (self.embedding_dim - 64))
            vec[idx] += 1.0 / (1.0 + math.log(1 + len(w)))

        # Normalize to unit sphere (L2 norm)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a natural language query or templated floorplan text."""
        if self.model is not None:
            try:
                emb = self.model.encode(text, convert_to_numpy=True)
                # Normalize
                norm = np.linalg.norm(emb)
                return emb / (norm + 1e-8)
            except Exception:
                pass
        return self._fallback_embed(text)

    def embed_plan(self, plan: Dict[str, Any]) -> np.ndarray:
        """Embeds a full floorplan dictionary."""
        text = plan_to_text_template(plan)
        if self.model is not None:
            try:
                emb = self.model.encode(text, convert_to_numpy=True)
                norm = np.linalg.norm(emb)
                return emb / (norm + 1e-8)
            except Exception:
                pass
        return self._fallback_embed(plan)

    def embed_batch_plans(self, plans: List[Dict[str, Any]]) -> np.ndarray:
        """Embeds a list of floorplans into a (N, D) numpy array."""
        embeddings = [self.embed_plan(p) for p in plans]
        return np.array(embeddings, dtype=np.float32)
