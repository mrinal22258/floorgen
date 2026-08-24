"""
FAISS Vector Index Builder & Nearest-Neighbor Retriever for FloorGen.
Stores and searches floorplan dense vector embeddings with FAISS (and pure NumPy fallback).
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class FloorplanVectorIndex:
    """
    FAISS-based vector index for semantic floorplan retrieval.
    Includes persistence to disk and top-k nearest neighbor retrieval.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.plan_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.use_faiss = False
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            # Inner Product (cosine similarity on L2-normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False
            self.index = None

    def add_plans(self, plans: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds floorplans and their corresponding embeddings to the index."""
        if len(plans) != len(embeddings):
            raise ValueError(f"Mismatched counts: {len(plans)} plans vs {len(embeddings)} embeddings")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        for p in plans:
            self.plan_ids.append(p["id"])
            self.metadata.append({
                "id": p["id"],
                "num_rooms": p.get("num_rooms", len(p.get("rooms", []))),
                "archetype": p.get("archetype", "unknown"),
                "rooms": [r.get("category", "") for r in p.get("rooms", [])]
            })

        if self.use_faiss and self.index is not None:
            self.index.add(embeddings)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the index for the top-k nearest floorplans to a query vector."""
        if self.embeddings is None or len(self.plan_ids) == 0:
            return []

        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        top_k = min(top_k, len(self.plan_ids))

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_vector, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Pure NumPy cosine similarity fallback
            sims = np.dot(self.embeddings, query_vector.T).flatten()
            indices = np.argsort(-sims)[:top_k]
            scores = sims[indices]

        results = []
        for rank, idx in enumerate(indices):
            if idx < 0 or idx >= len(self.plan_ids):
                continue
            results.append({
                "rank": rank + 1,
                "plan_id": self.plan_ids[idx],
                "score": float(scores[rank]),
                "metadata": self.metadata[idx]
            })
        return results

    def save(self, directory: str):
        """Saves index and metadata to directory."""
        os.makedirs(directory, exist_ok=True)
        meta_file = os.path.join(directory, "faiss_metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "plan_ids": self.plan_ids,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim
            }, f, indent=2)

        emb_file = os.path.join(directory, "embeddings.npy")
        if self.embeddings is not None:
            np.save(emb_file, self.embeddings)

        if self.use_faiss and self.index is not None:
            import faiss
            index_file = os.path.join(directory, "faiss.index")
            faiss.write_index(self.index, index_file)

    def load(self, directory: str):
        """Loads index and metadata from directory."""
        meta_file = os.path.join(directory, "faiss_metadata.json")
        if not os.path.exists(meta_file):
            return False

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.plan_ids = data.get("plan_ids", [])
            self.metadata = data.get("metadata", [])
            self.embedding_dim = data.get("embedding_dim", self.embedding_dim)

        emb_file = os.path.join(directory, "embeddings.npy")
        if os.path.exists(emb_file):
            self.embeddings = np.load(emb_file)

        index_file = os.path.join(directory, "faiss.index")
        if os.path.exists(index_file) and self.use_faiss:
            try:
                import faiss
                self.index = faiss.read_index(index_file)
            except Exception:
                pass

        return True
