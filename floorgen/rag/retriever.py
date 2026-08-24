"""
High-Level RAG Retriever for FloorGen.
Coordinates vector similarity search (FAISS) with topological graph filtering (Neo4j/NetworkX)
to retrieve the top-k most relevant floorplan exemplars to condition generative models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from floorgen.rag.embed import FloorplanEmbedder, plan_to_text_template
from floorgen.rag.faiss_index import FloorplanVectorIndex
from floorgen.rag.neo4j_store import FloorplanGraphStore


class FloorplanRAGRetriever:
    """
    RAG Retriever that embeds queries, retrieves top-k exemplars from FAISS,
    and optionally re-ranks or filters them using graph topological queries.
    """

    def __init__(
        self,
        embedder: Optional[FloorplanEmbedder] = None,
        vector_index: Optional[FloorplanVectorIndex] = None,
        graph_store: Optional[FloorplanGraphStore] = None
    ):
        self.embedder = embedder or FloorplanEmbedder()
        self.vector_index = vector_index or FloorplanVectorIndex(embedding_dim=self.embedder.embedding_dim)
        self.graph_store = graph_store or FloorplanGraphStore()
        self.corpus_plans: Dict[str, Dict[str, Any]] = {}

    def index_corpus(self, plans: List[Dict[str, Any]]):
        """Indexes a full dataset of floorplans into both FAISS and Graph Store."""
        print(f"[FloorGen Retriever] Embedding and indexing {len(plans)} floorplans...")
        for p in plans:
            self.corpus_plans[p["id"]] = p
            self.graph_store.add_plan_graph(p)

        embeddings = self.embedder.embed_batch_plans(plans)
        self.vector_index.add_plans(plans, embeddings)
        print("[FloorGen Retriever] Indexing complete.")

    def retrieve(
        self,
        room_list: Optional[List[str]] = None,
        text_brief: Optional[str] = None,
        required_adjacencies: Optional[List[tuple]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k floorplans matching the query.
        Returns list of matched plan dictionaries with similarity scores and layout geometries.
        """
        # Formulate query text
        if text_brief:
            query_str = text_brief
            if room_list:
                query_str += f" Rooms: {', '.join(room_list)}."
        elif room_list:
            query_str = f"Floorplan containing {len(room_list)} rooms: {', '.join(room_list)}."
        else:
            query_str = "Residential floorplan layout"

        # 1. Embed query
        query_emb = self.embedder.embed_text(query_str)

        # 2. FAISS vector search
        raw_results = self.vector_index.search(query_emb, top_k=top_k * 3)

        # 3. Optional Graph filter
        valid_plan_ids = None
        if room_list and required_adjacencies:
            valid_plan_ids = set(self.graph_store.query_by_subgraph(
                required_rooms=room_list,
                required_adjacencies=required_adjacencies
            ))

        retrieved_plans = []
        for res in raw_results:
            pid = res["plan_id"]
            if valid_plan_ids is not None and pid not in valid_plan_ids:
                continue
            
            plan_obj = self.corpus_plans.get(pid)
            if plan_obj:
                retrieved_plans.append({
                    "plan_id": pid,
                    "score": res["score"],
                    "rank": len(retrieved_plans) + 1,
                    "plan": plan_obj
                })
            
            if len(retrieved_plans) >= top_k:
                break

        # If graph filter was too strict, fallback to top FAISS results
        if len(retrieved_plans) == 0 and len(raw_results) > 0:
            for res in raw_results[:top_k]:
                pid = res["plan_id"]
                plan_obj = self.corpus_plans.get(pid)
                if plan_obj:
                    retrieved_plans.append({
                        "plan_id": pid,
                        "score": res["score"],
                        "rank": len(retrieved_plans) + 1,
                        "plan": plan_obj
                    })

        return retrieved_plans

    def format_conditioning_context(self, retrieved: List[Dict[str, Any]], max_rooms: int = 12) -> Dict[str, Any]:
        """
        Formats retrieved top-k exemplars into conditioning vectors & matrices
        ready for consumption by the diffusion or GAN generative cores.
        """
        if not retrieved:
            return {
                "num_retrieved": 0,
                "context_features": None,
                "exemplar_boxes": None,
                "exemplar_adj": None
            }

        k = len(retrieved)
        exemplar_boxes = []
        exemplar_adj = []
        
        for item in retrieved:
            p = item["plan"]
            rooms = p.get("rooms", [])
            boxes = np.zeros((max_rooms, 4), dtype=np.float32)
            adj = np.zeros((max_rooms, max_rooms), dtype=np.float32)
            
            for i, r in enumerate(rooms[:max_rooms]):
                bbox = r.get("bbox", [0, 0, 50, 50])
                boxes[i] = [
                    (bbox[0] / 128.0) - 1.0,
                    (bbox[1] / 128.0) - 1.0,
                    (bbox[2] / 128.0) - 1.0,
                    (bbox[3] / 128.0) - 1.0
                ]
            for u, v in p.get("adjacency", []):
                if u < max_rooms and v < max_rooms:
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
            
            exemplar_boxes.append(boxes)
            exemplar_adj.append(adj)

        return {
            "num_retrieved": k,
            "exemplar_boxes": np.array(exemplar_boxes, dtype=np.float32),
            "exemplar_adj": np.array(exemplar_adj, dtype=np.float32),
            "scores": [item["score"] for item in retrieved],
            "plan_ids": [item["plan_id"] for item in retrieved]
        }
