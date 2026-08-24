"""
Unit tests for FloorGen RAG Index & Retriever (Stage 1 & Stage 2).
"""

import pytest
import numpy as np
from floorgen.data.scripts.generate_sample_data import generate_single_plan
from floorgen.rag.embed import FloorplanEmbedder, plan_to_text_template
from floorgen.rag.faiss_index import FloorplanVectorIndex
from floorgen.rag.neo4j_store import FloorplanGraphStore
from floorgen.rag.retriever import FloorplanRAGRetriever


def test_embedder_and_template():
    plan = generate_single_plan(plan_id=1, archetype_idx=0)
    text = plan_to_text_template(plan)
    assert "rooms" in text.lower()

    embedder = FloorplanEmbedder()
    emb = embedder.embed_plan(plan)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (embedder.embedding_dim,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


def test_faiss_vector_index():
    plans = [generate_single_plan(plan_id=i, archetype_idx=i) for i in range(10)]
    embedder = FloorplanEmbedder()
    embs = embedder.embed_batch_plans(plans)

    index = FloorplanVectorIndex(embedding_dim=embedder.embedding_dim)
    index.add_plans(plans, embs)

    # Search with first plan's embedding (should return itself as rank 1)
    results = index.search(embs[0], top_k=3)
    assert len(results) == 3
    assert results[0]["plan_id"] == plans[0]["id"]
    assert results[0]["score"] > 0.90


def test_rag_retriever_pipeline():
    plans = [generate_single_plan(plan_id=i, archetype_idx=i) for i in range(15)]
    retriever = FloorplanRAGRetriever()
    retriever.index_corpus(plans)

    # Retrieve by room list
    res = retriever.retrieve(room_list=["living_room", "master_bedroom", "kitchen"], top_k=3)
    assert len(res) == 3
    assert "plan" in res[0]

    # Format conditioning context
    cond = retriever.format_conditioning_context(res, max_rooms=10)
    assert cond["num_retrieved"] == 3
    assert cond["exemplar_boxes"].shape == (3, 10, 4)
