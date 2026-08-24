"""
Unit tests for FloorGen Evaluation Harness (Stage 5).
"""

import pytest
import numpy as np
from floorgen.data.scripts.generate_sample_data import generate_single_plan
from floorgen.eval.fid_kid import compute_distribution_metrics
from floorgen.eval.graph_edit_distance import evaluate_batch_compatibility
from floorgen.eval.llm_judge import evaluate_structural_realism, generate_llm_judge_prompt


def test_fid_kid_metrics():
    plans_a = [generate_single_plan(plan_id=i, archetype_idx=0) for i in range(10)]
    plans_b = [generate_single_plan(plan_id=i+10, archetype_idx=0) for i in range(10)]

    metrics = compute_distribution_metrics(plans_a, plans_b)
    assert "FID" in metrics
    assert "KID" in metrics
    assert metrics["FID"] >= 0.0


def test_graph_edit_distance():
    plan = generate_single_plan(plan_id=1, archetype_idx=0)
    res = evaluate_batch_compatibility([plan], [plan])
    assert res["mean_GED"] == 0.0
    assert res["compatibility_score"] == 1.0


def test_llm_judge():
    plan = generate_single_plan(plan_id=1, archetype_idx=0)
    stats = evaluate_structural_realism(plan)
    assert "overall_realism" in stats
    assert stats["overall_realism"] > 50.0

    prompt = generate_llm_judge_prompt(plan, plan, plan)
    assert "Candidate A" in prompt
    assert "Candidate B" in prompt
