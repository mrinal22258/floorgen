"""
Unit tests for Local Ollama Qwen Architectural Brief Parser.
"""

import pytest
from floorgen.rag.llm_brief_parser import parse_brief_with_ollama


def test_llm_brief_parser_basic():
    brief = "Spacious 3 bedroom corner apartment with open kitchen, south facing balcony, and ensuite bathroom"
    result = parse_brief_with_ollama(brief)

    assert isinstance(result, dict)
    assert "rooms" in result
    assert "affinities" in result
    assert "master_bedroom" in result["rooms"]
    assert "kitchen" in result["rooms"]
    assert "balcony" in result["rooms"]
    assert len(result["rooms"]) >= 3


def test_llm_brief_parser_empty():
    result = parse_brief_with_ollama("")
    assert "living_room" in result["rooms"]
    assert "master_bedroom" in result["rooms"]
