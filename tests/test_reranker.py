"""Tests for kb_compiler.qmd.reranker."""

import pytest

from kb_compiler.qmd.chunker import Chunk
from kb_compiler.qmd.reranker import NullReranker


@pytest.mark.asyncio
async def test_null_reranker():
    chunks = [
        Chunk(id="1", source_path="a.md", concept_name="A", section_header=None, content="x", token_count=1),
        Chunk(id="2", source_path="b.md", concept_name="B", section_header=None, content="y", token_count=1),
        Chunk(id="3", source_path="c.md", concept_name="C", section_header=None, content="z", token_count=1),
    ]
    reranker = NullReranker()
    result = await reranker.rerank("query", chunks, top_n=2)

    assert len(result) == 2
    assert result[0].chunk.id == "1"
    assert result[1].chunk.id == "2"
    assert result[0].score == 1.0
