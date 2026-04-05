"""Tests for kb_compiler.qmd.qmd_store."""

import pytest

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

from kb_compiler.qmd.chunker import Chunk
from kb_compiler.qmd.qmd_store import QmdIndexStore

pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec not installed")


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "qmd.db"
    return QmdIndexStore(db_path=db, embedding_dim=4)


def test_rebuild_and_stats(store):
    chunks = [
        Chunk(id="c1", source_path="a.md", concept_name="A", section_header=None, content="hello world", token_count=2),
        Chunk(id="c2", source_path="b.md", concept_name="B", section_header="Sec", content="foo bar", token_count=2),
    ]
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ]
    store.rebuild(chunks, embeddings)
    stats = store.stats()
    assert stats["total_chunks"] == 2
    assert stats["total_concepts"] == 2
    assert stats["embedding_dim"] == 4


def test_search_hybrid_basic(store):
    chunks = [
        Chunk(id="c1", source_path="a.md", concept_name="A", section_header=None, content="hello world", token_count=2),
        Chunk(id="c2", source_path="b.md", concept_name="B", section_header="Sec", content="foo bar baz", token_count=3),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.rebuild(chunks, embeddings)

    # Use the first embedding as query to get exact-ish match on c1
    results = store.search_hybrid("hello", query_embedding=[1.0, 0.0, 0.0, 0.0], k=10)
    assert len(results) > 0
    ids = {r.chunk_id for r in results}
    assert "c1" in ids


def test_search_hybrid_with_reranking_scores(store):
    chunks = [
        Chunk(id="c1", source_path="a.md", concept_name="A", section_header=None, content="alpha", token_count=1),
        Chunk(id="c2", source_path="a.md", concept_name="A", section_header=None, content="beta", token_count=1),
    ]
    embeddings = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    store.rebuild(chunks, embeddings)

    results = store.search_hybrid("alpha", query_embedding=[0.0, 0.0, 0.0, 0.0], k=10)
    assert len(results) >= 1
    for r in results:
        assert r.score > 0
        assert r.content in ("alpha", "beta")


def test_title_boost_prioritizes_concept_name_match(store):
    """When query matches concept_name, it should outrank body-only matches."""
    chunks = [
        Chunk(
            id="c1",
            source_path="a.md",
            concept_name="Agent",
            section_header=None,
            content="some generic text here",
            token_count=4,
        ),
        Chunk(
            id="c2",
            source_path="b.md",
            concept_name="Random",
            section_header="Agent Architecture",
            content="this page talks about agent a lot and has many agent words repeated agent agent",
            token_count=12,
        ),
    ]
    embeddings = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    store.rebuild(chunks, embeddings)

    results = store.search_hybrid("agent", query_embedding=[0.0, 0.0, 0.0, 0.0], k=10)
    ids = [r.chunk_id for r in results]
    assert ids[0] == "c1", f"Expected title-matching 'Agent' first, got {ids}"
