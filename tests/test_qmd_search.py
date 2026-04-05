"""Tests for kb_compiler.qmd.qmd_search."""

import pytest

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

from kb_compiler.qmd.chunker import Chunker
from kb_compiler.qmd.embeddings import EmbeddingProvider
from kb_compiler.qmd.qmd_search import QmdSearchEngine, _extract_concept_name
from kb_compiler.qmd.qmd_store import QmdIndexStore
from kb_compiler.qmd.reranker import NullReranker

pytestmark = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec not installed")


class FakeEmbedder(EmbeddingProvider):
    """Deterministic embedder for tests."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Return a deterministic vector based on text length parity
        return [[1.0 if i % 2 == 0 else 0.0] + [0.0] * (self._dim - 1) for i in range(len(texts))]

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture
def engine(tmp_path):
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()

    # Write a fake concept article
    article = wiki_dir / "concepts"
    article.mkdir()
    (article / "Memory_Safety.md").write_text(
        "---\ntitle: Memory Safety\n---\n\n## Summary\n\nMemory safety prevents bugs.\n\n## Details\n\nUse after free is bad.\n",
        encoding="utf-8",
    )

    store = QmdIndexStore(db_path=tmp_path / "qmd.db", embedding_dim=4)
    embedder = FakeEmbedder(dim=4)
    return QmdSearchEngine(
        wiki_dir=wiki_dir,
        store=store,
        embedder=embedder,
        chunker=Chunker(max_tokens=50, overlap_tokens=10),
        reranker=NullReranker(),
    )


@pytest.mark.asyncio
async def test_build_index(engine):
    result = await engine.build_index()
    assert result["indexed_chunks"] > 0
    assert result["indexed_concepts"] == 1


@pytest.mark.asyncio
async def test_extract_concept_name_from_frontmatter():
    md = "---\ntitle: My Concept\n---\n\nBody"
    from pathlib import Path

    assert _extract_concept_name(md, Path("x.md")) == "My Concept"


@pytest.mark.asyncio
async def test_extract_concept_name_fallback():
    from pathlib import Path

    assert _extract_concept_name("no frontmatter", Path("my_concept.md")) == "my concept"


@pytest.mark.asyncio
async def test_retrieve(engine):
    await engine.build_index()
    results = await engine.retrieve("memory safety", top_k=3)
    assert len(results) > 0
    for r in results:
        assert r.concept_name == "Memory Safety"
        assert r.content


@pytest.mark.asyncio
async def test_retrieve_missing_index(engine, tmp_path):
    # Remove the db if it was created by fixture (it wasn't)
    with pytest.raises(FileNotFoundError):
        await engine.retrieve("anything")
