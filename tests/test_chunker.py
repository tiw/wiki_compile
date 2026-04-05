"""Tests for kb_compiler.qmd.chunker."""

import pytest

from kb_compiler.qmd.chunker import Chunker


def test_chunk_simple_markdown():
    md = "# Title\n\nIntro paragraph.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    chunker = Chunker(max_tokens=512)
    chunks = chunker.chunk(md, source_path="test.md", concept_name="Test")

    assert len(chunks) >= 2
    assert any(c.section_header == "Section A" for c in chunks)
    assert any(c.section_header == "Section B" for c in chunks)
    for c in chunks:
        assert c.source_path == "test.md"
        assert c.concept_name == "Test"
        assert c.content


def test_chunk_no_headers():
    md = "Just a plain text without any headers."
    chunker = Chunker()
    chunks = chunker.chunk(md, source_path="plain.md", concept_name="Plain")
    assert len(chunks) == 1
    assert chunks[0].section_header is None
    assert "plain text" in chunks[0].content


def test_chunk_sliding_window():
    # Create a very long section to force sliding window
    long_text = "Word. " * 2000
    md = f"## Big Section\n\n{long_text}"
    chunker = Chunker(max_tokens=50, overlap_tokens=10)
    chunks = chunker.chunk(md, source_path="big.md", concept_name="Big")

    assert len(chunks) > 1
    # All chunks should belong to the same section
    for c in chunks:
        assert c.section_header == "Big Section"


def test_chunk_id_uniqueness():
    md = "## A\n\na\n\n## B\n\nb\n"
    chunker = Chunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(md, source_path="id.md", concept_name="Id")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
