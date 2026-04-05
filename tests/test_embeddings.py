"""Tests for kb_compiler.qmd.embeddings."""

import pytest

from kb_compiler.qmd.embeddings import (
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    _guess_dim,
    create_embedding_provider,
)


def test_guess_dim():
    assert _guess_dim("all-MiniLM-L6-v2") == 384
    assert _guess_dim("text-embedding-3-small") == 384
    assert _guess_dim("text-embedding-3-large") == 768
    assert _guess_dim("nomic-embed-text") == 768
    assert _guess_dim("some-model-1024") == 1024


def test_create_embedding_provider_auto():
    # When empty, falls back to sentence-transformers
    provider = create_embedding_provider(provider="sentence-transformers")
    assert provider.dim == 384


def test_ollama_provider_properties():
    p = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")
    assert p.dim == 768


def test_openai_provider_properties():
    p = OpenAIEmbeddingProvider(
        base_url="http://api.example.com",
        model="text-embedding-3-small",
        dim_value=384,
    )
    assert p.dim == 384


@pytest.mark.asyncio
async def test_ollama_embed_mock():
    p = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        dim_value=3,
    )
    import httpx

    original_post = httpx.AsyncClient.post

    async def fake_post(self, url, **kwargs):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"embeddings": [[0.1, 0.2, 0.3]]}

        return FakeResponse()

    httpx.AsyncClient.post = fake_post
    try:
        result = await p.embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 3
    finally:
        httpx.AsyncClient.post = original_post
