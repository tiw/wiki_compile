"""Embedding providers for qmd search."""

import json
from dataclasses import dataclass
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    """Protocol for text embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    @property
    def dim(self) -> int:
        """Dimension of the embedding vectors."""
        ...


@dataclass
class OllamaEmbeddingProvider:
    """Embedding provider via Ollama or OpenAI-compatible API."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    api_key: str = ""
    dim_value: int = 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed endpoint."""
        # Normalize base_url
        url = self.base_url.rstrip("/") + "/api/embed"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        # Ollama /api/embed returns {"embeddings": [[...], [...]]}
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"Unexpected embedding response: {data.keys()}")
        return embeddings

    @property
    def dim(self) -> int:
        return self.dim_value


@dataclass
class OpenAIEmbeddingProvider:
    """Embedding provider via generic OpenAI-compatible /embeddings endpoint."""

    base_url: str
    model: str
    api_key: str = ""
    dim_value: int = 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = self.base_url.rstrip("/") + "/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model, "input": texts}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    @property
    def dim(self) -> int:
        return self.dim_value


class SentenceTransformerProvider:
    """Local embedding provider using sentence-transformers (lazy-loaded)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install 'kb-compiler[qmd]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        # all-MiniLM-L6-v2 is 384-dim
        if "mini" in self.model_name.lower():
            return 384
        return 768


def create_embedding_provider(
    provider: str = "auto",
    base_url: str = "",
    model: str = "",
    api_key: str = "",
) -> EmbeddingProvider:
    """Factory to create an appropriate embedding provider.

    Args:
        provider: 'auto', 'ollama', 'openai', or 'sentence-transformers'
        base_url: API base URL (for remote providers)
        model: Model name
        api_key: API key (for remote providers)
    """
    if provider == "auto":
        # Try Ollama first
        return _try_ollama_then_local(base_url, model, api_key)

    if provider == "ollama":
        url = base_url or "http://127.0.0.1:11434"
        mdl = model or "nomic-embed-text"
        return OllamaEmbeddingProvider(
            base_url=url,
            model=mdl,
            api_key=api_key,
            dim_value=_guess_dim(mdl),
        )

    if provider == "openai":
        if not base_url:
            raise ValueError("base_url is required for openai embedding provider")
        return OpenAIEmbeddingProvider(
            base_url=base_url,
            model=model or "text-embedding-3-small",
            api_key=api_key,
            dim_value=_guess_dim(model or "text-embedding-3-small"),
        )

    if provider == "sentence-transformers":
        return SentenceTransformerProvider(model_name=model or "all-MiniLM-L6-v2")

    raise ValueError(f"Unknown embedding provider: {provider}")


def _try_ollama_then_local(base_url: str, model: str, api_key: str) -> EmbeddingProvider:
    """Auto-detect: prefer Ollama if it seems configured, else sentence-transformers."""
    url = base_url or ""
    mdl = model or ""

    # Heuristic: if base_url contains 11434 or the model looks like an Ollama embed model
    if "11434" in url or ":8017" in url or "embed" in mdl.lower():
        return OllamaEmbeddingProvider(
            base_url=url or "http://127.0.0.1:11434",
            model=mdl or "nomic-embed-text",
            api_key=api_key,
            dim_value=_guess_dim(mdl or "nomic-embed-text"),
        )

    # Fallback to sentence-transformers
    return SentenceTransformerProvider(model_name=mdl or "all-MiniLM-L6-v2")


def _guess_dim(model_name: str) -> int:
    """Guess embedding dimension from model name."""
    name = model_name.lower()
    if "mini" in name or "small" in name:
        return 384
    if "nomic-embed-text" in name and "v1" not in name and "v1.5" not in name:
        return 768
    if "768" in name:
        return 768
    if "1024" in name:
        return 1024
    return 768
