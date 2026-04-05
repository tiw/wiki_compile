"""Rerankers for qmd search results."""

from dataclasses import dataclass
from typing import Protocol

from kb_compiler.qmd.chunker import Chunk


class Reranker(Protocol):
    """Protocol for result rerankers."""

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list["ScoredChunk"]:
        """Rerank chunks and return top_n."""
        ...


@dataclass
class ScoredChunk:
    """Chunk with rerank score."""

    chunk: Chunk
    score: float


class NullReranker:
    """No-op reranker that preserves original order."""

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=c, score=1.0) for c in chunks[:top_n]
        ]


class FlashRankReranker:
    """Ultra-light cross-encoder reranker (requires flashrank)."""

    def __init__(self, model: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model = model
        self._ranker = None

    def _load(self):
        if self._ranker is None:
            try:
                from flashrank import Ranker
            except ImportError as e:
                raise ImportError(
                    "flashrank is not installed. "
                    "Install it with: pip install 'kb-compiler[qmd]'"
                ) from e
            self._ranker = Ranker(model_name=self.model)
        return self._ranker

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[ScoredChunk]:
        ranker = self._load()
        try:
            from flashrank import RerankRequest
        except ImportError as e:
            raise ImportError("flashrank is not installed.") from e

        passages = [
            {"id": i, "text": c.content, "meta": c.concept_name}
            for i, c in enumerate(chunks)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        scored = []
        for r in results[:top_n]:
            idx = int(r["id"])
            scored.append(ScoredChunk(chunk=chunks[idx], score=float(r["score"])))
        return scored


class LLMReranker:
    """Pointwise LLM reranker for high-quality deep research queries."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[ScoredChunk]:
        scored = []
        for chunk in chunks:
            prompt = f"""Rate how relevant this passage is to the query.

Query: {query}

Passage:
{chunk.content[:1000]}

Relevance score (1-10, 10 = highly relevant). Output ONLY the number."""
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt="You are a search relevance judge. Output only a number from 1 to 10.",
                temperature=0.0,
            )
            try:
                score = float(response.content.strip())
            except (ValueError, TypeError):
                score = 0.0
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]
