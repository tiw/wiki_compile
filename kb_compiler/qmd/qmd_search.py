"""QmdSearchEngine: Orchestrate embedding, retrieval, and optional reranking."""

import frontmatter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

from kb_compiler.qmd.chunker import Chunk, Chunker
from kb_compiler.qmd.embeddings import EmbeddingProvider
from kb_compiler.qmd.qmd_store import QmdIndexStore, SearchResult
from kb_compiler.qmd.reranker import NullReranker, Reranker

console = Console()


@dataclass
class RetrievedChunk:
    """A chunk retrieved for a query."""

    chunk_id: str
    source_path: str
    concept_name: str
    section_header: Optional[str]
    content: str
    token_count: int
    score: float


class QmdSearchEngine:
    """End-to-end hybrid search engine for compiled wiki knowledge."""

    def __init__(
        self,
        wiki_dir: Path,
        store: QmdIndexStore,
        embedder: EmbeddingProvider,
        reranker: Optional[Reranker] = None,
        chunker: Optional[Chunker] = None,
    ):
        self.wiki_dir = wiki_dir
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or NullReranker()
        self.chunker = chunker or Chunker()

    async def build_index(self) -> dict:
        """Scan wiki, chunk, embed, and rebuild the index."""
        md_files = list(self.wiki_dir.rglob("*.md"))
        # Exclude INDEX.md and hidden files
        md_files = [f for f in md_files if not f.name.startswith(".") and f.name.lower() != "index.md"]

        if not md_files:
            console.print("[yellow]No wiki articles found to index.[/]")
            return {"indexed_chunks": 0, "indexed_concepts": 0}

        chunks: list[Chunk] = []
        for file_path in md_files:
            rel_path = str(file_path.relative_to(self.wiki_dir))
            content = file_path.read_text(encoding="utf-8")

            # Extract concept name from frontmatter or filename
            concept_name = _extract_concept_name(content, file_path)
            file_chunks = self.chunker.chunk(content, rel_path, concept_name)
            chunks.extend(file_chunks)

        if not chunks:
            console.print("[yellow]No chunks generated from wiki articles.[/]")
            return {"indexed_chunks": 0, "indexed_concepts": 0}

        console.print(f"[blue]Embedding {len(chunks)} chunks...[/]")
        # Batch embedding to avoid overwhelming the API
        batch_size = 32
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c.content for c in batch]
            batch_embs = await self.embedder.embed(batch_texts)
            embeddings.extend(batch_embs)

        console.print("[blue]Rebuilding qmd index...[/]")
        self.store.rebuild(chunks, embeddings)
        stats = self.store.stats()
        console.print(f"[green]Indexed {stats['total_chunks']} chunks across {stats['total_concepts']} concepts.[/]")
        return {
            "indexed_chunks": stats["total_chunks"],
            "indexed_concepts": stats["total_concepts"],
        }

    async def retrieve(self, query: str, top_k: int = 5, rerank_top_n: Optional[int] = None) -> list[RetrievedChunk]:
        """Retrieve top_k relevant chunks for a query."""
        # Check if index exists
        if not self.store.db_path.exists():
            raise FileNotFoundError(
                "qmd index not found. Run 'kb-compiler compile' or 'qmd index-rebuild' first."
            )

        # Embed query
        query_embs = await self.embedder.embed([query])
        query_embedding = query_embs[0]

        # Hybrid search
        results: list[SearchResult] = self.store.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            k=max(top_k * 4, 20),
        )

        if not results:
            return []

        # Optional reranking
        rerank_n = rerank_top_n or top_k
        if not isinstance(self.reranker, NullReranker) and len(results) > 1:
            chunks = [
                Chunk(
                    id=r.chunk_id,
                    source_path=r.source_path,
                    concept_name=r.concept_name,
                    section_header=r.section_header,
                    content=r.content,
                    token_count=r.token_count,
                )
                for r in results
            ]
            scored = await self.reranker.rerank(query, chunks, top_n=rerank_n)
            scored_ids = {s.chunk.id: s.score for s in scored}
            # Reorder results based on rerank scores
            results = sorted(
                results,
                key=lambda r: scored_ids.get(r.chunk_id, 0.0),
                reverse=True,
            )[:top_k]
        else:
            results = results[:top_k]

        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                source_path=r.source_path,
                concept_name=r.concept_name,
                section_header=r.section_header,
                content=r.content,
                token_count=r.token_count,
                score=r.score,
            )
            for r in results
        ]


def _extract_concept_name(content: str, file_path: Path) -> str:
    """Try to extract concept name from frontmatter title, fallback to filename."""
    try:
        post = frontmatter.loads(content)
        title = post.metadata.get("title")
        if title:
            return str(title)
    except Exception:
        pass
    return file_path.stem.replace("_", " ")
