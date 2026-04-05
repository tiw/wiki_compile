"""Standalone qmd CLI entry point."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kb_compiler.config import get_settings
from kb_compiler.qmd.chunker import Chunker
from kb_compiler.qmd.embeddings import create_embedding_provider
from kb_compiler.qmd.qmd_search import QmdSearchEngine
from kb_compiler.qmd.qmd_store import QmdIndexStore
from kb_compiler.qmd.reranker import (
    FlashRankReranker,
    LLMReranker,
    NullReranker,
)
from kb_compiler.core.llm import create_llm_client

app = typer.Typer(
    name="qmd",
    help="Query-markdown hybrid search CLI",
    rich_markup_mode="rich",
)
console = Console()


def _get_engine() -> QmdSearchEngine:
    """Initialize QmdSearchEngine from settings."""
    settings = get_settings()
    settings.ensure_directories()

    db_path = settings.meta_dir / "qmd.db"

    # Auto-detect embedding provider from local LLM settings if available
    embedder = create_embedding_provider(
        provider="auto",
        base_url=settings.local_llm_base_url if settings.llm_provider == "local" else "",
        model="",
        api_key=settings.local_llm_api_key if settings.llm_provider == "local" else "",
    )

    store = QmdIndexStore(db_path=db_path, embedding_dim=embedder.dim)

    return QmdSearchEngine(
        wiki_dir=settings.wiki_dir,
        store=store,
        embedder=embedder,
        chunker=Chunker(),
    )


def _make_reranker(rerank: Optional[str], settings):
    if not rerank or rerank == "none":
        return NullReranker()
    if rerank == "flashrank":
        return FlashRankReranker()
    if rerank == "llm":
        if settings.llm_provider == "local":
            llm = create_llm_client(
                api_key=settings.local_llm_api_key,
                base_url=settings.local_llm_base_url,
                model=settings.local_llm_model,
                code_mode=False,
                provider="local",
            )
        else:
            llm = create_llm_client(
                api_key=settings.kimi_api_key,
                base_url=settings.kimi_base_url,
                model=settings.kimi_model,
                code_mode=settings.kimi_code_mode,
                provider="kimi",
            )
        return LLMReranker(llm)
    raise typer.BadParameter(f"Unknown reranker: {rerank}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    rerank: Optional[str] = typer.Option(None, "--rerank", help="Reranker: none, flashrank, llm"),
):
    """Search the knowledge base with qmd."""
    settings = get_settings()
    try:
        engine = _get_engine()
    except ImportError:
        console.print(
            "[red]qmd requires sqlite-vec. Install with: pip install 'kb-compiler[qmd]'[/]"
        )
        raise typer.Exit(1)

    engine.reranker = _make_reranker(rerank, settings)

    async def run():
        try:
            results = await engine.retrieve(query, top_k=top_k)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)

        if not results:
            console.print("[yellow]No results found.[/]")
            return

        table = Table(title=f"Search Results for: {query}")
        table.add_column("Concept", style="cyan")
        table.add_column("Section", style="magenta")
        table.add_column("Score", justify="right")
        table.add_column("Snippet")

        for r in results:
            snippet = r.content.replace("\n", " ")[:120]
            table.add_row(
                r.concept_name,
                r.section_header or "",
                f"{r.score:.4f}",
                snippet,
            )

        console.print(table)

    asyncio.run(run())


@app.command()
def index_rebuild():
    """Rebuild the qmd search index."""
    try:
        engine = _get_engine()
    except ImportError:
        console.print(
            "[red]qmd requires sqlite-vec. Install with: pip install 'kb-compiler[qmd]'[/]"
        )
        raise typer.Exit(1)

    async def run():
        result = await engine.build_index()
        if result["indexed_chunks"] == 0:
            console.print("[yellow]No articles indexed.[/]")
        else:
            console.print(
                f"[green]Indexed {result['indexed_chunks']} chunks "
                f"from {result['indexed_concepts']} concepts.[/]"
            )

    asyncio.run(run())


@app.command()
def stats():
    """Show qmd index statistics."""
    settings = get_settings()
    db_path = settings.meta_dir / "qmd.db"

    if not db_path.exists():
        console.print("[yellow]qmd index not found. Run 'qmd index-rebuild' first.[/]")
        raise typer.Exit(1)

    settings.ensure_directories()
    try:
        embedder = create_embedding_provider(
            provider="auto",
            base_url=settings.local_llm_base_url if settings.llm_provider == "local" else "",
            model="",
            api_key=settings.local_llm_api_key if settings.llm_provider == "local" else "",
        )
        store = QmdIndexStore(db_path=db_path, embedding_dim=embedder.dim)
        info = store.stats()
    except ImportError:
        console.print(
            "[red]qmd requires sqlite-vec. Install with: pip install 'kb-compiler[qmd]'[/]"
        )
        raise typer.Exit(1)

    table = Table(title="qmd Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Chunks", str(info["total_chunks"]))
    table.add_row("Total Concepts", str(info["total_concepts"]))
    table.add_row("Embedding Dim", str(info["embedding_dim"]))
    table.add_row("DB Path", str(info["db_path"]))
    console.print(table)


if __name__ == "__main__":
    app()
