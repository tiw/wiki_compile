"""Main CLI entry point for kb-compiler."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kb_compiler.config import Settings, get_settings
from kb_compiler.core.llm import create_llm_client
from kb_compiler.core.metadata import MetadataTracker
from kb_compiler.core.obsidian import ObsidianClient
from kb_compiler.phases.compile import CompilationPipeline
from kb_compiler.phases.ingest import DocumentIngester, QuickCapture
from kb_compiler.phases.maintenance import FeedbackManager, WikiLinter
from kb_compiler.phases.query import QueryEngine, QueryOutputFormatter

app = typer.Typer(
    name="kb-compiler",
    help="LLM Knowledge Compiler - Transform raw documents into structured wiki",
    rich_markup_mode="rich",
)
console = Console()


def get_clients(settings: Settings):
    """Initialize and return all clients."""
    errors = settings.validate()
    if errors:
        for error in errors:
            console.print(f"[red]Config Error: {error}[/]")
        if settings.llm_provider == "kimi":
            console.print("\n[yellow]Set KIMI_API_KEY environment variable to use Kimi API.[/]")
        elif settings.llm_provider == "local":
            console.print("\n[yellow]Make sure local LLM is running at the configured URL.[/]")
        raise typer.Exit(1)

    # Select LLM configuration based on provider
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

    obsidian = ObsidianClient(
        vault_name=settings.obsidian_vault,
        vault_path=settings.obsidian_vault_path,
    )

    metadata = MetadataTracker(settings.meta_dir)

    return llm, obsidian, metadata


@app.callback()
def main(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    vault: Optional[str] = typer.Option(
        None, "--vault", "-v", help="Obsidian vault name or path"
    ),
):
    """Knowledge Compiler - Compile raw documents into structured wiki."""
    # Set vault from CLI if provided
    if vault:
        if "/" in vault or "\\" in vault or vault.startswith("~"):
            # It's a path
            os.environ["KB_OBSIDIAN_VAULT_PATH"] = str(Path(vault).expanduser())
        else:
            # It's a vault name
            os.environ["KB_OBSIDIAN_VAULT"] = vault


@app.command()
def init(
    path: Path = typer.Argument(
        ..., help="Path to create knowledge base"
    ),
    vault: Optional[str] = typer.Option(
        None, "--vault", help="Obsidian vault to link"
    ),
):
    """Initialize a new knowledge base directory."""
    console.print(Panel.fit(f"Initializing Knowledge Base at [cyan]{path}[/]"))

    # Create directory structure
    (path / "raw" / "articles").mkdir(parents=True, exist_ok=True)
    (path / "raw" / "papers").mkdir(parents=True, exist_ok=True)
    (path / "raw" / "inbox").mkdir(parents=True, exist_ok=True)
    (path / "wiki" / "concepts").mkdir(parents=True, exist_ok=True)
    (path / "output").mkdir(parents=True, exist_ok=True)
    (path / "_meta").mkdir(parents=True, exist_ok=True)

    # Create initial INDEX.md
    index_content = """---
title: Knowledge Base Index
last_updated: null
total_concepts: 0
---

# Knowledge Base Index

## Overview
This knowledge base is empty. Run `kb-compiler compile` after adding documents.

## Directory Structure

- `raw/` - Original documents
  - `articles/` - Web articles, blog posts
  - `papers/` - Academic papers, PDFs
  - `inbox/` - Quick captures and notes
- `wiki/` - Compiled knowledge
  - `concepts/` - Individual concept articles
  - `INDEX.md` - This file
- `output/` - Query results and exports
- `_meta/` - Compilation metadata
"""

    index_path = path / "wiki" / "INDEX.md"
    index_path.write_text(index_content)

    # Create config template
    config_template = f"""# Knowledge Compiler Configuration
# Place this in ~/.config/kb-compiler/config.yaml

kimi_api_key: "your-api-key-here"  # Set via KIMI_API_KEY environment variable
kimi_model: "moonshot-v1-128k"

kb_root: "{path}"

obsidian_vault_path: "{vault or str(path / 'wiki')}"
"""

    config_path = path / "config.yaml.example"
    config_path.write_text(config_template)

    console.print(f"[green]Created knowledge base at {path}[/]")
    console.print(f"[dim]Example config: {config_path}[/]")
    console.print("\n[bold]Next steps:[/]")
    console.print("1. Add documents to raw/ directory")
    console.print("2. Set KIMI_API_KEY environment variable")
    console.print("3. Run: kb-compiler compile")


@app.command()
def ingest(
    sources: list[str] = typer.Argument(..., help="URL(s), file path(s), or directory to ingest (supports wildcards)"),
    subdir: str = typer.Option("", "--subdir", "-d", help="Subdirectory in raw/"),
    tags: list[str] = typer.Option([], "--tag", "-t", help="Tags to add"),
):
    """Ingest documents from URL, file, directory, or glob pattern."""
    settings = get_settings()
    settings.ensure_directories()

    ingester = DocumentIngester(settings.raw_dir)
    import glob

    total_ingested = 0

    for source in sources:
        # Handle glob patterns
        if '*' in source or '?' in source:
            matched_files = glob.glob(source)
            if not matched_files:
                console.print(f"[yellow]No files match pattern: {source}[/]")
                continue

            for file_path_str in matched_files:
                source_path = Path(file_path_str)
                if source_path.is_file():
                    try:
                        path = ingester.ingest_file(source_path, subdir, metadata={"tags": tags})
                        total_ingested += 1
                    except Exception as e:
                        console.print(f"[yellow]Failed to ingest {source_path}: {e}[/]")
            continue

        source_path = Path(source)

        if source.startswith(("http://", "https://")):
            # Ingest URL
            path = ingester.ingest_url(source, metadata={"tags": tags})
            console.print(f"[green]Ingested to: {path}[/]")
            total_ingested += 1

        elif source_path.is_file():
            # Ingest file
            path = ingester.ingest_file(source_path, subdir, metadata={"tags": tags})
            console.print(f"[green]Ingested to: {path}[/]")
            total_ingested += 1

        elif source_path.is_dir():
            # Ingest directory
            paths = ingester.ingest_directory(source_path)
            total_ingested += len(paths)

        else:
            console.print(f"[red]Source not found: {source}[/]")

    if total_ingested > 0:
        console.print(f"[green]Total ingested: {total_ingested} item(s)[/]")
    else:
        raise typer.Exit(1)


@app.command()
def capture(
    content: Optional[str] = typer.Argument(None, help="Content to capture"),
    editor: bool = typer.Option(False, "--editor", "-e", help="Open editor"),
):
    """Quick capture a note to inbox."""
    settings = get_settings()
    settings.ensure_directories()

    quick = QuickCapture(settings.raw_dir)

    if editor or not content:
        # Open editor
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as f:
            if content:
                f.write(content)
            temp_path = f.name

        editor_cmd = os.environ.get("EDITOR", "vim")
        subprocess.call([editor_cmd, temp_path])

        content = Path(temp_path).read_text()
        Path(temp_path).unlink()

    path = quick.capture(content)
    console.print(f"[green]Captured to: {path}[/]")


@app.command()
def compile(
    full: bool = typer.Option(False, "--full", help="Force full recompile"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be compiled"),
):
    """Compile raw documents into wiki."""
    settings = get_settings()
    settings.ensure_directories()

    llm, obsidian, metadata = get_clients(settings)

    if dry_run:
        changed = metadata.get_changed_files(settings.raw_dir)
        console.print(f"[blue]Changed files ({len(changed)}):[/]")
        for f in changed[:20]:
            console.print(f"  - {f}")
        if len(changed) > 20:
            console.print(f"  ... and {len(changed) - 20} more")
        return

    pipeline = CompilationPipeline(
        llm_client=llm,
        obsidian_client=obsidian,
        metadata_tracker=metadata,
        raw_dir=settings.raw_dir,
        wiki_dir=settings.wiki_dir,
    )

    async def run():
        result = await pipeline.run(incremental=not full)

        if result["status"] == "success":
            console.print(f"[green]Compiled {result['files_processed']} files[/]")
            console.print(f"[green]Created {len(result['concepts'])} concepts:[/]")
            for concept in result["concepts"]:
                console.print(f"  - {concept}")
        elif result["status"] == "no_changes":
            console.print("[yellow]No changes to compile.[/]")
        else:
            console.print(f"[red]Compilation failed: {result}[/]")

    asyncio.run(run())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    save: Optional[str] = typer.Option(None, "--save", "-s", help="Save result to file"),
    explore: Optional[str] = typer.Option(None, "--explore", "-e", help="Explore a concept"),
    compare: tuple[str, str] = typer.Option((None, None), "--compare", "-c", help="Compare two concepts"),
):
    """Query the compiled wiki knowledge."""
    settings = get_settings()
    llm, obsidian, _ = get_clients(settings)

    engine = QueryEngine(llm, obsidian, settings.wiki_dir)

    async def run():
        if explore:
            result = await engine.explore(explore)
        elif compare[0] and compare[1]:
            result = await engine.compare(compare[0], compare[1])
        else:
            result = await engine.query(question)

        # Display result
        QueryOutputFormatter.format_terminal(result)

        # Save if requested
        if save:
            feedback = FeedbackManager(settings.output_dir, obsidian)
            filepath = feedback.save_query_result(question or explore or f"{compare[0]}_vs_{compare[1]}", result)
            console.print(f"[dim]Saved to: {filepath}[/]")

    asyncio.run(run())


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues where possible"),
    contradictions: bool = typer.Option(False, "--contradictions", help="Check for contradictions"),
    suggest: bool = typer.Option(False, "--suggest", help="Suggest new concepts"),
):
    """Check wiki health and maintenance."""
    settings = get_settings()
    llm, obsidian, metadata = get_clients(settings)

    linter = WikiLinter(llm, obsidian, metadata)

    async def run():
        # Basic health check
        report = await linter.analyze_health()
        linter.print_report(report)

        # Check contradictions
        if contradictions:
            console.print("\n[blue]Analyzing for contradictions...[/]")
            issues = await linter.find_contradictions()
            if issues:
                console.print(f"[yellow]Found {len(issues)} potential contradictions:[/]")
                for issue in issues:
                    console.print(f"\n  [bold]{issue['concepts'][0]} vs {issue['concepts'][1]}[/]")
                    console.print(f"  Issue: {issue['issue']}")
                    console.print(f"  Severity: {issue['severity']}")
            else:
                console.print("[green]No contradictions found.[/]")

        # Suggest new concepts
        if suggest:
            console.print("\n[blue]Generating concept suggestions...[/]")
            suggestions = await linter.suggest_new_concepts()
            if suggestions:
                console.print("[green]Suggested new concepts:[/]")
                for s in suggestions:
                    console.print(f"\n  [bold]{s['name']}[/]")
                    console.print(f"  Reason: {s['reason']}")
                    console.print(f"  Connects: {', '.join(s['connects'])}")
            else:
                console.print("[dim]No new concept suggestions.[/]")

    asyncio.run(run())


@app.command()
def stats():
    """Show knowledge base statistics."""
    settings = get_settings()
    settings.ensure_directories()

    _, _, metadata = get_clients(settings)

    meta_stats = metadata.get_stats()

    table = Table(title="Knowledge Base Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Last Compile", meta_stats.get("last_compile", "Never"))
    table.add_row("Files Tracked", str(meta_stats.get("total_files_tracked", 0)))
    table.add_row("Total Concepts", str(meta_stats.get("total_concepts", 0)))

    # Count files in raw directory
    raw_files = list(settings.raw_dir.rglob("*"))
    raw_files = [f for f in raw_files if f.is_file() and not f.name.startswith(".")]
    table.add_row("Raw Files", str(len(raw_files)))

    console.print(table)


if __name__ == "__main__":
    app()
