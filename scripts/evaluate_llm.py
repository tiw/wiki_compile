#!/usr/bin/env python3
"""
LLM Concept Extraction Evaluation Script

Compares local LLM (MLX) against Kimi golden dataset.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from kb_compiler.config import Settings
from kb_compiler.core.llm import create_llm_client
from kb_compiler.phases.compile import ConceptExtractor

console = Console()


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single document."""
    file_path: str
    golden_concepts: list[str]
    local_concepts: list[str]
    true_positives: list[str]  # Correctly extracted
    false_positives: list[str]  # Extra concepts
    false_negatives: list[str]  # Missed concepts
    precision: float
    recall: float
    f1_score: float


class LLMEvaluator:
    """Evaluate local LLM against golden dataset."""

    def __init__(self, golden_dataset_path: str, raw_dir: str):
        self.golden_dataset_path = golden_dataset_path
        self.raw_dir = Path(raw_dir)
        self.golden_data = self._load_golden_dataset()

    def _load_golden_dataset(self) -> dict:
        """Load golden concepts from compile_state.json."""
        with open(self.golden_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract file -> concepts mapping
        # Store both relative path and basename for flexible lookup
        golden_concepts = {}
        for file_path, meta in data.get("files", {}).items():
            # Get relative path from KB root (e.g., "AI研发转型研究/04-六象限演进与技术设计.md")
            path_obj = Path(file_path)
            try:
                # Try to get path relative to raw/ directory
                rel_path = path_obj.relative_to(Path("/Users/ting/KnowledgeBase/raw"))
            except ValueError:
                # If not under raw/, use the filename only
                rel_path = path_obj.name
            golden_concepts[str(rel_path)] = meta.get("concepts_extracted", [])

        return golden_concepts

    def _normalize_concept(self, name: str) -> str:
        """Normalize concept name for comparison."""
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    def _calculate_metrics(
        self,
        file_path: str,
        golden: list[str],
        local: list[str]
    ) -> EvaluationResult:
        """Calculate precision, recall, F1 for a document."""
        # Normalize for comparison
        golden_norm = {self._normalize_concept(c): c for c in golden}
        local_norm = {self._normalize_concept(c): c for c in local}

        # Find matches
        matched_norm = set(golden_norm.keys()) & set(local_norm.keys())
        true_positives = [local_norm[n] for n in matched_norm]

        # False positives (local has, golden doesn't)
        fp_norm = set(local_norm.keys()) - set(golden_norm.keys())
        false_positives = [local_norm[n] for n in fp_norm]

        # False negatives (golden has, local doesn't)
        fn_norm = set(golden_norm.keys()) - set(local_norm.keys())
        false_negatives = [golden_norm[n] for n in fn_norm]

        # Calculate metrics
        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return EvaluationResult(
            file_path=file_path,
            golden_concepts=golden,
            local_concepts=local,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1
        )

    async def evaluate_with_local_llm(
        self,
        max_files: Optional[int] = None,
        batch_size: int = 2
    ) -> list[EvaluationResult]:
        """Run local LLM on documents and compare with golden dataset."""
        # Setup local LLM client
        local_client = create_llm_client(
            api_key=os.environ.get("KB_LOCAL_LLM_API_KEY", "dandan"),
            base_url=os.environ.get("KB_LOCAL_LLM_BASE_URL", "http://127.0.0.1:8017/v1"),
            model=os.environ.get("KB_LOCAL_LLM_MODEL", "Qwen3___5-27B-Claude-4___6-Opus-Distilled-MLX-6bit"),
            provider="local"
        )

        extractor = ConceptExtractor(local_client)
        results = []

        # Get files to evaluate
        files = list(self.golden_data.keys())
        if max_files:
            files = files[:max_files]

        console.print(f"[blue]Evaluating {len(files)} documents with local LLM...[/]")

        for i, filename in enumerate(files):
            # Handle both flat and nested paths
            file_path = self.raw_dir / filename
            if not file_path.exists():
                # Try with just the basename
                file_path = self.raw_dir / Path(filename).name
            if not file_path.exists():
                console.print(f"[yellow]Skipping {filename}: file not found[/]")
                continue

            # Read document
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get golden concepts
            golden = self.golden_data[filename]

            console.print(f"\n[dim]Processing {i+1}/{len(files)}: {filename}[/]")
            console.print(f"  Golden concepts: {len(golden)}")

            # Extract with local LLM
            try:
                documents = [(filename, content)]
                local_concepts = await extractor.extract_concepts(documents)
                local_concept_names = [c.name for c in local_concepts]

                console.print(f"  Local concepts: {len(local_concept_names)}")

                # Calculate metrics
                result = self._calculate_metrics(
                    filename, golden, local_concept_names
                )
                results.append(result)

                console.print(f"  Precision: {result.precision:.2%}")
                console.print(f"  Recall: {result.recall:.2%}")
                console.print(f"  F1: {result.f1_score:.2%}")

            except Exception as e:
                console.print(f"[red]Error processing {filename}: {e}[/]")
                continue

        return results

    def print_summary(self, results: list[EvaluationResult]):
        """Print evaluation summary."""
        if not results:
            console.print("[red]No evaluation results to display[/]")
            return

        # Overall metrics
        total_golden = sum(len(r.golden_concepts) for r in results)
        total_local = sum(len(r.local_concepts) for r in results)
        total_tp = sum(len(r.true_positives) for r in results)
        total_fp = sum(len(r.false_positives) for r in results)
        total_fn = sum(len(r.false_negatives) for r in results)

        avg_precision = sum(r.precision for r in results) / len(results)
        avg_recall = sum(r.recall for r in results) / len(results)
        avg_f1 = sum(r.f1_score for r in results) / len(results)

        # Create summary table
        table = Table(title="LLM Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Documents Evaluated", str(len(results)))
        table.add_row("", "")
        table.add_row("Total Golden Concepts", str(total_golden))
        table.add_row("Total Local Concepts", str(total_local))
        table.add_row("", "")
        table.add_row("True Positives", str(total_tp))
        table.add_row("False Positives", str(total_fp))
        table.add_row("False Negatives", str(total_fn))
        table.add_row("", "")
        table.add_row("Avg Precision", f"{avg_precision:.2%}")
        table.add_row("Avg Recall", f"{avg_recall:.2%}")
        table.add_row("Avg F1 Score", f"{avg_f1:.2%}")

        console.print()
        console.print(table)

        # Per-document breakdown
        doc_table = Table(title="Per-Document Results")
        doc_table.add_column("File", style="cyan")
        doc_table.add_column("Golden", justify="right")
        doc_table.add_column("Local", justify="right")
        doc_table.add_column("Precision", justify="right")
        doc_table.add_column("Recall", justify="right")
        doc_table.add_column("F1", justify="right")

        for r in results:
            doc_table.add_row(
                r.file_path[:40] + "..." if len(r.file_path) > 40 else r.file_path,
                str(len(r.golden_concepts)),
                str(len(r.local_concepts)),
                f"{r.precision:.0%}",
                f"{r.recall:.0%}",
                f"{r.f1_score:.0%}"
            )

        console.print()
        console.print(doc_table)

        # Show examples
        console.print("\n[bold]Sample Analysis:[/]")
        for r in results[:3]:  # Show first 3 documents
            console.print(f"\n[cyan]{r.file_path}:[/]")
            if r.false_positives[:3]:
                console.print(f"  [yellow]Extra (FP): {', '.join(r.false_positives[:3])}[/]")
            if r.false_negatives[:3]:
                console.print(f"  [red]Missed (FN): {', '.join(r.false_negatives[:3])}[/]")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate local LLM against Kimi golden dataset")
    parser.add_argument("--golden", default="/Users/ting/KnowledgeBase/_meta/compile_state.json",
                        help="Path to golden dataset (compile_state.json)")
    parser.add_argument("--raw-dir", default="/Users/ting/KnowledgeBase/raw",
                        help="Path to raw documents directory")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to evaluate")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Number of documents per batch")

    args = parser.parse_args()

    # Verify local LLM is available
    console.print("[blue]Checking local LLM availability...[/]")
    import httpx
    try:
        response = httpx.get(
            f"{os.environ.get('KB_LOCAL_LLM_BASE_URL', 'http://127.0.0.1:8017/v1')}/models",
            headers={"Authorization": f"Bearer {os.environ.get('KB_LOCAL_LLM_API_KEY', 'dandan')}"},
            timeout=5.0
        )
        if response.status_code == 200:
            console.print("[green]✓ Local LLM is running[/]")
        else:
            console.print(f"[red]✗ Local LLM returned status {response.status_code}[/]")
            return
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to local LLM: {e}[/]")
        return

    # Run evaluation
    evaluator = LLMEvaluator(args.golden, args.raw_dir)
    results = await evaluator.evaluate_with_local_llm(
        max_files=args.max_files,
        batch_size=args.batch_size
    )

    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
