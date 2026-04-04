"""Phase 4-5: Feedback and Maintenance - Query feedback and wiki health."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from kb_compiler.core.llm import LINTER_SYSTEM_PROMPT, KimiClient
from kb_compiler.core.metadata import MetadataTracker
from kb_compiler.core.obsidian import ObsidianClient
from kb_compiler.phases.query import QueryResult

console = Console()


class FeedbackManager:
    """Manage query result feedback into knowledge base."""

    def __init__(self, output_dir: Path, obsidian_client: ObsidianClient):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.obsidian = obsidian_client

    def save_query_result(
        self,
        query: str,
        result: QueryResult,
        format: str = "markdown",
    ) -> Path:
        """Save a query result to output directory.

        Args:
            query: Original query string
            result: Query result
            format: Output format (markdown, json)

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:50])

        if format == "json":
            filename = f"query_{timestamp}_{safe_query}.json"
            filepath = self.output_dir / filename

            data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "result": asdict(result),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        else:  # markdown
            filename = f"query_{timestamp}_{safe_query}.md"
            filepath = self.output_dir / filename

            lines = [
                "---",
                f"query: {query}",
                f"timestamp: {datetime.now().isoformat()}",
                f"confidence: {result.confidence}",
                f"sources: {json.dumps(result.sources)}",
                "---",
                "",
                f"# Query: {query}",
                "",
                "## Answer",
                "",
                result.answer,
                "",
            ]

            if result.suggestions:
                lines.extend([
                    "## Related Concepts",
                    "",
                ])
                for suggestion in result.suggestions:
                    lines.append(f"- [[{suggestion}]]")
                lines.append("")

            filepath.write_text("\n".join(lines), encoding="utf-8")

        console.print(f"[green]Saved query result to: {filepath}[/]")
        return filepath

    def save_as_concept(
        self,
        concept_name: str,
        content: str,
        sources: list[str],
    ) -> Path:
        """Save query result as a new concept article.

        This allows deep research to be integrated back into the wiki.
        """
        from kb_compiler.phases.compile import WikiCompiler

        filename = WikiCompiler._sanitize_filename(concept_name) + ".md"
        filepath = f"concepts/{filename}"

        lines = [
            "---",
            f"title: {concept_name}",
            f"sources: {json.dumps(sources)}",
            f"derived_from: query",
            f"created_at: {datetime.now().isoformat()}",
            "---",
            "",
            f"# {concept_name}",
            "",
            "*This article was generated from a deep query and may need review.*",
            "",
            content,
        ]

        self.obsidian.write_note(filepath, "\n".join(lines))
        console.print(f"[green]Created concept from query: {filepath}[/]")

        return Path(filepath)


class WikiLinter:
    """Lint and maintain wiki health."""

    def __init__(
        self,
        llm_client: KimiClient,
        obsidian_client: ObsidianClient,
        metadata_tracker: MetadataTracker,
    ):
        self.llm = llm_client
        self.obsidian = obsidian_client
        self.metadata = metadata_tracker

    async def analyze_health(self) -> dict:
        """Analyze wiki health and return report."""
        console.print("[blue]Analyzing wiki health...[/]")

        issues = {
            "contradictions": [],
            "isolated": [],
            "missing_refs": [],
            "orphaned": [],
        }

        # Get all concepts
        all_concepts = self.metadata.list_all_concepts()
        if not all_concepts:
            return {"status": "empty", "issues": issues}

        # Check for isolated concepts (no backlinks)
        for concept in all_concepts:
            backlinks = self.obsidian.get_backlinks(concept)
            if not backlinks:
                issues["isolated"].append(concept)

        # Check for orphaned files (no metadata)
        try:
            all_notes = self.obsidian.list_notes("concepts")
            tracked = set()
            for note in all_notes:
                # Extract concept name from path
                name = Path(note).stem.replace("_", " ")
                tracked.add(name)

            # Find files not in metadata
            for note in all_notes:
                name = Path(note).stem.replace("_", " ")
                if name not in all_concepts:
                    issues["orphaned"].append(str(note))
        except Exception as e:
            console.print(f"[yellow]Could not check orphaned files: {e}[/]")

        return {
            "status": "ok",
            "total_concepts": len(all_concepts),
            "issues": issues,
        }

    async def find_contradictions(self, concepts: Optional[list[str]] = None) -> list[dict]:
        """Use LLM to find contradictions between concept articles."""
        if concepts is None:
            concepts = self.metadata.list_all_concepts()[:10]  # Limit for API

        if len(concepts) < 2:
            return []

        # Load concept contents
        concept_data = []
        for name in concepts:
            # Try to read concept file
            for filename in [f"concepts/{name}.md", f"concepts/{name.replace(' ', '_')}.md"]:
                try:
                    _, content = self.obsidian.read_note(filename)
                    concept_data.append(f"=== {name} ===\n{content[:1500]}")
                    break
                except FileNotFoundError:
                    continue

        if len(concept_data) < 2:
            return []

        prompt = f"""Analyze these concept articles for contradictions or inconsistencies:

{chr(10).join(concept_data)}

Identify any:
1. Direct contradictions (A says X, B says not-X)
2. Inconsistencies in data/numbers
3. Conflicting definitions
4. Temporal inconsistencies

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Issue descriptions must be in Chinese

Output as JSON array:
[
  {{
    "concepts": ["Concept A", "Concept B"],
    "issue": "Description of contradiction",
    "severity": "high/medium/low"
  }}
]

If no contradictions found, return empty array []."""

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=LINTER_SYSTEM_PROMPT,
            temperature=0.3,
        )

        try:
            # Extract JSON - more robust parsing
            content = response.content.strip()

            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            elif "```json" in content:
                match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            elif "```" in content:
                match = re.search(r"```\s*\n(.*?)\n```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            content = content.strip()
            if not content.startswith('['):
                start_idx = content.find('[')
                if start_idx != -1:
                    content = content[start_idx:]

            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            console.print(f"[yellow]Could not parse contradiction analysis: {e}[/]")
            return []

    async def suggest_new_concepts(self) -> list[dict]:
        """Suggest new concepts based on existing content."""
        concepts = self.metadata.list_all_concepts()

        if len(concepts) < 3:
            return []

        # Sample some concept contents
        sample_data = []
        for name in concepts[:10]:
            for filename in [f"concepts/{name}.md", f"concepts/{name.replace(' ', '_')}.md"]:
                try:
                    _, content = self.obsidian.read_note(filename)
                    sample_data.append(f"=== {name} ===\n{content[:1000]}")
                    break
                except FileNotFoundError:
                    continue

        prompt = f"""Based on these existing concepts, suggest new concepts that should be created:

{chr(10).join(sample_data)}

Suggest 3-5 new concepts that:
1. Are mentioned but not yet full articles
2. Would bridge gaps between existing concepts
3. Would provide important context

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Concept names should be in Chinese
- Reason descriptions must be in Chinese

Output as JSON array:
[
  {{
    "name": "Concept Name",
    "reason": "Why this concept should exist",
    "connects": ["Existing Concept A", "Existing Concept B"]
  }}
]"""

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=LINTER_SYSTEM_PROMPT,
            temperature=0.4,
        )

        try:
            content = response.content.strip()

            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            elif "```json" in content:
                match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            elif "```" in content:
                match = re.search(r"```\s*\n(.*?)\n```", content, re.DOTALL)
                if match:
                    content = match.group(1)

            content = content.strip()
            if not content.startswith('['):
                start_idx = content.find('[')
                if start_idx != -1:
                    content = content[start_idx:]

            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            console.print(f"[yellow]Could not parse concept suggestions: {e}[/]")
            return []

    def print_report(self, report: dict) -> None:
        """Print health report to console."""
        console.print()
        console.print("[bold]Wiki Health Report[/bold]")
        console.print("─" * 50)

        if report.get("status") == "empty":
            console.print("[yellow]Wiki is empty. Run compile first.[/]")
            return

        issues = report.get("issues", {})

        # Summary table
        table = Table(title="Issue Summary")
        table.add_column("Issue Type", style="cyan")
        table.add_column("Count", justify="right")

        for issue_type, items in issues.items():
            count = len(items)
            color = "green" if count == 0 else "yellow" if count < 5 else "red"
            table.add_row(issue_type, f"[{color}]{count}[/{color}]")

        console.print(table)

        # Detailed issues
        if issues.get("isolated"):
            console.print("\n[bold]Isolated Concepts (no incoming links):[/]")
            for concept in issues["isolated"][:10]:
                console.print(f"  - {concept}")
            if len(issues["isolated"]) > 10:
                console.print(f"  ... and {len(issues['isolated']) - 10} more")

        if issues.get("orphaned"):
            console.print("\n[bold]Orphaned Files (not in metadata):[/]")
            for file in issues["orphaned"][:5]:
                console.print(f"  - {file}")
            if len(issues["orphaned"]) > 5:
                console.print(f"  ... and {len(issues['orphaned']) - 5} more")

        console.print()
