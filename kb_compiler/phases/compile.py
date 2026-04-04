"""Phase 2: Compile - Transform raw documents into structured wiki."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import frontmatter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from kb_compiler.core.llm import COMPILER_SYSTEM_PROMPT, KimiClient
from kb_compiler.core.metadata import MetadataTracker
from kb_compiler.core.obsidian import ObsidianClient

console = Console()


@dataclass
class Concept:
    """A concept extracted from documents."""

    name: str
    summary: str
    key_facts: list[str]
    sources: list[str]
    related: list[str]
    contradictions: list[str]
    open_questions: list[str]


class ConceptExtractor:
    """Extract concepts from raw documents using LLM."""

    CONCEPT_EXTRACTION_PROMPT = """Analyze the following documents and extract key concepts.

Documents:
{documents}

For each key concept identified, provide:
1. Concept name (short, clear, unique - max 5 words)
2. One-sentence summary (max 200 chars)
3. Key facts with specific data/numbers (max 3 items, each max 150 chars)
4. Source files where this concept appears
5. Related concepts that should link to this one (max 5 items)

IMPORTANT JSON FORMATTING RULES:
- Output MUST be valid JSON
- NO newlines inside string values - use \\n for line breaks
- NO trailing commas
- All strings must be on single lines
- Escape quotes with \\"

Output as JSON array:
[
  {{
    "name": "Concept Name",
    "summary": "One sentence definition",
    "key_facts": ["Fact 1 with data", "Fact 2"],
    "sources": ["source_file.md"],
    "related": ["Related Concept 1", "Related Concept 2"],
    "contradictions": ["Source A says X but Source B says Y"],
    "open_questions": ["What about Z?"]
  }}
]

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- Concept names should be in Chinese
- Summaries must be in Chinese
- All JSON string values must be in Chinese

Rules:
- Extract **5-12 concepts** per batch (aim for quality over quantity)
- Include core concepts, methodologies, frameworks, and metaphors mentioned
- Include specific techniques, models, and analogies
- Use clear, searchable Chinese concept names
- Preserve specific numbers, dates, and quotes
- Note when sources disagree
- Identify genuinely related concepts, not just similar words
- Keep all text concise to avoid JSON parsing errors

SELF-CHECK: After listing concepts, review if you missed any important terms, frameworks, or metaphors from the source documents. If so, add them."""

    def __init__(self, llm_client: KimiClient):
        self.llm = llm_client

    async def extract_concepts(
        self,
        documents: list[tuple[str, str]],  # (path, content)
        existing_concepts: Optional[list[str]] = None,
    ) -> list[Concept]:
        """Extract concepts from documents using per-document extraction for better coverage."""
        all_concepts = []
        seen_names = set()

        # Process documents in batches to avoid overwhelming the API
        # Smaller batches = more API calls but more reliable responses
        batch_size = 2
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Extract concepts from this batch
            batch_concepts = await self._extract_from_batch(batch, existing_concepts)

            # Merge and deduplicate
            for concept in batch_concepts:
                # Normalize name for deduplication
                normalized_name = concept.name.lower().replace(" ", "").replace("_", "")
                if normalized_name not in seen_names:
                    all_concepts.append(concept)
                    seen_names.add(normalized_name)

            console.print(f"[dim]Extracted {len(batch_concepts)} concepts from batch {i//batch_size + 1}...[/]")

        console.print(f"[green]Total unique concepts extracted: {len(all_concepts)}[/]")
        return all_concepts

    async def _extract_from_batch(
        self,
        documents: list[tuple[str, str]],
        existing_concepts: Optional[list[str]] = None,
    ) -> list[Concept]:
        """Extract concepts from a batch of documents."""
        doc_texts = []
        for path, content in documents:
            # Strip frontmatter for LLM
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
            # Increase content limit for better coverage
            doc_texts.append(f"=== {path} ===\n{content[:8000]}\n")

        doc_section = "\n\n".join(doc_texts)

        # Add existing concepts to prompt to avoid duplicates
        existing_section = ""
        if existing_concepts:
            existing_section = f"\n\nExisting concepts (do not duplicate these): {', '.join(existing_concepts[:20])}"

        prompt = self.CONCEPT_EXTRACTION_PROMPT.format(
            documents=doc_section
        ) + existing_section

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=COMPILER_SYSTEM_PROMPT,
            temperature=0.4,  # Slightly higher for more creative extraction
        )

        # Parse JSON response
        try:
            # Try to extract JSON from markdown code block
            content = response.content.strip()

            # Find JSON array in the response - handle nested brackets
            # Look for the outermost array
            if "```json" in content:
                json_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r"```\s*\n(.*?)\n```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)

            content = content.strip()

            # Clean up common JSON formatting issues
            # Remove trailing commas before closing brackets
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            # Fix escaped newlines in strings
            content = content.replace('\\n', '\n')
            # Remove control characters
            content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)

            if not content.startswith('['):
                # Try to find array start
                start_idx = content.find('[')
                if start_idx != -1:
                    content = content[start_idx:]

            # Find matching closing bracket
            if content.startswith('['):
                depth = 0
                end_idx = 0
                for i, char in enumerate(content):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                content = content[:end_idx]

            concepts_data = json.loads(content)
            concepts = []

            for data in concepts_data:
                concepts.append(
                    Concept(
                        name=data["name"],
                        summary=data.get("summary", ""),
                        key_facts=data.get("key_facts", []),
                        sources=data.get("sources", []),
                        related=data.get("related", []),
                        contradictions=data.get("contradictions", []),
                        open_questions=data.get("open_questions", []),
                    )
                )

            return concepts

        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Failed to parse concepts, attempting recovery: {e}[/]")

            # Try to fix incomplete JSON by adding missing closing brackets
            try:
                # Count opening and closing brackets
                open_brackets = content.count('[') + content.count('{')
                close_brackets = content.count(']') + content.count('}')

                # Add missing closing brackets
                while open_brackets > close_brackets:
                    if content.rstrip()[-1:] in ['}', ']']:
                        content = content.rstrip() + ']'
                    else:
                        content = content.rstrip() + '}]'
                    close_brackets += 1

                # Try parsing again
                concepts_data = json.loads(content)
                concepts = []
                for data in concepts_data:
                    concepts.append(
                        Concept(
                            name=data["name"],
                            summary=data.get("summary", ""),
                            key_facts=data.get("key_facts", []),
                            sources=data.get("sources", []),
                            related=data.get("related", []),
                            contradictions=data.get("contradictions", []),
                            open_questions=data.get("open_questions", []),
                        )
                    )
                console.print(f"[green]Recovered {len(concepts)} concepts from incomplete JSON[/]")
                return concepts

            except Exception as recovery_error:
                console.print(f"[red]Recovery failed: {recovery_error}[/]")
                # Save full response to file for debugging
                debug_file = "/tmp/kb_compiler_debug.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(response.content)
                console.print(f"[red]Full response saved to: {debug_file}[/]")
                console.print(f"[red]Response length: {len(response.content)} chars[/]")
                return []


class WikiCompiler:
    """Compile concepts into wiki articles."""

    WIKI_PROMPT = """Create a comprehensive wiki article for the concept: {concept_name}

Based on these sources:
{sources}

Existing related concepts in wiki:
{related_concepts}

Create a markdown article with this structure:

---
title: {concept_name}
sources: {sources_list}
related: {related_links}
last_compiled: {date}
---

## Summary
{summary}

## Key Facts
- Fact 1 (with specific data/numbers)
- Fact 2

## Detailed Explanation
Comprehensive explanation based on sources.

## Source Details
### Source A
What this source says about the concept.

### Source B
What this source says (note any differences).

## Contradictions
- If sources disagree, explain the different positions

## Open Questions
- What remains unanswered

## Related Concepts
See also: [[Related Concept 1]], [[Related Concept 2]]

CRITICAL RULES:
- **ALL output must be in Chinese (中文)**
- The entire article content should be written in Chinese
- Use Chinese concept names for wiki-links

Rules:
- Use wiki-links [[Concept Name]] for all concept references
- Preserve specific data and numbers
- Note source disagreements explicitly
- Write for future you (comprehensive but concise)"""

    def __init__(
        self,
        llm_client: KimiClient,
        obsidian_client: ObsidianClient,
        metadata_tracker: MetadataTracker,
    ):
        self.llm = llm_client
        self.obsidian = obsidian_client
        self.metadata = metadata_tracker

    async def compile_concept(
        self,
        concept: Concept,
        source_contents: dict[str, str],  # path -> content
        all_concepts: list[str],
    ) -> str:
        """Compile a single concept into wiki article."""
        from datetime import datetime

        # Format sources for prompt
        sources_text = []
        for source in concept.sources:
            content = source_contents.get(source, "")
            # Strip frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
            sources_text.append(f"=== {source} ===\n{content[:3000]}\n")

        sources_section = "\n\n".join(sources_text)

        # Format related concepts as wiki-links
        related_links = " ".join(f"[[{c}]]" for c in concept.related if c in all_concepts)

        prompt = self.WIKI_PROMPT.format(
            concept_name=concept.name,
            sources=sources_section,
            related_concepts=", ".join(all_concepts),
            sources_list=json.dumps(concept.sources),
            related_links=related_links,
            date=datetime.now().isoformat(),
            summary=concept.summary,
        )

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=COMPILER_SYSTEM_PROMPT,
            temperature=0.3,
        )

        return response.content

    async def compile_all(
        self,
        concepts: list[Concept],
        source_contents: dict[str, str],
    ) -> list[Path]:
        """Compile all concepts into wiki articles."""
        from datetime import datetime

        all_concept_names = [c.name for c in concepts]
        created_files = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Compiling concepts...", total=len(concepts))

            for concept in concepts:
                progress.update(task, description=f"Compiling: {concept.name}")

                # Compile article
                article_content = await self.compile_concept(
                    concept, source_contents, all_concept_names
                )

                # Write to wiki
                filename = self._sanitize_filename(concept.name) + ".md"
                filepath = f"concepts/{filename}"

                self.obsidian.write_note(filepath, article_content)

                # Update metadata
                self.metadata.update_concept(
                    name=concept.name,
                    source_files=concept.sources,
                    related=concept.related,
                )

                created_files.append(Path(filepath))
                progress.advance(task)

        return created_files

    def update_index(self, concepts: list[Concept]) -> Path:
        """Update the wiki INDEX.md with all concepts."""
        from datetime import datetime

        # Group concepts alphabetically
        by_letter: dict[str, list[Concept]] = {}
        for c in concepts:
            first_letter = c.name[0].upper() if c.name else "#"
            by_letter.setdefault(first_letter, []).append(c)

        # Build index content
        lines = [
            "---",
            f"title: Knowledge Base Index",
            f"last_updated: {datetime.now().isoformat()}",
            f"total_concepts: {len(concepts)}",
            "---",
            "",
            "# Knowledge Base Index",
            "",
            "## Overview",
            f"This knowledge base contains **{len(concepts)}** compiled concepts.",
            "",
            "## All Concepts",
            "",
        ]

        for letter in sorted(by_letter.keys()):
            lines.append(f"### {letter}")
            lines.append("")
            for concept in sorted(by_letter[letter], key=lambda x: x.name):
                lines.append(f"- [[{concept.name}]] - {concept.summary[:100]}...")
            lines.append("")

        # Add recent updates section
        lines.extend([
            "## Recently Updated",
            "",
        ])
        recent = sorted(concepts, key=lambda x: x.sources[0] if x.sources else "", reverse=True)[:10]
        for concept in recent:
            lines.append(f"- [[{concept.name}]]")

        content = "\n".join(lines)

        # Write index
        self.obsidian.write_note("INDEX.md", content)
        console.print("[green]Updated wiki/INDEX.md[/]")

        return Path("INDEX.md")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Convert concept name to safe filename."""
        # Remove or replace unsafe characters
        safe = re.sub(r'[<>:"/\\|?*]', "_", name)
        safe = safe.strip(". ")
        return safe if safe else "unnamed_concept"


class CompilationPipeline:
    """Full compilation pipeline from raw docs to wiki."""

    def __init__(
        self,
        llm_client: KimiClient,
        obsidian_client: ObsidianClient,
        metadata_tracker: MetadataTracker,
        raw_dir: Path,
        wiki_dir: Path,
    ):
        self.llm = llm_client
        self.obsidian = obsidian_client
        self.metadata = metadata_tracker
        self.raw_dir = raw_dir
        self.wiki_dir = wiki_dir

        self.extractor = ConceptExtractor(llm_client)
        self.compiler = WikiCompiler(llm_client, obsidian_client, metadata_tracker)

    async def run(self, incremental: bool = True) -> dict:
        """Run the full compilation pipeline."""
        # Find changed files
        if incremental:
            changed_files = self.metadata.get_changed_files(self.raw_dir)
            if not changed_files:
                console.print("[yellow]No changed files to compile.[/]")
                return {"status": "no_changes", "concepts": []}

            console.print(f"[blue]Found {len(changed_files)} changed files to compile.[/]")
        else:
            changed_files = list(self.raw_dir.rglob("*.md"))
            changed_files = [f for f in changed_files if not f.name.startswith(".")]
            console.print(f"[blue]Full compile: processing {len(changed_files)} files.[/]")

        # Load file contents
        documents = []
        source_contents = {}

        for file_path in changed_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                rel_path = str(file_path.relative_to(self.raw_dir))
                documents.append((rel_path, content))
                source_contents[rel_path] = content
            except Exception as e:
                console.print(f"[yellow]Failed to read {file_path}: {e}[/]")

        if not documents:
            return {"status": "error", "reason": "no_readable_files"}

        # Extract concepts
        console.print("[blue]Extracting concepts from documents...[/]")
        existing_concepts = self.metadata.list_all_concepts() if incremental else None
        concepts = await self.extractor.extract_concepts(documents, existing_concepts)

        if not concepts:
            console.print("[yellow]No concepts extracted.[/]")
            return {"status": "no_concepts"}

        console.print(f"[green]Extracted {len(concepts)} concepts.[/]")

        # Compile to wiki
        console.print("[blue]Compiling concepts to wiki articles...[/]")
        created_files = await self.compiler.compile_all(concepts, source_contents)

        # Update index
        self.compiler.update_index(concepts)

        # Update file metadata
        for file_path in changed_files:
            file_concepts = [
                c.name for c in concepts if str(file_path.relative_to(self.raw_dir)) in c.sources
            ]
            self.metadata.update_file(file_path, file_concepts)

        self.metadata.update_last_compile()
        self.metadata.save()

        return {
            "status": "success",
            "files_processed": len(documents),
            "concepts": [c.name for c in concepts],
            "created_files": [str(f) for f in created_files],
        }
