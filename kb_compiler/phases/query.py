"""Phase 3: Query - Answer questions based on compiled wiki."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

from kb_compiler.core.llm import QUERIER_SYSTEM_PROMPT, KimiClient
from kb_compiler.core.obsidian import ObsidianClient

console = Console()


@dataclass
class QueryResult:
    """Result of a knowledge query."""

    answer: str
    sources: list[str]  # List of concept names used
    confidence: str  # high/medium/low
    suggestions: list[str]  # Related concepts to explore


class ContextRetriever:
    """Retrieve relevant context from wiki for queries."""

    def __init__(self, obsidian_client: ObsidianClient, wiki_dir: Path):
        self.obsidian = obsidian_client
        self.wiki_dir = wiki_dir

    def extract_wiki_links(self, content: str) -> list[str]:
        """Extract all wiki-links from content."""
        return re.findall(r"\[\[([^\]]+)\]\]", content)

    def get_concept_content(self, concept_name: str) -> Optional[str]:
        """Get content of a concept article."""
        # Try direct path
        paths_to_try = [
            f"concepts/{concept_name}.md",
            f"concepts/{concept_name.replace(' ', '_')}.md",
            f"{concept_name}.md",
        ]

        for path in paths_to_try:
            try:
                metadata, content = self.obsidian.read_note(path)
                return content
            except FileNotFoundError:
                continue

        return None

    def find_relevant_concepts(
        self,
        query: str,
        max_concepts: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Find concepts relevant to the query.

        Returns list of (concept_name, content, relevance_score)
        """
        # Get all concepts from INDEX.md
        try:
            _, index_content = self.obsidian.read_note("INDEX.md")
        except FileNotFoundError:
            console.print("[red]INDEX.md not found. Run compile first.[/]")
            return []

        # Extract concept names from index
        all_concepts = re.findall(r"- \[\[([^\]]+)\]\]", index_content)

        # Simple relevance scoring based on keyword overlap
        query_terms = set(query.lower().split())
        scored_concepts = []

        for concept in all_concepts:
            concept_terms = set(concept.lower().split())
            overlap = len(query_terms & concept_terms)
            score = overlap / max(len(query_terms), len(concept_terms)) if concept_terms else 0

            if score > 0:
                content = self.get_concept_content(concept)
                if content:
                    # Boost score based on content matches
                    content_lower = content.lower()
                    content_matches = sum(1 for term in query_terms if term in content_lower)
                    score += content_matches * 0.1
                    scored_concepts.append((concept, content, score))

        # Sort by score and return top N
        scored_concepts.sort(key=lambda x: x[2], reverse=True)
        return scored_concepts[:max_concepts]

    def expand_context(
        self,
        concepts: list[tuple[str, str, float]],
        depth: int = 1,
    ) -> dict[str, str]:
        """Expand context by including related concepts.

        Args:
            concepts: Initial concept list
            depth: How many levels of related concepts to include

        Returns:
            Dict mapping concept names to their content
        """
        context = {name: content for name, content, _ in concepts}

        if depth > 0:
            for name, content, _ in concepts:
                related = self.extract_wiki_links(content)
                for related_name in related[:3]:  # Limit related concepts
                    if related_name not in context:
                        related_content = self.get_concept_content(related_name)
                        if related_content:
                            context[related_name] = related_content

        return context


class QueryEngine:
    """Answer questions using compiled wiki knowledge."""

    def __init__(
        self,
        llm_client: KimiClient,
        obsidian_client: ObsidianClient,
        wiki_dir: Path,
    ):
        self.llm = llm_client
        self.obsidian = obsidian_client
        self.retriever = ContextRetriever(obsidian_client, wiki_dir)

    async def query(
        self,
        question: str,
        context_depth: int = 1,
        max_context_length: int = 8000,
    ) -> QueryResult:
        """Answer a question based on wiki knowledge."""
        # Find relevant concepts
        relevant = self.retriever.find_relevant_concepts(question)

        if not relevant:
            return QueryResult(
                answer="No relevant concepts found in the knowledge base. Try running `kb-compiler compile` first.",
                sources=[],
                confidence="low",
                suggestions=[],
            )

        # Expand context
        context = self.retriever.expand_context(relevant, depth=context_depth)

        # Format context for LLM
        context_parts = []
        total_length = 0

        for name, content in context.items():
            part = f"=== {name} ===\n{content[:1500]}\n\n"
            if total_length + len(part) > max_context_length:
                break
            context_parts.append(part)
            total_length += len(part)

        context_str = "".join(context_parts)

        # Build query prompt
        prompt = f"""Based on the following wiki content, answer the question:

{context_str}

Question: {question}

Provide a comprehensive answer that:
1. Directly addresses the question
2. Cites relevant concepts using [[Concept Name]] format
3. Notes any contradictions or uncertainties
4. Suggests related concepts to explore

Answer:"""

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=QUERIER_SYSTEM_PROMPT,
            temperature=0.3,
        )

        # Extract sources from answer
        used_sources = re.findall(r"\[\[([^\]]+)\]\]", response.content)
        used_sources = list(set(used_sources))

        # Determine confidence
        confidence = "high" if len(used_sources) >= 2 else "medium" if used_sources else "low"

        # Generate suggestions
        all_related = set()
        for name in used_sources:
            content = context.get(name, "")
            all_related.update(self.retriever.extract_wiki_links(content))

        suggestions = list(all_related - set(used_sources))[:5]

        return QueryResult(
            answer=response.content,
            sources=used_sources,
            confidence=confidence,
            suggestions=suggestions,
        )

    async def compare(
        self,
        concept1: str,
        concept2: str,
    ) -> QueryResult:
        """Compare two concepts."""
        content1 = self.retriever.get_concept_content(concept1)
        content2 = self.retriever.get_concept_content(concept2)

        if not content1 or not content2:
            missing = []
            if not content1:
                missing.append(concept1)
            if not content2:
                missing.append(concept2)
            return QueryResult(
                answer=f"Concepts not found: {', '.join(missing)}",
                sources=[],
                confidence="low",
                suggestions=[],
            )

        prompt = f"""Compare these two concepts:

=== {concept1} ===
{content1[:2000]}

=== {concept2} ===
{content2[:2000]}

Provide a comparison covering:
1. Key similarities
2. Key differences
3. How they relate to each other
4. When to use/reference each

Comparison:"""

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=QUERIER_SYSTEM_PROMPT,
            temperature=0.3,
        )

        return QueryResult(
            answer=response.content,
            sources=[concept1, concept2],
            confidence="high",
            suggestions=[],
        )

    async def explore(
        self,
        concept: str,
        depth: int = 2,
    ) -> QueryResult:
        """Explore a concept and its connections."""
        content = self.retriever.get_concept_content(concept)

        if not content:
            # Try to find similar concepts
            all_concepts = self.retriever.find_relevant_concepts(concept, max_concepts=10)
            suggestions = [c[0] for c in all_concepts if c[0].lower() != concept.lower()]

            return QueryResult(
                answer=f"Concept '{concept}' not found.",
                sources=[],
                confidence="low",
                suggestions=suggestions[:5],
            )

        # Get connected concepts
        links = self.retriever.extract_wiki_links(content)
        connected = {}

        for link in links[:depth * 3]:
            link_content = self.retriever.get_concept_content(link)
            if link_content:
                connected[link] = link_content[:1000]

        # Build exploration view
        connected_parts = []
        for name, txt in connected.items():
            connected_parts.append(f"=== {name} ===\n{txt[:500]}")
        connected_text = "\n".join(connected_parts)

        prompt = f"""Explore this concept and its connections:

=== {concept} ===
{content[:2000]}

Connected concepts:
{connected_text}

Provide:
1. Core definition and importance
2. Key connections and how they relate
3. Broader context within the knowledge base
4. Suggested paths for further exploration

Exploration:"""

        response = await self.llm.complete(
            prompt=prompt,
            system_prompt=QUERIER_SYSTEM_PROMPT,
            temperature=0.4,
        )

        return QueryResult(
            answer=response.content,
            sources=[concept] + list(connected.keys()),
            confidence="high",
            suggestions=list(connected.keys())[5:],
        )


class QueryOutputFormatter:
    """Format query results in various output formats."""

    @staticmethod
    def format_markdown(result: QueryResult) -> str:
        """Format as markdown for saving."""
        lines = [
            f"# Query Result",
            "",
            f"**Confidence:** {result.confidence}",
            f"**Sources:** {', '.join(f'[[{s}]]' for s in result.sources)}",
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

        return "\n".join(lines)

    @staticmethod
    def format_terminal(result: QueryResult) -> None:
        """Print formatted result to terminal."""
        console.print()
        console.print("─" * 60)
        console.print(f"[bold cyan]Answer[/bold cyan] [dim](confidence: {result.confidence})[/dim]")
        console.print("─" * 60)
        console.print(RichMarkdown(result.answer))
        console.print("─" * 60)

        if result.sources:
            console.print(f"[dim]Sources: {', '.join(f'[[{s}]]' for s in result.sources)}[/dim]")

        if result.suggestions:
            console.print(f"[dim]Related: {', '.join(f'[[{s}]]' for s in result.suggestions)}[/dim]")

        console.print()
