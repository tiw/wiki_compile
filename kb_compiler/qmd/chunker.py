"""Chunker: Split wiki markdown into retrievable fragments."""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A retrievable text chunk from a concept article."""

    id: str  # source_path + section + index
    source_path: str
    concept_name: str
    section_header: str | None
    content: str
    token_count: int


class Chunker:
    """Split markdown content into semantic chunks."""

    DEFAULT_MAX_TOKENS = 512
    DEFAULT_OVERLAP_TOKENS = 128
    TOKENS_PER_CHAR = 0.6  # Rough Chinese/English mix estimate

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.max_chars = int(max_tokens / self.TOKENS_PER_CHAR)
        self.overlap_chars = int(overlap_tokens / self.TOKENS_PER_CHAR)

    def chunk(self, markdown_content: str, source_path: str, concept_name: str) -> list[Chunk]:
        """Split markdown into chunks by headers, then sliding window if too large."""
        sections = self._split_by_headers(markdown_content)
        chunks: list[Chunk] = []

        for section_idx, (header, content) in enumerate(sections):
            section_chars = int(self.max_tokens / self.TOKENS_PER_CHAR)
            if len(content) <= section_chars:
                chunk_id = f"{source_path}#{section_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        source_path=source_path,
                        concept_name=concept_name,
                        section_header=header or None,
                        content=content.strip(),
                        token_count=self._estimate_tokens(content),
                    )
                )
            else:
                # Sliding window for oversized sections
                sub_chunks = self._sliding_window(content)
                for sub_idx, sub_content in enumerate(sub_chunks):
                    chunk_id = f"{source_path}#{section_idx}-{sub_idx}"
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            source_path=source_path,
                            concept_name=concept_name,
                            section_header=header or None,
                            content=sub_content.strip(),
                            token_count=self._estimate_tokens(sub_content),
                        )
                    )

        return chunks

    def _split_by_headers(self, content: str) -> list[tuple[str | None, str]]:
        """Split markdown by ## and ### headers."""
        # Normalize line endings
        content = content.replace("\r\n", "\n")
        # Match ## or ### headers
        pattern = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
        matches = list(pattern.finditer(content))

        if not matches:
            return [(None, content)]

        sections: list[tuple[str | None, str]] = []
        # Content before first header
        first_start = matches[0].start()
        if first_start > 0:
            preamble = content[:first_start].strip()
            if preamble:
                sections.append((None, preamble))

        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            # Include header in content for context
            full_section = f"{match.group(0).strip()}\n\n{section_content}"
            sections.append((header, full_section))

        return sections

    def _sliding_window(self, content: str) -> list[str]:
        """Split long content into overlapping chunks by character count."""
        chunks: list[str] = []
        start = 0
        length = len(content)

        while start < length:
            end = min(start + self.max_chars, length)
            # Try to break at paragraph or sentence boundary
            if end < length:
                end = self._find_break_point(content, end)
            chunks.append(content[start:end])
            if end >= length:
                break
            start = end - self.overlap_chars
            if start >= end:
                break

        return chunks

    def _find_break_point(self, content: str, target: int) -> int:
        """Find a good break point near target index (paragraph > sentence > any)."""
        search_range = min(100, target // 4)
        # Look for double newline (paragraph) backward
        for i in range(target, max(target - search_range, 0), -1):
            if content[i : i + 2] == "\n\n":
                return i + 2
        # Look for sentence ending punctuation + space/newline
        for i in range(target, max(target - search_range, 0), -1):
            if content[i] in "。.!?" and i + 1 < len(content) and content[i + 1] in " \n":
                return i + 1
        return target

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return int(len(text) * self.TOKENS_PER_CHAR)
