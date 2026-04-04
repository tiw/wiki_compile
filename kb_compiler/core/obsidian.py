"""Obsidian vault interaction via obsidian-cli and direct file operations."""

import json
import subprocess
from pathlib import Path
from typing import Optional

import frontmatter
import yaml
from rich.console import Console

console = Console()


class ObsidianClient:
    """Client for interacting with Obsidian vault."""

    def __init__(
        self,
        vault_name: Optional[str] = None,
        vault_path: Optional[Path] = None,
    ):
        self.vault_name = vault_name
        self.vault_path = vault_path
        self._use_cli = vault_name is not None

        if vault_path and not vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

    def _run_cli(self, *args: str) -> str:
        """Run obsidian-cli command."""
        if not self._use_cli:
            raise RuntimeError("CLI not configured (no vault name)")

        cmd = ["obsidian-cli", "-v", self.vault_name, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            console.print(f"[red]obsidian-cli error: {e.stderr}[/]")
            raise

    def read_note(self, path: str) -> tuple[dict, str]:
        """Read a note and return (frontmatter, content).

        Args:
            path: Path to note relative to vault root (e.g., "wiki/concepts/AI.md")

        Returns:
            Tuple of (frontmatter dict, content string)
        """
        if self.vault_path:
            full_path = self.vault_path / path
            if not full_path.exists():
                raise FileNotFoundError(f"Note not found: {full_path}")

            with open(full_path, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
                return dict(post.metadata), post.content
        else:
            # Use CLI - obsidian-cli doesn't support direct read,
            # so we need to use the vault path if available
            raise NotImplementedError(
                "Direct note reading requires vault_path. "
                "Please provide obsidian_vault_path in config."
            )

    def write_note(
        self,
        path: str,
        content: str,
        frontmatter_data: Optional[dict] = None,
    ) -> None:
        """Write a note with optional frontmatter.

        Args:
            path: Path relative to vault root
            content: Note content
            frontmatter_data: Optional frontmatter dictionary
        """
        if self.vault_path:
            full_path = self.vault_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if content already has frontmatter
            if content.startswith("---"):
                # Content already has frontmatter, write directly
                full_path.write_text(content, encoding="utf-8")
            else:
                # Add frontmatter manually to avoid python-frontmatter issues
                import yaml
                if frontmatter_data:
                    yaml_content = yaml.dump(
                        frontmatter_data,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False
                    )
                    full_content = f"---\n{yaml_content}---\n\n{content}"
                else:
                    full_content = content
                full_path.write_text(full_content, encoding="utf-8")

            console.print(f"[green]Wrote: {path}[/]")
        else:
            # For CLI-only mode, we need to write to a temp file and use create
            raise NotImplementedError(
                "Note writing requires vault_path. "
                "Please provide obsidian_vault_path in config."
            )

    def note_exists(self, path: str) -> bool:
        """Check if a note exists."""
        if self.vault_path:
            return (self.vault_path / path).exists()
        else:
            # Try to read via CLI (list and check)
            try:
                notes = self.list_notes()
                return path in notes
            except Exception:
                return False

    def list_notes(self, directory: Optional[str] = None) -> list[str]:
        """List all notes in the vault or a specific directory."""
        if self.vault_path:
            search_path = self.vault_path
            if directory:
                search_path = search_path / directory

            notes = []
            for md_file in search_path.rglob("*.md"):
                rel_path = md_file.relative_to(self.vault_path)
                notes.append(str(rel_path))
            return sorted(notes)
        else:
            # Use CLI
            result = self._run_cli("list", "notes")
            notes = []
            for line in result.strip().split("\n"):
                line = line.strip()
                if line and line.endswith(".md"):
                    notes.append(line)
            return notes

    def get_note_links(self, path: str) -> dict:
        """Get outgoing and incoming links for a note.

        Returns:
            Dict with 'outgoing' and 'incoming' link lists
        """
        metadata, content = self.read_note(path)

        # Parse wiki-links from content
        import re

        wiki_links = re.findall(r"\[\[([^\]]+)\]\]", content)

        return {
            "outgoing": list(set(wiki_links)),
            "incoming": [],  # Would need graph analysis for incoming
        }

    def update_frontmatter(self, path: str, updates: dict) -> None:
        """Update frontmatter of an existing note."""
        metadata, content = self.read_note(path)
        metadata.update(updates)
        self.write_note(path, content, metadata)

    def create_directory(self, path: str) -> None:
        """Create a directory in the vault."""
        if self.vault_path:
            (self.vault_path / path).mkdir(parents=True, exist_ok=True)
        # CLI doesn't have directory creation command

    def search_notes(self, query: str) -> list[str]:
        """Search notes by content (simple text search)."""
        if self.vault_path:
            matches = []
            for md_file in self.vault_path.rglob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            rel_path = md_file.relative_to(self.vault_path)
                            matches.append(str(rel_path))
                except Exception:
                    continue
            return matches
        else:
            raise NotImplementedError("Search requires vault_path")

    def get_all_wiki_links(self) -> dict:
        """Build a map of all wiki links in the vault.

        Returns:
            Dict mapping note paths to their outgoing links
        """
        notes = self.list_notes()
        links_map = {}

        for note_path in notes:
            try:
                _, content = self.read_note(note_path)
                import re

                wiki_links = re.findall(r"\[\[([^\]]+)\]\]", content)
                links_map[note_path] = wiki_links
            except Exception:
                continue

        return links_map

    def get_backlinks(self, concept_name: str) -> list[str]:
        """Get all notes that link to a concept.

        Args:
            concept_name: Name of concept (without .md extension)

        Returns:
            List of note paths that link to this concept
        """
        all_links = self.get_all_wiki_links()
        backlinks = []

        for note_path, links in all_links.items():
            # Check if any link matches the concept name
            for link in links:
                if link.lower() == concept_name.lower() or link.lower().startswith(
                    concept_name.lower() + "."
                ):
                    backlinks.append(note_path)
                    break

        return backlinks
