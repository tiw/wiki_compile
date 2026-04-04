"""Phase 1: Ingest - Document intake from multiple sources."""

import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
import yaml
from markitdown import MarkItDown
from rich.console import Console

console = Console()


class DocumentIngester:
    """Ingest documents from various sources into raw/ directory."""

    SUPPORTED_EXTENSIONS = {
        ".md",
        ".txt",
        ".pdf",
        ".html",
        ".htm",
        ".docx",
        ".pptx",
        ".xlsx",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
    }

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.md = MarkItDown()

    def ingest_url(self, url: str, metadata: Optional[dict] = None) -> Path:
        """Ingest a web URL as markdown."""
        console.print(f"[blue]Fetching: {url}[/]")

        try:
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            response.raise_for_status()

            # Generate filename from URL
            parsed = urlparse(url)
            domain = parsed.netloc.replace(".", "_")
            path = parsed.path.strip("/").replace("/", "_") or "index"
            filename = f"{domain}_{path}.md"
            filepath = self.raw_dir / "articles" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert HTML to markdown
            result = self.md.convert_string(response.text)
            content = result.text_content

            # Add frontmatter
            frontmatter = {
                "source_url": url,
                "title": self._extract_title(response.text) or filename,
                "ingested_at": self._now(),
                **(metadata or {}),
            }

            self._write_with_frontmatter(filepath, content, frontmatter)
            console.print(f"[green]Ingested URL to: {filepath}[/]")
            return filepath

        except Exception as e:
            console.print(f"[red]Failed to ingest URL {url}: {e}[/]")
            raise

    def ingest_file(self, source_path: Path, subdir: str = "", metadata: Optional[dict] = None) -> Path:
        """Ingest a local file into raw directory."""
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Determine destination
        dest_dir = self.raw_dir / subdir if subdir else self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        ext = source_path.suffix.lower()

        if ext in (".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"):
            # Text files - copy as-is or convert
            dest_path = dest_dir / source_path.name
            content = source_path.read_text(encoding="utf-8")

            # Add frontmatter if not present
            if not content.startswith("---"):
                frontmatter = {
                    "source_file": str(source_path),
                    "ingested_at": self._now(),
                    **(metadata or {}),
                }
                self._write_with_frontmatter(dest_path, content, frontmatter)
            else:
                shutil.copy2(source_path, dest_path)

        elif ext == ".pdf":
            dest_path = dest_dir / f"{source_path.stem}.md"
            result = self.md.convert(str(source_path))
            content = result.text_content

            frontmatter = {
                "source_file": str(source_path),
                "title": source_path.stem,
                "ingested_at": self._now(),
                **(metadata or {}),
            }
            self._write_with_frontmatter(dest_path, content, frontmatter)

        elif ext in (".html", ".htm"):
            dest_path = dest_dir / f"{source_path.stem}.md"
            result = self.md.convert(str(source_path))
            content = result.text_content

            frontmatter = {
                "source_file": str(source_path),
                "title": source_path.stem,
                "ingested_at": self._now(),
                **(metadata or {}),
            }
            self._write_with_frontmatter(dest_path, content, frontmatter)

        elif ext in (".docx", ".pptx", ".xlsx"):
            dest_path = dest_dir / f"{source_path.stem}.md"
            result = self.md.convert(str(source_path))
            content = result.text_content

            frontmatter = {
                "source_file": str(source_path),
                "title": source_path.stem,
                "ingested_at": self._now(),
                **(metadata or {}),
            }
            self._write_with_frontmatter(dest_path, content, frontmatter)

        else:
            # Binary files - copy as-is with metadata sidecar
            dest_path = dest_dir / source_path.name
            shutil.copy2(source_path, dest_path)

            # Create metadata file
            meta_path = dest_dir / f"{source_path.name}.meta.yaml"
            meta_content = {
                "source_file": str(source_path),
                "original_name": source_path.name,
                "ingested_at": self._now(),
                **(metadata or {}),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.dump(meta_content, f)

        console.print(f"[green]Ingested: {source_path.name} -> {dest_path}[/]")
        return dest_path

    def ingest_text(self, content: str, filename: str, subdir: str = "", metadata: Optional[dict] = None) -> Path:
        """Ingest raw text content."""
        dest_dir = self.raw_dir / subdir if subdir else self.raw_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not filename.endswith(".md"):
            filename += ".md"

        filepath = dest_dir / filename

        frontmatter = {
            "title": filename.replace(".md", ""),
            "ingested_at": self._now(),
            **(metadata or {}),
        }

        self._write_with_frontmatter(filepath, content, frontmatter)
        console.print(f"[green]Ingested text to: {filepath}[/]")
        return filepath

    def ingest_directory(self, source_dir: Path, recursive: bool = True) -> list[Path]:
        """Ingest all supported files from a directory."""
        ingested = []

        pattern = "**/*" if recursive else "*"
        for file_path in source_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    # Preserve relative directory structure
                    rel_path = file_path.relative_to(source_dir)
                    subdir = str(rel_path.parent) if str(rel_path.parent) != "." else ""

                    dest_path = self.ingest_file(file_path, subdir)
                    ingested.append(dest_path)
                except Exception as e:
                    console.print(f"[yellow]Skipping {file_path}: {e}[/]")

        console.print(f"[green]Ingested {len(ingested)} files from {source_dir}[/]")
        return ingested

    def _write_with_frontmatter(self, path: Path, content: str, frontmatter: dict) -> None:
        """Write content with YAML frontmatter."""
        yaml_content = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
        full_content = f"---\n{yaml_content}---\n\n{content}"
        path.write_text(full_content, encoding="utf-8")

    def _extract_title(self, html: str) -> Optional[str]:
        """Extract title from HTML."""
        import re
        match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _now() -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class QuickCapture:
    """Quick capture for ideas and notes."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.inbox_dir = raw_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, content: str, tags: Optional[list[str]] = None) -> Path:
        """Quickly capture a note to inbox."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.md"
        filepath = self.inbox_dir / filename

        frontmatter = {
            "type": "quick_capture",
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
        }

        yaml_content = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
        full_content = f"---\n{yaml_content}---\n\n{content}"
        filepath.write_text(full_content, encoding="utf-8")

        console.print(f"[green]Captured to inbox: {filepath}[/]")
        return filepath
