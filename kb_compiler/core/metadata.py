"""Metadata tracking for incremental compilation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileMeta:
    """Metadata for a tracked file."""

    path: str
    hash: str
    size: int
    mtime: float
    last_compiled: Optional[str] = None
    concepts_extracted: list[str] = None

    def __post_init__(self):
        if self.concepts_extracted is None:
            self.concepts_extracted = []


@dataclass
class ConceptMeta:
    """Metadata for a compiled concept."""

    name: str
    source_files: list[str]
    created_at: str
    updated_at: str
    related_concepts: list[str] = None
    backlink_count: int = 0

    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []


class MetadataTracker:
    """Tracks file and concept metadata for incremental compilation."""

    META_FILE = "compile_state.json"

    def __init__(self, meta_dir: Path):
        self.meta_dir = meta_dir
        self.meta_file = meta_dir / self.META_FILE
        self._data = self._load()

    def _load(self) -> dict:
        """Load metadata from disk."""
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "version": 1,
            "last_compile": None,
            "files": {},
            "concepts": {},
        }

    def save(self) -> None:
        """Save metadata to disk."""
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """Compute MD5 hash of file content."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_file_meta(self, path: str) -> Optional[FileMeta]:
        """Get metadata for a file."""
        if path in self._data["files"]:
            return FileMeta(**self._data["files"][path])
        return None

    def update_file(self, file_path: Path, concepts: list[str] = None) -> None:
        """Update metadata for a file."""
        path_str = str(file_path)
        stat = file_path.stat()
        file_hash = self.compute_hash(file_path)

        self._data["files"][path_str] = {
            "path": path_str,
            "hash": file_hash,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "last_compiled": datetime.now().isoformat(),
            "concepts_extracted": concepts or [],
        }

    def is_file_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last compilation."""
        path_str = str(file_path)
        existing = self.get_file_meta(path_str)

        if not existing:
            return True

        try:
            stat = file_path.stat()
            current_hash = self.compute_hash(file_path)

            return (
                current_hash != existing.hash
                or stat.st_size != existing.size
                or stat.st_mtime != existing.mtime
            )
        except FileNotFoundError:
            return True

    def get_changed_files(self, raw_dir: Path) -> list[Path]:
        """Get list of changed files in raw directory."""
        changed = []

        for file_path in raw_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                if self.is_file_changed(file_path):
                    changed.append(file_path)

        return changed

    def get_concept_meta(self, name: str) -> Optional[ConceptMeta]:
        """Get metadata for a concept."""
        if name in self._data["concepts"]:
            return ConceptMeta(**self._data["concepts"][name])
        return None

    def update_concept(self, name: str, source_files: list[str], related: list[str]) -> None:
        """Update metadata for a concept."""
        now = datetime.now().isoformat()
        existing = self.get_concept_meta(name)

        self._data["concepts"][name] = {
            "name": name,
            "source_files": source_files,
            "created_at": existing.created_at if existing else now,
            "updated_at": now,
            "related_concepts": related,
            "backlink_count": 0,  # Updated separately
        }

    def list_all_concepts(self) -> list[str]:
        """List all tracked concepts."""
        return list(self._data["concepts"].keys())

    def get_stale_concepts(self, active_sources: set[str]) -> list[str]:
        """Find concepts whose source files no longer exist."""
        stale = []
        for name, meta in self._data["concepts"].items():
            # Check if any source file still exists
            if not any(src in active_sources for src in meta.get("source_files", [])):
                stale.append(name)
        return stale

    def update_last_compile(self) -> None:
        """Update last compilation timestamp."""
        self._data["last_compile"] = datetime.now().isoformat()

    def get_stats(self) -> dict:
        """Get compilation statistics."""
        return {
            "last_compile": self._data["last_compile"],
            "total_files_tracked": len(self._data["files"]),
            "total_concepts": len(self._data["concepts"]),
        }
