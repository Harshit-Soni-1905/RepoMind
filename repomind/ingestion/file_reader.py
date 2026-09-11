"""File reader: safely reads Python source files into memory.

This module handles reading files with proper encoding detection,
size limits, and error handling.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from repomind.config import config


@dataclass(frozen=True)
class SourceFile:
    """Represents a Python source file loaded into memory.

    Attributes:
        relative_path: Path relative to repository root
        absolute_path: Absolute path to the file
        content: Full source code as string
        size_bytes: File size in bytes
        line_count: Number of lines in the file
    """
    relative_path: Path
    absolute_path: Path
    content: str
    size_bytes: int
    line_count: int


class FileSizeError(Exception):
    """Raised when a file exceeds the maximum allowed size."""
    pass


class FileReadError(Exception):
    """Raised when a file cannot be read."""
    pass


class FileReader:
    """Reads Python source files safely with size and encoding checks."""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize file reader.

        Args:
            repo_root: Optional repository root for computing relative paths.
                      If None, relative_path will be the same as absolute_path.
        """
        self.repo_root = Path(repo_root).resolve() if repo_root else None

    def read(self, file_path: Path) -> SourceFile:
        """Read a Python source file into memory.

        Args:
            file_path: Path to the file to read

        Returns:
            SourceFile object containing file metadata and content

        Raises:
            FileNotFoundError: If file doesn't exist
            FileSizeError: If file exceeds MAX_FILE_SIZE
            FileReadError: If file cannot be read (encoding, permissions, etc.)

        Example:
            >>> reader = FileReader(repo_root=Path("/repo"))
            >>> source = reader.read(Path("/repo/src/main.py"))
            >>> print(source.line_count)
            42
        """
        path_obj = Path(file_path)
        if self.repo_root and not path_obj.is_absolute():
            file_path = (self.repo_root / path_obj).resolve()
        else:
            file_path = path_obj.resolve()

        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise FileReadError(f"Path is not a file: {file_path}")

        # Check file size
        size_bytes = file_path.stat().st_size

        if size_bytes > config.MAX_FILE_SIZE:
            raise FileSizeError(
                f"File size ({size_bytes} bytes) exceeds maximum "
                f"({config.MAX_FILE_SIZE} bytes): {file_path}"
            )

        # Read file content
        try:
            content = self._read_with_encoding(file_path)
        except Exception as e:
            raise FileReadError(f"Failed to read file {file_path}: {e}") from e

        # Compute relative path
        if self.repo_root:
            try:
                relative_path = file_path.relative_to(self.repo_root)
            except ValueError:
                # File is outside repo root, use absolute path
                relative_path = file_path
        else:
            relative_path = file_path

        # Count lines
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        return SourceFile(
            relative_path=relative_path,
            absolute_path=file_path,
            content=content,
            size_bytes=size_bytes,
            line_count=line_count,
        )

    def _read_with_encoding(self, file_path: Path) -> str:
        """Read file with encoding detection.

        Tries UTF-8 first, falls back to latin-1 if UTF-8 fails.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            Exception: If file cannot be read with any encoding
        """
        # Try UTF-8 first (most common for Python source)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            pass

        # Fall back to latin-1 (accepts all byte sequences)
        # This ensures we can always read the file, even if encoding is wrong
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            # If even latin-1 fails, try reading as binary and decode with replacement
            with open(file_path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")


def read_file(file_path: Path, repo_root: Optional[Path] = None) -> SourceFile:
    """Convenience function to read a single Python file.

    Args:
        file_path: Path to file to read
        repo_root: Optional repository root for relative path computation

    Returns:
        SourceFile object

    Raises:
        FileNotFoundError: If file doesn't exist
        FileSizeError: If file exceeds size limit
        FileReadError: If file cannot be read

    Example:
        >>> source = read_file(Path("main.py"))
        >>> print(source.content[:50])
    """
    reader = FileReader(repo_root=repo_root)
    return reader.read(file_path)
