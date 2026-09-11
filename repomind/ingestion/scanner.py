"""Repository scanner: discovers Python files in a directory tree.

This module walks a repository directory, finds all .py files, and respects
.gitignore patterns and configured skip directories.
"""

from pathlib import Path
from typing import List, Optional
import pathspec

from repomind.config import config


class RepositoryScanner:
    """Scans a repository directory for Python source files.

    Respects .gitignore patterns and skips common non-source directories
    like __pycache__, .venv, .git, etc.
    """

    def __init__(self, root_path: Path):
        """Initialize scanner for a repository.

        Args:
            root_path: Root directory of the repository to scan

        Raises:
            ValueError: If root_path doesn't exist or isn't a directory
        """
        self.root_path = Path(root_path).resolve()

        if not self.root_path.exists():
            raise ValueError(f"Repository path does not exist: {self.root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {self.root_path}")

        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns if the file exists."""
        gitignore_path = self.root_path / ".gitignore"

        if gitignore_path.exists() and gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    patterns = f.read().splitlines()
                # Filter out comments and empty lines
                patterns = [p.strip() for p in patterns if p.strip() and not p.strip().startswith("#")]
                self._gitignore_spec = pathspec.PathSpec.from_lines("gitignore", patterns)
            except Exception:
                # If .gitignore parsing fails, continue without it
                self._gitignore_spec = None

    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be skipped during traversal.

        Args:
            dir_path: Directory to check

        Returns:
            True if directory should be skipped, False otherwise
        """
        dir_name = dir_path.name

        # Skip configured directories (venv, __pycache__, etc.)
        if dir_name in config.SKIP_DIRECTORIES:
            return True

        # Check .gitignore patterns
        if self._gitignore_spec:
            # Get path relative to repo root for .gitignore matching
            try:
                rel_path = dir_path.relative_to(self.root_path)
                # Check both with and without trailing slash
                if self._gitignore_spec.match_file(str(rel_path)):
                    return True
                if self._gitignore_spec.match_file(str(rel_path) + "/"):
                    return True
            except ValueError:
                # Path is not relative to root (shouldn't happen in practice)
                pass

        return False

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included in scan results.

        Args:
            file_path: File to check

        Returns:
            True if file should be included, False otherwise
        """
        # Only include .py files
        if file_path.suffix != ".py":
            return False

        # Check .gitignore patterns
        if self._gitignore_spec:
            try:
                rel_path = file_path.relative_to(self.root_path)
                if self._gitignore_spec.match_file(str(rel_path)):
                    return False
            except ValueError:
                # Path is not relative to root
                pass

        return True

    def scan(self) -> List[Path]:
        """Scan the repository and return all Python files.

        Returns:
            List of absolute Path objects for all discovered .py files,
            sorted lexicographically for deterministic results

        Example:
            >>> scanner = RepositoryScanner(Path("/path/to/repo"))
            >>> python_files = scanner.scan()
            >>> print(python_files[0])
            /path/to/repo/src/main.py
        """
        python_files: List[Path] = []

        # Walk directory tree
        for path in self._walk_directory(self.root_path):
            if path.is_file() and self._should_include_file(path):
                python_files.append(path)

        # Sort for deterministic ordering
        return sorted(python_files)

    def _walk_directory(self, directory: Path):
        """Recursively walk a directory, yielding all paths.

        Skips directories based on skip rules.

        Args:
            directory: Directory to walk

        Yields:
            Path objects for all files and directories found
        """
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    if not self._should_skip_directory(item):
                        # Recursively walk subdirectory
                        yield from self._walk_directory(item)
                else:
                    yield item
        except PermissionError:
            # Skip directories we can't read
            pass


def scan_repository(repo_path: Path) -> List[Path]:
    """Convenience function to scan a repository for Python files.

    Args:
        repo_path: Root directory of repository

    Returns:
        List of absolute Path objects for all .py files found

    Raises:
        ValueError: If repo_path doesn't exist or isn't a directory

    Example:
        >>> files = scan_repository(Path("./my_project"))
        >>> print(f"Found {len(files)} Python files")
    """
    scanner = RepositoryScanner(repo_path)
    return scanner.scan()
