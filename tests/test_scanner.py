"""Tests for repository scanner."""

import pytest
from pathlib import Path
import tempfile
import shutil

from repomind.ingestion.scanner import RepositoryScanner, scan_repository


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository structure for testing.

    Structure:
        repo/
        ├── main.py
        ├── utils.py
        ├── src/
        │   ├── __init__.py
        │   └── core.py
        ├── tests/
        │   └── test_main.py
        ├── __pycache__/
        │   └── main.cpython-39.pyc
        ├── .venv/
        │   └── lib/
        │       └── something.py
        └── docs/
            └── README.md
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create Python files
    (repo / "main.py").write_text("print('main')")
    (repo / "utils.py").write_text("def helper(): pass")

    # Create src directory with files
    src = repo / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "core.py").write_text("class Core: pass")

    # Create tests directory
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_something(): pass")

    # Create __pycache__ (should be skipped)
    pycache = repo / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-39.pyc").write_bytes(b"fake bytecode")

    # Create .venv (should be skipped)
    venv = repo / ".venv"
    venv.mkdir()
    venv_lib = venv / "lib"
    venv_lib.mkdir()
    (venv_lib / "something.py").write_text("# This should be ignored")

    # Create non-Python file
    docs = repo / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Documentation")

    return repo


def test_scanner_finds_python_files(sample_repo):
    """Scanner should find all .py files."""
    scanner = RepositoryScanner(sample_repo)
    files = scanner.scan()

    # Convert to relative paths for easier assertion
    rel_files = [f.relative_to(sample_repo) for f in files]
    rel_files_str = [str(f).replace("\\", "/") for f in rel_files]

    assert "main.py" in rel_files_str
    assert "utils.py" in rel_files_str
    assert "src/__init__.py" in rel_files_str
    assert "src/core.py" in rel_files_str
    assert "tests/test_main.py" in rel_files_str


def test_scanner_skips_pycache(sample_repo):
    """Scanner should skip __pycache__ directories."""
    scanner = RepositoryScanner(sample_repo)
    files = scanner.scan()

    # No files from __pycache__ should be present
    for f in files:
        assert "__pycache__" not in str(f)


def test_scanner_skips_venv(sample_repo):
    """Scanner should skip .venv directories."""
    scanner = RepositoryScanner(sample_repo)
    files = scanner.scan()

    # No files from .venv should be present
    for f in files:
        assert ".venv" not in str(f)
        assert "venv" not in f.parts


def test_scanner_skips_non_python_files(sample_repo):
    """Scanner should only return .py files."""
    scanner = RepositoryScanner(sample_repo)
    files = scanner.scan()

    # All files should have .py extension
    for f in files:
        assert f.suffix == ".py"

    # Check that README.md is not included
    rel_files = [f.relative_to(sample_repo) for f in files]
    assert Path("docs/README.md") not in rel_files


def test_scanner_respects_gitignore(tmp_path):
    """Scanner should respect .gitignore patterns."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create files
    (repo / "main.py").write_text("print('main')")
    (repo / "secret.py").write_text("API_KEY = 'secret'")
    (repo / "config.py").write_text("CONFIG = {}")

    # Create build directory
    build = repo / "build"
    build.mkdir()
    (build / "output.py").write_text("# Generated file")

    # Create .gitignore
    gitignore_content = """
# Ignore secret files
secret.py

# Ignore build directory
build/
"""
    (repo / ".gitignore").write_text(gitignore_content)

    scanner = RepositoryScanner(repo)
    files = scanner.scan()

    rel_files = [f.relative_to(repo) for f in files]
    rel_files_str = [str(f).replace("\\", "/") for f in rel_files]

    # main.py and config.py should be found
    assert "main.py" in rel_files_str
    assert "config.py" in rel_files_str

    # secret.py and build/ should be ignored
    assert "secret.py" not in rel_files_str
    assert "build/output.py" not in rel_files_str


def test_scanner_returns_sorted_results(sample_repo):
    """Scanner should return files in sorted order."""
    scanner = RepositoryScanner(sample_repo)
    files = scanner.scan()

    # Results should be sorted
    assert files == sorted(files)


def test_scanner_raises_on_nonexistent_path():
    """Scanner should raise ValueError for non-existent path."""
    with pytest.raises(ValueError, match="does not exist"):
        RepositoryScanner(Path("/nonexistent/path/to/repo"))


def test_scanner_raises_on_file_path(tmp_path):
    """Scanner should raise ValueError when given a file instead of directory."""
    file_path = tmp_path / "file.py"
    file_path.write_text("print('test')")

    with pytest.raises(ValueError, match="not a directory"):
        RepositoryScanner(file_path)


def test_scan_repository_convenience_function(sample_repo):
    """Test the convenience function scan_repository."""
    files = scan_repository(sample_repo)

    # Should return same results as using RepositoryScanner directly
    scanner = RepositoryScanner(sample_repo)
    expected_files = scanner.scan()

    assert files == expected_files


def test_scanner_handles_permission_errors(tmp_path):
    """Scanner should gracefully handle directories it can't read."""
    # This test is platform-dependent and may not work on all systems
    # Skip on Windows where permission handling is different
    import sys
    if sys.platform == "win32":
        pytest.skip("Permission test not reliable on Windows")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "accessible.py").write_text("print('ok')")

    restricted = repo / "restricted"
    restricted.mkdir()
    (restricted / "hidden.py").write_text("print('hidden')")

    # Remove read permissions
    restricted.chmod(0o000)

    try:
        scanner = RepositoryScanner(repo)
        files = scanner.scan()

        # Should find accessible.py, skip restricted/
        rel_files = [f.relative_to(repo) for f in files]
        assert Path("accessible.py") in rel_files
        assert Path("restricted/hidden.py") not in rel_files
    finally:
        # Restore permissions for cleanup
        restricted.chmod(0o755)


def test_scanner_empty_directory(tmp_path):
    """Scanner should handle empty directories gracefully."""
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()

    scanner = RepositoryScanner(empty_repo)
    files = scanner.scan()

    assert files == []


def test_scanner_with_nested_structure(tmp_path):
    """Scanner should handle deeply nested directory structures."""
    repo = tmp_path / "repo"
    repo.mkdir()

    # Create deeply nested structure
    nested = repo / "a" / "b" / "c" / "d" / "e"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("print('deep')")

    scanner = RepositoryScanner(repo)
    files = scanner.scan()

    assert len(files) == 1
    assert files[0].name == "deep.py"
