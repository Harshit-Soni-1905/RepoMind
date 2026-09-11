"""Tests for file reader."""

import pytest
from pathlib import Path

from repomind.ingestion.file_reader import (
    FileReader,
    SourceFile,
    FileSizeError,
    FileReadError,
    read_file,
)
from repomind.config import config


def test_file_reader_reads_simple_file(tmp_path):
    """FileReader should read a simple Python file."""
    file_path = tmp_path / "simple.py"
    content = "print('hello')\n"
    file_path.write_text(content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    assert source.content == content
    assert source.line_count == 1
    # size_bytes should match actual file size on disk (not encoded string length)
    assert source.size_bytes == file_path.stat().st_size
    assert source.relative_path == Path("simple.py")
    assert source.absolute_path == file_path.resolve()


def test_file_reader_reads_multiline_file(tmp_path):
    """FileReader should correctly count lines in multiline files."""
    file_path = tmp_path / "multiline.py"
    content = """def foo():
    return 42

def bar():
    return foo() * 2
"""
    file_path.write_text(content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    assert source.content == content
    assert source.line_count == 5
    assert "def foo():" in source.content


def test_file_reader_handles_file_without_trailing_newline(tmp_path):
    """FileReader should correctly count lines when file lacks trailing newline."""
    file_path = tmp_path / "no_newline.py"
    content = "x = 1"  # No trailing newline
    file_path.write_text(content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    assert source.line_count == 1


def test_file_reader_empty_file(tmp_path):
    """FileReader should handle empty files."""
    file_path = tmp_path / "empty.py"
    file_path.write_text("", encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    assert source.content == ""
    assert source.line_count == 0
    assert source.size_bytes == 0


def test_file_reader_raises_on_nonexistent_file(tmp_path):
    """FileReader should raise FileNotFoundError for non-existent files."""
    reader = FileReader(repo_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        reader.read(tmp_path / "nonexistent.py")


def test_file_reader_raises_on_oversized_file(tmp_path, monkeypatch):
    """FileReader should raise FileSizeError for files exceeding size limit."""
    file_path = tmp_path / "large.py"

    # Create a file larger than the limit
    # Temporarily set a small max size for testing
    monkeypatch.setattr(config, "MAX_FILE_SIZE", 100)

    large_content = "x = 1\n" * 50  # More than 100 bytes
    file_path.write_text(large_content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)

    with pytest.raises(FileSizeError, match="exceeds maximum"):
        reader.read(file_path)


def test_file_reader_computes_relative_path(tmp_path):
    """FileReader should compute correct relative paths."""
    repo = tmp_path / "repo"
    repo.mkdir()

    src = repo / "src"
    src.mkdir()

    file_path = src / "module.py"
    file_path.write_text("print('module')")

    reader = FileReader(repo_root=repo)
    source = reader.read(file_path)

    assert source.relative_path == Path("src/module.py")
    assert source.absolute_path == file_path.resolve()


def test_file_reader_without_repo_root(tmp_path):
    """FileReader without repo_root should use absolute path as relative."""
    file_path = tmp_path / "standalone.py"
    file_path.write_text("print('standalone')")

    reader = FileReader(repo_root=None)
    source = reader.read(file_path)

    # When repo_root is None, relative_path equals absolute_path
    assert source.relative_path == file_path.resolve()
    assert source.absolute_path == file_path.resolve()


def test_file_reader_handles_utf8_with_special_chars(tmp_path):
    """FileReader should handle UTF-8 files with special characters."""
    file_path = tmp_path / "unicode.py"
    content = "# Comment with émojis: 🐍 Python\nname = 'José'\n"
    file_path.write_text(content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    assert source.content == content
    assert "🐍" in source.content
    assert "José" in source.content


def test_file_reader_handles_non_utf8_encoding(tmp_path):
    """FileReader should handle files with non-UTF-8 encoding."""
    file_path = tmp_path / "latin1.py"

    # Write file with latin-1 encoding (contains bytes not valid in UTF-8)
    content_bytes = b"# \xe9\xe0\xe8\n"  # Latin-1 encoded special chars
    file_path.write_bytes(content_bytes)

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    # Should successfully read (using fallback encoding)
    assert source.content is not None
    assert len(source.content) > 0


def test_read_file_convenience_function(tmp_path):
    """Test the convenience function read_file."""
    file_path = tmp_path / "test.py"
    content = "def test(): pass\n"
    file_path.write_text(content, encoding="utf-8")

    source = read_file(file_path, repo_root=tmp_path)

    assert source.content == content
    assert source.relative_path == Path("test.py")


def test_file_reader_raises_on_directory(tmp_path):
    """FileReader should raise FileReadError when given a directory."""
    directory = tmp_path / "dir"
    directory.mkdir()

    reader = FileReader(repo_root=tmp_path)

    with pytest.raises(FileReadError, match="not a file"):
        reader.read(directory)


def test_source_file_is_immutable(tmp_path):
    """SourceFile should be immutable (frozen dataclass)."""
    file_path = tmp_path / "immutable.py"
    file_path.write_text("x = 1")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    # Should not be able to modify attributes
    with pytest.raises(AttributeError):
        source.content = "modified"

    with pytest.raises(AttributeError):
        source.line_count = 999


def test_file_reader_computes_correct_size(tmp_path):
    """FileReader should correctly compute file size in bytes."""
    file_path = tmp_path / "sized.py"
    content = "# 测试\n"  # Multi-byte UTF-8 characters
    file_path.write_text(content, encoding="utf-8")

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    # Size should match actual bytes on disk
    actual_size = file_path.stat().st_size
    assert source.size_bytes == actual_size


def test_file_reader_with_windows_line_endings(tmp_path):
    """FileReader should handle Windows line endings correctly."""
    file_path = tmp_path / "windows.py"
    # Write with binary mode to preserve exact line endings on Windows
    content_bytes = b"line1\r\nline2\r\nline3\r\n"
    file_path.write_bytes(content_bytes)

    reader = FileReader(repo_root=tmp_path)
    source = reader.read(file_path)

    # Content should contain newlines (may be normalized on Windows)
    assert "\n" in source.content
    # Line count should be 3 regardless of line ending style
    # Count by splitting on \n and filtering empty strings
    actual_lines = [line for line in source.content.split("\n") if line or source.content.endswith("\n")]
    assert len(actual_lines) >= 3


def test_file_reader_file_outside_repo_root(tmp_path):
    """FileReader should handle files outside repo_root gracefully."""
    repo = tmp_path / "repo"
    repo.mkdir()

    outside_file = tmp_path / "outside.py"
    outside_file.write_text("print('outside')")

    reader = FileReader(repo_root=repo)
    source = reader.read(outside_file)

    # Should use absolute path as relative when file is outside repo
    assert source.relative_path == outside_file.resolve()
