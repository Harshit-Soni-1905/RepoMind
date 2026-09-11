"""Ingestion subsystem: repository scanning and file reading.

This subsystem is responsible for discovering Python files in a repository
and reading them safely with proper encoding detection and error handling.
"""

from repomind.ingestion.scanner import RepositoryScanner, scan_repository
from repomind.ingestion.file_reader import (
    FileReader,
    SourceFile,
    FileSizeError,
    FileReadError,
    read_file,
)

__all__ = [
    "RepositoryScanner",
    "scan_repository",
    "FileReader",
    "SourceFile",
    "FileSizeError",
    "FileReadError",
    "read_file",
]
