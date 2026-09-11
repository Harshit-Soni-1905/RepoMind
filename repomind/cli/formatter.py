"""CLI output formatting utilities for RepoMind.

Provides clean, human-readable terminal output for info, warnings, errors,
agent thought/tool progress steps, and final answers.
"""

import sys
from typing import Optional


class CLIFormatter:
    """Formats terminal output for RepoMind CLI."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose
        self.debug = debug

    def info(self, message: str) -> None:
        """Print an informational message."""
        print(f"[INFO] {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        print(f"[SUCCESS] {message}")

    def warning(self, message: str) -> None:
        """Print a warning message."""
        print(f"[WARNING] {message}", file=sys.stderr)

    def error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Print an error message and optional debug traceback."""
        print(f"[ERROR] {message}", file=sys.stderr)
        if self.debug and exception:
            import traceback
            print("\n--- Traceback (Debug Mode) ---", file=sys.stderr)
            traceback.print_exception(type(exception), exception, exception.__traceback__, file=sys.stderr)

    def step(self, iteration: int, tool_name: str, summary: str) -> None:
        """Print a verbose agent execution step.

        Example: [1] semantic_search -> 5 results
        """
        if self.verbose or self.debug:
            print(f"[{iteration}] {tool_name} -> {summary}")

    def answer(self, answer_text: str) -> None:
        """Print the final answer section."""
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(answer_text)
        print("=" * 60 + "\n")
