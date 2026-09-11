"""Main CLI entry point for RepoMind.

Parses command line arguments and dispatches to appropriate command logic.
"""

import sys
import argparse
from typing import List, Optional

from repomind.cli.formatter import CLIFormatter
from repomind.cli.commands import run_index_command, run_ask_command


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the ArgumentParser for RepoMind CLI."""
    parser = argparse.ArgumentParser(
        prog="repomind",
        description="RepoMind: AI-powered codebase understanding CLI for Python repositories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subparsers for commands (ask, index)
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a natural language question about a repository")
    ask_parser.add_argument("repo_path", help="Path to Python repository directory")
    ask_parser.add_argument("query", help="Question to ask about the codebase")
    ask_parser.add_argument(
        "--provider",
        choices=["gemini", "free-local"],
        help="LLM provider override (default: from config)",
    )
    ask_parser.add_argument(
        "--model",
        help="LLM model name override (default: from config)",
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        help="Number of vector search results (default: from config)",
    )
    ask_parser.add_argument(
        "--depth",
        type=int,
        help="Graph traversal depth (default: from config)",
    )
    ask_parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum ReAct loop iterations (default: from config)",
    )
    ask_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display verbose agent step execution summaries",
    )
    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="Display full exception tracebacks on error",
    )

    # Command: index
    index_parser = subparsers.add_parser("index", help="Scan, parse, chunk, embed, and build graph index for a repository")
    index_parser.add_argument("repo_path", help="Path to Python repository directory")
    index_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display verbose progress logs",
    )
    index_parser.add_argument(
        "--debug",
        action="store_true",
        help="Display full exception tracebacks on error",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point function.

    Args:
        args: List of command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit status code (0 for success, 1 for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    formatter = CLIFormatter(
        verbose=getattr(parsed_args, "verbose", False),
        debug=getattr(parsed_args, "debug", False),
    )

    if parsed_args.command == "ask":
        success = run_ask_command(
            repo_path_str=parsed_args.repo_path,
            query=parsed_args.query,
            formatter=formatter,
            provider_name=parsed_args.provider,
            model_name=parsed_args.model,
            top_k=parsed_args.top_k,
            depth=parsed_args.depth,
            max_iterations=parsed_args.max_iterations,
        )
        return 0 if success else 1

    elif parsed_args.command == "index":
        success = run_index_command(
            repo_path_str=parsed_args.repo_path,
            formatter=formatter,
        )
        return 0 if success else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
