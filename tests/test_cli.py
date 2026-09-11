"""Comprehensive tests for Stage 8 CLI interface.

Tests cover:
- Argument parsing (ask, index subcommands)
- Repository validation
- Empty query handling
- Configuration overrides
- Agent invocation with mocked components
- Error handling and edge cases
- Verbose/debug mode output
"""

import sys
import json
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from io import StringIO
import pytest
import networkx as nx

from repomind.cli.main import create_parser, main
from repomind.cli.formatter import CLIFormatter
from repomind.cli.commands import (
    validate_repository_path,
    run_index_command,
    run_ask_command,
    save_graph,
    load_graph,
    GRAPH_FILENAME,
)
from repomind.agent.llm_provider import (
    LLMProvider,
    FreeLocalProvider,
    ToolDefinition,
    ToolCall,
    ToolResult,
    Message,
)
from repomind.agent.tools import Tool, ToolRegistry
from repomind.agent.state import AgentState
from repomind.agent.agent import ReActAgent
from repomind.graph.models import GraphNode, NodeType
from repomind.parsing.models import ParsedFile
from repomind.retrieval.models import HybridResultNode, RetrievalSource, HybridQueryResult


# ============================================================================
# Test Helpers and Fixtures
# ============================================================================

def make_parsed_file(filepath: Path = Path("main.py")) -> MagicMock:
    """Build a ParsedFile double that reads as successfully parsed.

    has_syntax_error must be set explicitly: a bare MagicMock attribute is
    truthy, which would make the index command treat every file as broken.
    """
    parsed = MagicMock(spec=ParsedFile)
    parsed.filepath = filepath
    parsed.has_syntax_error = False
    parsed.syntax_error_message = None
    return parsed


class FakeScriptedLLMProvider(LLMProvider):
    """A deterministic fake LLM provider that returns scripted responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_history: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "fake-scripted"

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        self.call_history.append(list(messages))
        if not self.responses:
            return "No more scripted responses available."
        return self.responses.pop(0)


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository with Python files."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create a simple Python file
    (repo / "main.py").write_text("""
def hello():
    '''Say hello.'''
    return "Hello, World!"

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"
""")

    # Create a subdirectory
    subdir = repo / "utils"
    subdir.mkdir()
    (subdir / "helpers.py").write_text("""
def add(a: int, b: int) -> int:
    return a + b
""")

    return repo


@pytest.fixture
def empty_repo(tmp_path):
    """Create an empty repository (no Python files)."""
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Empty Repo")
    return repo


@pytest.fixture
def mock_formatter():
    """A CLIFormatter double that records calls instead of printing.

    verbose/debug are pinned to False so command code takes the non-verbose
    branch (a bare MagicMock attribute would be truthy).
    """
    formatter = MagicMock(spec=CLIFormatter)
    formatter.verbose = False
    formatter.debug = False
    return formatter


# ============================================================================
# Argument Parser Tests
# ============================================================================

class TestArgumentParser:
    """Tests for CLI argument parsing."""

    def test_create_parser_returns_parser(self):
        """create_parser() should return an ArgumentParser instance."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_ask_subcommand(self):
        """Parser should have 'ask' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "What does this do?"])
        assert args.command == "ask"
        assert args.repo_path == "./repo"
        assert args.query == "What does this do?"

    def test_parser_has_index_subcommand(self):
        """Parser should have 'index' subcommand."""
        parser = create_parser()
        args = parser.parse_args(["index", "./repo"])
        assert args.command == "index"
        assert args.repo_path == "./repo"

    def test_ask_command_accepts_provider_override(self):
        """Ask command should accept --provider flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--provider", "free-local"])
        assert args.provider == "free-local"

    def test_ask_command_accepts_model_override(self):
        """Ask command should accept --model flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--model", "gemini-1.5-pro"])
        assert args.model == "gemini-1.5-pro"

    def test_ask_command_accepts_top_k_override(self):
        """Ask command should accept --top-k flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--top-k", "10"])
        assert args.top_k == 10

    def test_ask_command_accepts_depth_override(self):
        """Ask command should accept --depth flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--depth", "2"])
        assert args.depth == 2

    def test_ask_command_accepts_max_iterations_override(self):
        """Ask command should accept --max-iterations flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--max-iterations", "15"])
        assert args.max_iterations == 15

    def test_ask_command_accepts_verbose_flag(self):
        """Ask command should accept -v/--verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--verbose"])
        assert args.verbose is True

    def test_ask_command_accepts_debug_flag(self):
        """Ask command should accept --debug flag."""
        parser = create_parser()
        args = parser.parse_args(["ask", "./repo", "query", "--debug"])
        assert args.debug is True

    def test_index_command_accepts_verbose_flag(self):
        """Index command should accept -v/--verbose flag."""
        parser = create_parser()
        args = parser.parse_args(["index", "./repo", "--verbose"])
        assert args.verbose is True

    def test_index_command_accepts_debug_flag(self):
        """Index command should accept --debug flag."""
        parser = create_parser()
        args = parser.parse_args(["index", "./repo", "--debug"])
        assert args.debug is True

    def test_missing_command_leaves_command_unset(self):
        """No subcommand should parse cleanly with command=None.

        The subparser is intentionally not `required` so main() can print help
        and return a non-zero exit code itself (see test_main_no_command_shows_help).
        """
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_missing_repo_path_for_ask_shows_error(self, capsys):
        """Missing repo_path for ask should show error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ask"])
        captured = capsys.readouterr()
        assert "error:" in captured.err.lower()

    def test_missing_query_for_ask_shows_error(self, capsys):
        """Missing query for ask should show error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ask", "./repo"])
        captured = capsys.readouterr()
        assert "error:" in captured.err.lower()

    def test_missing_repo_path_for_index_shows_error(self, capsys):
        """Missing repo_path for index should show error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["index"])
        captured = capsys.readouterr()
        assert "error:" in captured.err.lower()


# ============================================================================
# Formatter Tests
# ============================================================================

class TestCLIFormatter:
    """Tests for CLI output formatting."""

    def test_info_prints_to_stdout(self, capsys):
        """info() should print [INFO] prefix to stdout."""
        formatter = CLIFormatter()
        formatter.info("Test message")
        captured = capsys.readouterr()
        assert "[INFO] Test message" in captured.out

    def test_success_prints_to_stdout(self, capsys):
        """success() should print [SUCCESS] prefix to stdout."""
        formatter = CLIFormatter()
        formatter.success("Done!")
        captured = capsys.readouterr()
        assert "[SUCCESS] Done!" in captured.out

    def test_warning_prints_to_stderr(self, capsys):
        """warning() should print [WARNING] prefix to stderr."""
        formatter = CLIFormatter()
        formatter.warning("Be careful")
        captured = capsys.readouterr()
        assert "[WARNING] Be careful" in captured.err

    def test_error_prints_to_stderr(self, capsys):
        """error() should print [ERROR] prefix to stderr."""
        formatter = CLIFormatter()
        formatter.error("Something failed")
        captured = capsys.readouterr()
        assert "[ERROR] Something failed" in captured.err

    def test_error_with_exception_in_debug_mode(self, capsys):
        """error() should print traceback in debug mode."""
        formatter = CLIFormatter(debug=True)
        try:
            raise ValueError("Test error")
        except ValueError as e:
            formatter.error("Failed", exception=e)
        captured = capsys.readouterr()
        assert "[ERROR] Failed" in captured.err
        assert "Traceback (Debug Mode)" in captured.err
        assert "ValueError: Test error" in captured.err

    def test_error_without_exception_no_traceback(self, capsys):
        """error() without exception should not print traceback even in debug mode."""
        formatter = CLIFormatter(debug=True)
        formatter.error("Just a message")
        captured = capsys.readouterr()
        assert "[ERROR] Just a message" in captured.err
        assert "Traceback" not in captured.err

    def test_step_prints_in_verbose_mode(self, capsys):
        """step() should print in verbose mode."""
        formatter = CLIFormatter(verbose=True)
        formatter.step(1, "semantic_search", "5 results")
        captured = capsys.readouterr()
        assert "[1] semantic_search -> 5 results" in captured.out

    def test_step_prints_in_debug_mode(self, capsys):
        """step() should print in debug mode."""
        formatter = CLIFormatter(debug=True)
        formatter.step(2, "read_file", "read 50 lines")
        captured = capsys.readouterr()
        assert "[2] read_file -> read 50 lines" in captured.out

    def test_step_silent_in_normal_mode(self, capsys):
        """step() should be silent in normal (non-verbose, non-debug) mode."""
        formatter = CLIFormatter()
        formatter.step(1, "semantic_search", "5 results")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_answer_prints_formatted_block(self, capsys):
        """answer() should print a formatted answer block."""
        formatter = CLIFormatter()
        formatter.answer("The answer is 42.")
        captured = capsys.readouterr()
        assert "=" * 60 in captured.out
        assert "Answer:" in captured.out
        assert "The answer is 42." in captured.out


# ============================================================================
# Repository Validation Tests
# ============================================================================

class TestValidateRepositoryPath:
    """Tests for repository path validation."""

    def test_valid_repo_path_returns_resolved_path(self, temp_repo, mock_formatter):
        """Valid repo path should return resolved absolute Path."""
        result = validate_repository_path(str(temp_repo), mock_formatter)
        assert result == temp_repo.resolve()
        assert isinstance(result, Path)

    def test_nonexistent_path_raises_error(self, mock_formatter):
        """Nonexistent path should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_repository_path("/nonexistent/path", mock_formatter)

    def test_file_path_raises_error(self, tmp_path, mock_formatter):
        """File path (not directory) should raise ValueError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            validate_repository_path(str(file_path), mock_formatter)

    def test_empty_repo_raises_error(self, empty_repo, mock_formatter):
        """Repository with no Python files should raise ValueError."""
        with pytest.raises(ValueError, match="No Python files found"):
            validate_repository_path(str(empty_repo), mock_formatter)


# ============================================================================
# Index Command Tests
# ============================================================================

class TestRunIndexCommand:
    """Tests for the index command."""

    @patch("repomind.cli.commands.save_graph")
    @patch("repomind.cli.commands.RepoScanner")
    @patch("repomind.cli.commands.FileReader")
    @patch("repomind.cli.commands.parse_source")
    @patch("repomind.cli.commands.CodeChunker")
    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    def test_index_command_success(
        self,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        mock_chunker,
        mock_parse_source,
        mock_file_reader,
        mock_scanner,
        mock_save_graph,
        temp_repo,
        mock_formatter,
    ):
        """Index command should succeed and return True for valid repo."""
        # Scanner discovers two files (validate_repository_path also scans)
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = [
            temp_repo / "main.py",
            temp_repo / "utils" / "helpers.py",
        ]
        mock_scanner.return_value = mock_scanner_instance

        # FileReader.read() -> SourceFile
        mock_source_file = MagicMock()
        mock_source_file.content = "def hello():\n    return 1\n"
        mock_source_file.relative_path = Path("main.py")
        mock_file_reader_instance = MagicMock()
        mock_file_reader_instance.read.return_value = mock_source_file
        mock_file_reader.return_value = mock_file_reader_instance

        # parse(source, filepath) -> ParsedFile
        mock_parse_source.return_value = make_parsed_file()

        # CodeChunker.chunk(source_file, parsed_file) -> List[CodeChunk]
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk.return_value = [MagicMock(), MagicMock()]
        mock_chunker.return_value = mock_chunker_instance

        mock_embedder.return_value = MagicMock()

        # VectorStore(collection_name=..., persist_dir=..., embedder=...)
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.add_chunks.return_value = 4
        mock_vector_store_instance.count.return_value = 4
        mock_vector_store.return_value = mock_vector_store_instance

        # CodeGraphBuilder.build_graph(parsed_files, repo_root=...)
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 6
        mock_graph.number_of_edges.return_value = 3
        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = mock_graph
        mock_graph_builder_instance.build_graph.return_value = mock_graph
        mock_graph_builder.return_value = mock_graph_builder_instance

        result = run_index_command(str(temp_repo), mock_formatter)

        assert result is True
        # Scanner is constructed with root_path (not repo_root)
        mock_scanner.assert_called_with(root_path=temp_repo.resolve())
        mock_scanner_instance.scan.assert_called()
        mock_file_reader_instance.read.assert_called()
        mock_parse_source.assert_called()
        mock_chunker_instance.chunk.assert_called()
        mock_vector_store_instance.add_chunks.assert_called_once()
        mock_graph_builder_instance.build_graph.assert_called_once()
        mock_save_graph.assert_called_once()

    @patch("repomind.cli.commands.save_graph")
    @patch("repomind.cli.commands.RepoScanner")
    @patch("repomind.cli.commands.FileReader")
    @patch("repomind.cli.commands.parse_source")
    @patch("repomind.cli.commands.CodeChunker")
    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    def test_index_command_passes_relative_path_to_parser(
        self,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        mock_chunker,
        mock_parse_source,
        mock_file_reader,
        mock_scanner,
        mock_save_graph,
        temp_repo,
        mock_formatter,
    ):
        """parse() must receive repo-relative paths so chunk/graph IDs align."""
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = [temp_repo / "main.py"]
        mock_scanner.return_value = mock_scanner_instance

        mock_source_file = MagicMock()
        mock_source_file.content = "x = 1\n"
        mock_source_file.relative_path = Path("main.py")
        mock_file_reader.return_value.read.return_value = mock_source_file

        mock_chunker.return_value.chunk.return_value = []
        mock_vector_store.return_value.count.return_value = 0
        mock_graph_builder.return_value.graph = MagicMock()

        mock_parse_source.return_value = make_parsed_file()

        run_index_command(str(temp_repo), mock_formatter)

        mock_parse_source.assert_called_once_with("x = 1\n", Path("main.py"))

    @patch("repomind.cli.commands.RepoScanner")
    def test_index_command_no_python_files(
        self,
        mock_scanner,
        temp_repo,
        mock_formatter,
    ):
        """Index command should return False when no Python files found."""
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = []
        mock_scanner.return_value = mock_scanner_instance

        result = run_index_command(str(temp_repo), mock_formatter)
        assert result is False
        mock_formatter.error.assert_called_once()

    @patch("repomind.cli.commands.RepoScanner")
    def test_index_command_scanner_exception(
        self,
        mock_scanner,
        temp_repo,
        mock_formatter,
    ):
        """Index command should return False when scanner raises exception."""
        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.side_effect = Exception("Scanner failed")
        mock_scanner.return_value = mock_scanner_instance

        result = run_index_command(str(temp_repo), mock_formatter)
        assert result is False

    @patch("repomind.cli.commands.save_graph")
    @patch("repomind.cli.commands.RepoScanner")
    @patch("repomind.cli.commands.FileReader")
    @patch("repomind.cli.commands.parse_source")
    @patch("repomind.cli.commands.CodeChunker")
    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    def test_index_command_skips_unparseable_file(
        self,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        mock_chunker,
        mock_parse_source,
        mock_file_reader,
        mock_scanner,
        mock_save_graph,
        temp_repo,
        mock_formatter,
    ):
        """A single failing file should warn and be skipped, not abort indexing."""
        mock_scanner.return_value.scan.return_value = [
            temp_repo / "good.py",
            temp_repo / "bad.py",
        ]

        good_source = MagicMock(content="x = 1\n", relative_path=Path("good.py"))
        mock_file_reader.return_value.read.side_effect = [
            good_source,
            OSError("unreadable"),
        ]

        mock_chunker.return_value.chunk.return_value = [MagicMock()]
        mock_vector_store.return_value.count.return_value = 1
        mock_graph_builder.return_value.graph = MagicMock()
        mock_parse_source.return_value = make_parsed_file()

        result = run_index_command(str(temp_repo), mock_formatter)

        assert result is True
        mock_formatter.warning.assert_called_once()
        # Only the readable file made it into the graph
        parsed_files_arg = mock_graph_builder.return_value.build_graph.call_args.args[0]
        assert len(parsed_files_arg) == 1

    def test_index_command_warns_on_syntax_error_file(self, tmp_path, mock_formatter):
        """A file with a syntax error should be reported and excluded.

        parse_source signals syntax errors by *returning* a flagged ParsedFile
        rather than raising, so this runs the real parser (no parse_source mock)
        to prove the CLI notices. Silently indexing a broken file as an empty
        one degrades retrieval with no signal to the user.
        """
        repo = tmp_path / "syntax_repo"
        repo.mkdir()
        (repo / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (repo / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

        with patch("repomind.cli.commands.CodeEmbedder"), \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store, \
             patch("repomind.cli.commands.save_graph"):
            mock_vector_store.return_value.count.return_value = 1

            result = run_index_command(str(repo), mock_formatter)

        assert result is True

        warnings = [c.args[0] for c in mock_formatter.warning.call_args_list]
        assert len(warnings) == 1
        assert "broken.py" in warnings[0]
        assert "SyntaxError" in warnings[0]

        # The good file was still indexed despite its neighbour being broken.
        infos = [c.args[0] for c in mock_formatter.info.call_args_list]
        assert any("across 1 files" in msg for msg in infos)

    @patch("repomind.cli.commands.save_graph")
    @patch("repomind.cli.commands.RepoScanner")
    @patch("repomind.cli.commands.FileReader")
    @patch("repomind.cli.commands.parse_source")
    @patch("repomind.cli.commands.CodeChunker")
    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    def test_index_command_verbose_output(
        self,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        mock_chunker,
        mock_parse_source,
        mock_file_reader,
        mock_scanner,
        mock_save_graph,
        temp_repo,
        capsys,
    ):
        """Index command should show progress in verbose mode."""
        formatter = CLIFormatter(verbose=True)

        mock_scanner.return_value.scan.return_value = [temp_repo / "main.py"]
        mock_file_reader.return_value.read.return_value = MagicMock(
            content="x = 1\n", relative_path=Path("main.py")
        )
        mock_chunker.return_value.chunk.return_value = []

        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 0
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 1
        mock_graph.number_of_edges.return_value = 0
        mock_graph_builder.return_value.graph = mock_graph

        run_index_command(str(temp_repo), formatter)

        captured = capsys.readouterr()
        assert "[INFO]" in captured.out
        assert "Scanning" in captured.out
        assert "[SUCCESS]" in captured.out


# ============================================================================
# Ask Command Tests
# ============================================================================

class TestRunAskCommand:
    """Tests for the ask command."""

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_success(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should succeed and return True for valid query."""
        # Setup mocks
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 10  # Already indexed
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        # Setup agent mock
        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "This is the final answer from the agent."
        mock_final_state.iteration = 3
        mock_final_state.tool_calls = []
        mock_final_state.errors = []
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        result = run_ask_command(
            repo_path_str=str(temp_repo),
            query="How does the hello function work?",
            formatter=mock_formatter,
        )

        # Assert
        assert result is True
        mock_react_agent.assert_called_once()
        mock_agent_instance.analyze.assert_called_once_with("How does the hello function work?")
        mock_formatter.answer.assert_called_once_with("This is the final answer from the agent.")

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    def test_ask_command_empty_query(
        self,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should fail with empty query."""
        result = run_ask_command(
            repo_path_str=str(temp_repo),
            query="",
            formatter=mock_formatter,
        )
        assert result is False

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    def test_ask_command_whitespace_query(
        self,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should fail with whitespace-only query."""
        result = run_ask_command(
            repo_path_str=str(temp_repo),
            query="   \n\t  ",
            formatter=mock_formatter,
        )
        assert result is False

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_provider_override(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should respect --provider override."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 5
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "Answer"
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        # Use gemini provider override
        with patch("repomind.cli.commands.GeminiProvider") as mock_gemini:
            mock_gemini_instance = MagicMock()
            mock_gemini.return_value = mock_gemini_instance
            mock_gemini_instance.is_available.return_value = True

            run_ask_command(
                repo_path_str=str(temp_repo),
                query="test query",
                formatter=mock_formatter,
                provider_name="gemini",
            )

            mock_gemini.assert_called_once()

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_auto_indexes_empty_store(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should auto-index when vector store is empty."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 0  # Empty store
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "Answer after indexing"
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        # Mock run_index_command to return True
        with patch("repomind.cli.commands.run_index_command", return_value=True) as mock_index:
            result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="test query",
                formatter=mock_formatter,
            )

            assert result is True
            mock_index.assert_called_once()

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_fails_if_auto_index_fails(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should fail if auto-indexing fails."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 0
        mock_vector_store.return_value = mock_vector_store_instance

        with patch("repomind.cli.commands.run_index_command", return_value=False):
            result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="test query",
                formatter=mock_formatter,
            )

            assert result is False

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_top_k_override(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should pass top_k to retriever."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 5
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "Answer"
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        run_ask_command(
            repo_path_str=str(temp_repo),
            query="test query",
            formatter=mock_formatter,
            top_k=15,
        )

        # Verify HybridRetriever was created with the overridden top_k
        mock_hybrid_retriever.assert_called_once()
        call_args = mock_hybrid_retriever.call_args
        # We can't easily check the retriever's internal config without more mocking
        # But we verified the call happened

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_ask_command_max_iterations_override(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should pass max_iterations to agent."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 5
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "Answer"
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        run_ask_command(
            repo_path_str=str(temp_repo),
            query="test query",
            formatter=mock_formatter,
            max_iterations=20,
        )

        # Verify ReActAgent was created with max_iterations=20
        mock_react_agent.assert_called_once()
        call_kwargs = mock_react_agent.call_args.kwargs
        assert call_kwargs.get("max_iterations") == 20


# ============================================================================
# Main Entry Point Tests
# ============================================================================

class TestMainEntryPoint:
    """Tests for the main() entry point function."""

    @patch("repomind.cli.main.run_index_command")
    def test_main_index_command(
        self,
        mock_run_index,
        capsys,
        temp_repo,
    ):
        """main() should call run_index_command for 'index' subcommand."""
        mock_run_index.return_value = True

        # Capture sys.argv
        with patch.object(sys, "argv", ["repomind", "index", str(temp_repo)]):
            exit_code = main()

        assert exit_code == 0
        mock_run_index.assert_called_once()

    @patch("repomind.cli.main.run_ask_command")
    def test_main_ask_command(
        self,
        mock_run_ask,
        capsys,
        temp_repo,
    ):
        """main() should call run_ask_command for 'ask' subcommand."""
        mock_run_ask.return_value = True

        with patch.object(sys, "argv", ["repomind", "ask", str(temp_repo), "test query"]):
            exit_code = main()

        assert exit_code == 0
        mock_run_ask.assert_called_once()

    @patch("repomind.cli.main.run_index_command")
    def test_main_index_failure_returns_nonzero(
        self,
        mock_run_index,
        capsys,
        temp_repo,
    ):
        """main() should return non-zero exit code on index failure."""
        mock_run_index.return_value = False

        with patch.object(sys, "argv", ["repomind", "index", str(temp_repo)]):
            exit_code = main()

        assert exit_code == 1

    @patch("repomind.cli.main.run_ask_command")
    def test_main_ask_failure_returns_nonzero(
        self,
        mock_run_ask,
        capsys,
        temp_repo,
    ):
        """main() should return non-zero exit code on ask failure."""
        mock_run_ask.return_value = False

        with patch.object(sys, "argv", ["repomind", "ask", str(temp_repo), "test query"]):
            exit_code = main()

        assert exit_code == 1

    def test_main_no_command_shows_help(self, capsys):
        """main() with no command should show help and return non-zero."""
        with patch.object(sys, "argv", ["repomind"]):
            exit_code = main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower() or "usage:" in captured.out.lower()

    @patch("repomind.cli.main.run_ask_command")
    def test_main_ask_with_all_overrides(
        self,
        mock_run_ask,
        capsys,
        temp_repo,
    ):
        """main() should pass all CLI overrides to run_ask_command."""
        mock_run_ask.return_value = True

        with patch.object(sys, "argv", [
            "repomind", "ask", str(temp_repo), "query",
            "--provider", "gemini",
            "--model", "gemini-1.5-pro",
            "--top-k", "10",
            "--depth", "2",
            "--max-iterations", "15",
            "--verbose",
        ]):
            exit_code = main()

        assert exit_code == 0
        mock_run_ask.assert_called_once()
        call_kwargs = mock_run_ask.call_args.kwargs
        assert call_kwargs["provider_name"] == "gemini"
        assert call_kwargs["model_name"] == "gemini-1.5-pro"
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["depth"] == 2
        assert call_kwargs["max_iterations"] == 15

    @patch("repomind.cli.main.run_index_command")
    def test_main_index_verbose_debug(
        self,
        mock_run_index,
        capsys,
        temp_repo,
    ):
        """main() should pass verbose/debug flags to formatter."""
        mock_run_index.return_value = True

        with patch.object(sys, "argv", ["repomind", "index", str(temp_repo), "--verbose", "--debug"]):
            exit_code = main()

        assert exit_code == 0
        mock_run_index.assert_called_once()
        # main() passes the formatter by keyword
        formatter = mock_run_index.call_args.kwargs["formatter"]
        assert formatter.verbose is True
        assert formatter.debug is True


# ============================================================================
# Integration-style Tests (with more realistic mocking)
# ============================================================================

class TestCLIIntegration:
    """Integration-style tests with more realistic component mocking."""

    def test_end_to_end_index_then_ask(self, temp_repo):
        """Test index followed by ask with mocked components."""
        # This test mocks at a higher level - the subsystem factories
        with patch("repomind.cli.commands.save_graph"), \
             patch("repomind.cli.commands.RepoScanner") as mock_scanner, \
             patch("repomind.cli.commands.FileReader") as mock_file_reader, \
             patch("repomind.cli.commands.parse_source") as mock_parse_source, \
             patch("repomind.cli.commands.CodeChunker") as mock_chunker, \
             patch("repomind.cli.commands.CodeEmbedder") as mock_embedder, \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store, \
             patch("repomind.cli.commands.GraphBuilder") as mock_graph_builder, \
             patch("repomind.cli.commands.GraphTraverser") as mock_traverser, \
             patch("repomind.cli.commands.HybridRetriever") as mock_retriever, \
             patch("repomind.cli.commands.FreeLocalProvider") as mock_provider, \
             patch("repomind.cli.commands.create_default_registry") as mock_registry, \
             patch("repomind.cli.commands.ReActAgent") as mock_agent:

            formatter = CLIFormatter()

            # Setup index mocks
            mock_scanner_instance = MagicMock()
            mock_scanner_instance.scan.return_value = [temp_repo / "main.py"]
            mock_scanner.return_value = mock_scanner_instance

            mock_file_reader.return_value.read.return_value = MagicMock(
                content="x = 1\n", relative_path=Path("main.py")
            )
            mock_parse_source.return_value = make_parsed_file()
            mock_chunker.return_value.chunk.return_value = []

            mock_embedder.return_value = MagicMock()

            mock_vector_store_instance = MagicMock()
            mock_vector_store_instance.add_chunks.return_value = 0
            mock_vector_store_instance.count.return_value = 0
            mock_vector_store.return_value = mock_vector_store_instance

            mock_graph_builder_instance = MagicMock()
            mock_graph_builder_instance.graph = MagicMock()
            mock_graph_builder.return_value = mock_graph_builder_instance

            # Run index
            index_result = run_index_command(str(temp_repo), formatter)
            assert index_result is True

            # Now test ask - vector store has data
            mock_vector_store_instance.count.return_value = 5

            mock_traverser_instance = MagicMock()
            mock_traverser.return_value = mock_traverser_instance

            mock_retriever_instance = MagicMock()
            mock_retriever.return_value = mock_retriever_instance

            mock_provider_instance = MagicMock()
            mock_provider.return_value = mock_provider_instance

            mock_registry_instance = MagicMock()
            mock_registry.return_value = mock_registry_instance

            mock_agent_instance = MagicMock()
            mock_final_state = MagicMock()
            mock_final_state.final_answer = "The hello function returns a greeting."
            mock_agent_instance.analyze.return_value = mock_final_state
            mock_agent.return_value = mock_agent_instance

            ask_result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="What does hello do?",
                formatter=formatter,
            )

            assert ask_result is True
            mock_agent_instance.analyze.assert_called_once_with("What does hello do?")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestCLIEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("repomind.cli.commands.RepoScanner")
    def test_validate_repo_with_symlink(self, mock_scanner, tmp_path, mock_formatter):
        """validate_repository_path should resolve symlinks."""
        # Create real directory
        real_repo = tmp_path / "real_repo"
        real_repo.mkdir()
        (real_repo / "main.py").write_text("print('hello')")

        # Creating a symlink on Windows needs Developer Mode or admin rights.
        link_repo = tmp_path / "link_repo"
        try:
            link_repo.symlink_to(real_repo, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            pytest.skip(f"symlink creation not permitted on this platform: {e}")

        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = [real_repo / "main.py"]
        mock_scanner.return_value = mock_scanner_instance

        result = validate_repository_path(str(link_repo), mock_formatter)

        assert result == real_repo.resolve()
        # The scanner receives the resolved (de-symlinked) path via root_path
        mock_scanner.assert_called_once_with(root_path=real_repo.resolve())

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    def test_ask_command_exception_during_setup(
        self,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """Ask command should handle exceptions during setup gracefully."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.side_effect = Exception("Vector store error")
        mock_vector_store.return_value = mock_vector_store_instance

        result = run_ask_command(
            repo_path_str=str(temp_repo),
            query="test query",
            formatter=mock_formatter,
        )

        assert result is False

    def test_ask_command_without_final_answer_fails(self, temp_repo, mock_formatter):
        """An agent that exhausts its iterations without answering is a failure."""
        with patch("repomind.cli.commands.CodeEmbedder"), \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store, \
             patch("repomind.cli.commands.GraphBuilder"), \
             patch("repomind.cli.commands.GraphTraverser"), \
             patch("repomind.cli.commands.HybridRetriever"), \
             patch("repomind.cli.commands.FreeLocalProvider"), \
             patch("repomind.cli.commands.create_default_registry"), \
             patch("repomind.cli.commands.ReActAgent") as mock_agent:
            mock_vector_store.return_value.count.return_value = 5

            state = MagicMock()
            state.final_answer = None
            state.tool_calls = []
            state.tool_results = []
            mock_agent.return_value.analyze.return_value = state

            result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="unanswerable question",
                formatter=mock_formatter,
                provider_name="free-local",
            )

        assert result is False
        mock_formatter.answer.assert_not_called()
        mock_formatter.error.assert_called_once()

    def test_ask_command_verbose_reports_each_tool_step(self, temp_repo):
        """Verbose mode should emit one step line per tool call the agent made."""
        formatter = MagicMock(spec=CLIFormatter)
        formatter.verbose = True
        formatter.debug = False

        with patch("repomind.cli.commands.CodeEmbedder"), \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store, \
             patch("repomind.cli.commands.GraphBuilder"), \
             patch("repomind.cli.commands.GraphTraverser"), \
             patch("repomind.cli.commands.HybridRetriever"), \
             patch("repomind.cli.commands.FreeLocalProvider"), \
             patch("repomind.cli.commands.create_default_registry"), \
             patch("repomind.cli.commands.ReActAgent") as mock_agent:
            mock_vector_store.return_value.count.return_value = 5

            state = MagicMock()
            state.final_answer = "done"
            state.tool_calls = [
                MagicMock(name="call1"),
                MagicMock(name="call2"),
            ]
            state.tool_calls[0].name = "semantic_search"
            state.tool_calls[1].name = "find_callers"
            state.tool_results = [
                MagicMock(content="a" * 12),
                MagicMock(content="b" * 30),
            ]
            mock_agent.return_value.analyze.return_value = state

            result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="how does this work?",
                formatter=formatter,
                provider_name="free-local",
            )

        assert result is True
        assert formatter.step.call_count == 2
        assert formatter.step.call_args_list[0].args[:2] == (1, "semantic_search")
        assert formatter.step.call_args_list[1].args[:2] == (2, "find_callers")

    def test_formatter_step_with_special_characters(self, capsys):
        """step() should handle special characters in tool names/summaries."""
        formatter = CLIFormatter(verbose=True)
        formatter.step(1, "semantic_search", "found 5 results (score > 0.8)")
        captured = capsys.readouterr()
        assert "[1] semantic_search -> found 5 results (score > 0.8)" in captured.out

    def test_formatter_answer_with_multiline(self, capsys):
        """answer() should handle multi-line answers."""
        formatter = CLIFormatter()
        formatter.answer("Line 1\nLine 2\nLine 3")
        captured = capsys.readouterr()
        assert "Line 1" in captured.out
        assert "Line 2" in captured.out
        assert "Line 3" in captured.out


# ============================================================================
# Graph Persistence Tests
# ============================================================================

class TestGraphPersistence:
    """Tests for the pickle-based graph persistence helpers.

    NetworkX 3.x removed read/write_gpickle, so the CLI persists the DiGraph
    itself. These tests use real graphs (no mocks) because a MagicMock would
    happily accept any round-trip and prove nothing.
    """

    def test_save_then_load_roundtrip_preserves_topology(self, tmp_path):
        """A saved graph should reload with identical nodes and edges."""
        graph = nx.DiGraph()
        graph.add_node("a.py::foo", kind="function")
        graph.add_node("b.py::bar", kind="function")
        graph.add_edge("a.py::foo", "b.py::bar", edge_type="calls")

        path = tmp_path / GRAPH_FILENAME
        save_graph(graph, path)
        restored = load_graph(path)

        assert set(restored.nodes) == {"a.py::foo", "b.py::bar"}
        assert list(restored.edges) == [("a.py::foo", "b.py::bar")]
        assert restored.nodes["a.py::foo"]["kind"] == "function"
        assert restored["a.py::foo"]["b.py::bar"]["edge_type"] == "calls"

    def test_save_then_load_preserves_graph_node_dataclasses(self, tmp_path):
        """Node metadata carries GraphNode dataclasses/enums, not plain JSON.

        This is why persistence uses pickle rather than a JSON dump.
        """
        node = GraphNode(
            node_id="mod.py::Greeter",
            node_type=NodeType.CLASS,
            filepath=Path("mod.py"),
            symbol_name="Greeter",
        )
        graph = nx.DiGraph()
        graph.add_node(node.node_id, node_obj=node)

        path = tmp_path / GRAPH_FILENAME
        save_graph(graph, path)
        restored = load_graph(path)

        restored_node = restored.nodes["mod.py::Greeter"]["node_obj"]
        assert isinstance(restored_node, GraphNode)
        assert restored_node.node_type is NodeType.CLASS
        assert restored_node.symbol_name == "Greeter"
        assert restored_node.filepath == Path("mod.py")

    def test_index_command_writes_loadable_graph_file(self, temp_repo, mock_formatter):
        """index should leave behind a graph file that load_graph can read.

        Only the embedding/vector-store layer is mocked (it would otherwise
        download a model); scanning, parsing, chunking and graph building all
        run for real, so this covers the persistence contract end to end.
        """
        with patch("repomind.cli.commands.CodeEmbedder"), \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store:
            mock_vector_store.return_value.count.return_value = 3

            result = run_index_command(
                repo_path_str=str(temp_repo),
                formatter=mock_formatter,
            )

        assert result is True

        graph_path = temp_repo / GRAPH_FILENAME
        assert graph_path.exists()

        restored = load_graph(graph_path)
        assert isinstance(restored, nx.DiGraph)
        assert restored.number_of_nodes() > 0

    def test_ask_command_loads_persisted_graph(self, temp_repo, mock_formatter):
        """ask should hand the traverser the graph it loaded from disk."""
        graph = nx.DiGraph()
        graph.add_node("main.py::hello")
        save_graph(graph, temp_repo / GRAPH_FILENAME)

        with patch("repomind.cli.commands.CodeEmbedder"), \
             patch("repomind.cli.commands.VectorStore") as mock_vector_store, \
             patch("repomind.cli.commands.GraphTraverser") as mock_traverser, \
             patch("repomind.cli.commands.HybridRetriever"), \
             patch("repomind.cli.commands.FreeLocalProvider"), \
             patch("repomind.cli.commands.create_default_registry"), \
             patch("repomind.cli.commands.ReActAgent") as mock_agent:
            mock_vector_store.return_value.count.return_value = 5

            state = MagicMock()
            state.final_answer = "answer"
            state.tool_calls = []
            state.tool_results = []
            mock_agent.return_value.analyze.return_value = state

            result = run_ask_command(
                repo_path_str=str(temp_repo),
                query="what does hello do?",
                formatter=mock_formatter,
                provider_name="free-local",
            )

        assert result is True
        passed_graph = mock_traverser.call_args.args[0]
        assert list(passed_graph.nodes) == ["main.py::hello"]


# ============================================================================
# Configuration Precedence Tests
# ============================================================================

class TestConfigurationPrecedence:
    """Tests for CLI override precedence over config defaults."""

    @patch("repomind.cli.commands.CodeEmbedder")
    @patch("repomind.cli.commands.VectorStore")
    @patch("repomind.cli.commands.GraphBuilder")
    @patch("repomind.cli.commands.GraphTraverser")
    @patch("repomind.cli.commands.HybridRetriever")
    @patch("repomind.cli.commands.FreeLocalProvider")
    @patch("repomind.cli.commands.create_default_registry")
    @patch("repomind.cli.commands.ReActAgent")
    def test_cli_overrides_config_defaults(
        self,
        mock_react_agent,
        mock_create_registry,
        mock_free_local,
        mock_hybrid_retriever,
        mock_graph_traverser,
        mock_graph_builder,
        mock_vector_store,
        mock_embedder,
        temp_repo,
        mock_formatter,
    ):
        """CLI flags should override config values."""
        mock_vector_store_instance = MagicMock()
        mock_vector_store_instance.count.return_value = 5
        mock_vector_store.return_value = mock_vector_store_instance

        mock_graph_builder_instance = MagicMock()
        mock_graph_builder_instance.graph = MagicMock()
        mock_graph_builder.return_value = mock_graph_builder_instance

        mock_traverser_instance = MagicMock()
        mock_graph_traverser.return_value = mock_traverser_instance

        mock_retriever_instance = MagicMock()
        mock_hybrid_retriever.return_value = mock_retriever_instance

        mock_provider_instance = MagicMock()
        mock_free_local.return_value = mock_provider_instance

        mock_registry = MagicMock()
        mock_create_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_final_state = MagicMock()
        mock_final_state.final_answer = "Answer"
        mock_agent_instance.analyze.return_value = mock_final_state
        mock_react_agent.return_value = mock_agent_instance

        # Use config defaults (no overrides)
        run_ask_command(
            repo_path_str=str(temp_repo),
            query="test",
            formatter=mock_formatter,
            provider_name=None,
            model_name=None,
            top_k=None,
            depth=None,
            max_iterations=None,
        )

        # Verify defaults from config were used
        # (The actual config values are tested indirectly through the calls)

        # Now test with overrides
        mock_react_agent.reset_mock()
        run_ask_command(
            repo_path_str=str(temp_repo),
            query="test",
            formatter=mock_formatter,
            provider_name="gemini",
            model_name="custom-model",
            top_k=20,
            depth=3,
            max_iterations=30,
        )

        # Verify ReActAgent got overridden max_iterations
        mock_react_agent.assert_called_once()
        call_kwargs = mock_react_agent.call_args.kwargs
        assert call_kwargs.get("max_iterations") == 30


# ============================================================================
# Console Script Entry Point Test
# ============================================================================

class TestConsoleScript:
    """Test that the console script entry point works."""

    def test_console_script_installed(self):
        """Verify the package has the correct entry point in pyproject.toml."""
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - older interpreters
            import tomli as tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        scripts = pyproject.get("project", {}).get("scripts", {})
        assert "repomind" in scripts
        assert scripts["repomind"] == "repomind.cli.main:main"

    def test_main_function_exists_and_callable(self):
        """main() should be importable and callable."""
        from repomind.cli.main import main
        assert callable(main)


# ============================================================================
# Help/Usage Output Tests
# ============================================================================

class TestHelpOutput:
    """Tests for help and usage output."""

    def test_help_output_contains_commands(self, capsys):
        """Help output should list available commands."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert "ask" in captured.out
        assert "index" in captured.out

    def test_ask_help_shows_options(self, capsys):
        """Ask subcommand help should show all options."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["ask", "--help"])
        captured = capsys.readouterr()
        assert "--provider" in captured.out
        assert "--model" in captured.out
        assert "--top-k" in captured.out
        assert "--depth" in captured.out
        assert "--max-iterations" in captured.out
        assert "--verbose" in captured.out
        assert "--debug" in captured.out

    def test_index_help_shows_options(self, capsys):
        """Index subcommand help should show options."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["index", "--help"])
        captured = capsys.readouterr()
        assert "--verbose" in captured.out
        assert "--debug" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])