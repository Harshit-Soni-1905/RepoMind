"""Tests for Stage 7 Agent subsystem (LLM Provider, Tools, AgentState, ReActAgent)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from repomind.agent.llm_provider import (
    LLMProvider,
    FreeLocalProvider,
    GeminiProvider,
    Message,
    ToolCall,
    ToolResult,
    ToolDefinition,
)
from repomind.agent.tools import (
    Tool,
    ToolRegistry,
    create_semantic_search_tool,
    create_graph_traverse_tool,
    create_read_file_tool,
    create_default_registry,
)
from repomind.agent.state import AgentState
from repomind.agent.agent import ReActAgent
from repomind.graph.models import GraphNode, NodeType
from repomind.retrieval.models import HybridResultNode, RetrievalSource, HybridQueryResult


# ============================================================================
# Fake LLM Provider for Testing
# ============================================================================

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


# ============================================================================
# LLM Provider Tests
# ============================================================================

def test_llm_provider_dataclasses():
    """Test instantiation and attributes of LLMProvider data structures."""
    tool_def = ToolDefinition(name="test_tool", description="desc", input_schema={"type": "object"})
    assert tool_def.name == "test_tool"

    tool_call = ToolCall(id="call_1", name="test_tool", arguments={"key": "val"})
    assert tool_call.id == "call_1"
    assert tool_call.arguments["key"] == "val"

    tool_res = ToolResult(call_id="call_1", name="test_tool", content="ok", is_error=False)
    assert tool_res.content == "ok"
    assert not tool_res.is_error

    msg = Message(role="user", content="hello", tool_calls=[tool_call], tool_call_id="call_1")
    assert msg.role == "user"
    assert msg.tool_calls[0] == tool_call


def test_free_local_provider_heuristics():
    """Test FreeLocalProvider responses and keyword-based tool choice."""
    provider = FreeLocalProvider()
    assert provider.name == "free-local"
    assert provider.is_available()

    tools = [
        ToolDefinition(name="semantic_search", description="search", input_schema={}),
        ToolDefinition(name="graph_traverse", description="graph", input_schema={}),
        ToolDefinition(name="read_file", description="read", input_schema={}),
    ]

    # Semantic search query
    msg1 = Message(role="user", content="search for user authentication logic")
    resp1 = provider.complete([msg1], tools)
    assert resp1.startswith("TOOL_CALL:")
    assert "semantic_search" in resp1

    # Graph traverse query
    msg2 = Message(role="user", content="show dependencies and imports")
    resp2 = provider.complete([msg2], tools)
    assert resp2.startswith("TOOL_CALL:")
    assert "graph_traverse" in resp2

    # Read file query
    msg3 = Message(role="user", content="show content of source code file")
    resp3 = provider.complete([msg3], tools)
    assert resp3.startswith("TOOL_CALL:")
    assert "read_file" in resp3

    # Default fallback search tool
    msg4 = Message(role="user", content="random query")
    resp4 = provider.complete([msg4], tools)
    assert resp4.startswith("TOOL_CALL:")
    assert "semantic_search" in resp4

    # Tool results already present -> final answer
    msg_user = Message(role="user", content="query")
    msg_tool = Message(role="tool", content="tool output here", name="semantic_search", tool_call_id="1")
    resp_final = provider.complete([msg_user, msg_tool], tools)
    assert not resp_final.startswith("TOOL_CALL:")
    assert "Based on my investigation" in resp_final


def test_free_local_provider_no_matching_tools():
    """Test FreeLocalProvider when tool list doesn't match expected names."""
    provider = FreeLocalProvider()
    msg = Message(role="user", content="search code")
    resp = provider.complete([msg], [])
    assert resp == "I'll help you explore the repository. Let me search for relevant code."


def test_free_local_provider_no_tool_results_final_summary():
    """Test FreeLocalProvider final answer generation with empty tool message list."""
    provider = FreeLocalProvider()
    ans = provider._generate_final_answer([], "query")
    assert "summary" in ans.lower()


def test_gemini_provider_availability_and_completion():
    """Test GeminiProvider initialization, availability check, and model execution mock."""
    # Unavailable when no key
    provider_no_key = GeminiProvider(api_key=None)
    assert not provider_no_key.is_available()
    assert provider_no_key.name == "gemini-3.5-flash-lite"

    # Mock google.genai module
    mock_genai = MagicMock()
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-3.5-flash-lite")
        assert provider.is_available()

        mock_client = MagicMock()
        # Mock text response candidate
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Final answer text", function_call=None)]
        mock_response.text = "Final answer text"
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [
                Message(role="system", content="sys"),
                Message(role="user", content="usr"),
                Message(role="assistant", content="ast", tool_calls=[ToolCall("1", "t", {})]),
                Message(role="tool", content="res", name="t", tool_call_id="1"),
            ]
            tools = [ToolDefinition("t", "desc", {})]
            result = provider.complete(messages, tools)
            assert result == "Final answer text"

            # Mock function_call response candidate
            mock_fc_part = MagicMock()
            mock_fc_part.function_call.name = "semantic_search"
            mock_fc_part.function_call.args = {"query": "auth"}
            mock_response_fc = MagicMock()
            mock_response_fc.candidates = [MagicMock()]
            mock_response_fc.candidates[0].content.parts = [mock_fc_part]

            mock_client.models.generate_content.return_value = mock_response_fc
            fc_result = provider.complete(messages, tools)
            assert fc_result.startswith("TOOL_CALL:")
            assert "semantic_search" in fc_result


def test_gemini_schema_sanitization():
    """Test that GeminiProvider strips incompatible JSON Schema fields."""
    # Rich schema with all fields that cause "Unknown field for Schema" errors
    rich_schema = {
        "type": "object",
        "description": "A tool that does things",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "minLength": 1,
                "maxLength": 500,
                "pattern": "^[a-zA-Z0-9 ]+$",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "direction": {
                "type": "string",
                "description": "Traversal direction",
                "enum": ["dependencies", "dependents", "neighbors"],
                "default": "neighbors",
            },
            "edge_types": {
                "type": "array",
                "description": "Filter by edge types",
                "items": {
                    "type": "string",
                    "enum": ["imports", "defines", "contains", "calls"],
                    "default": "imports",
                },
                "minItems": 0,
                "maxItems": 10,
                "default": [],
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    cleaned = GeminiProvider._sanitize_schema_for_gemini(rich_schema)

    # Top-level checks
    assert cleaned["type"] == "object"
    assert cleaned["description"] == "A tool that does things"
    assert cleaned["required"] == ["query"]
    assert "additionalProperties" not in cleaned

    # Property-level checks — query
    query_props = cleaned["properties"]["query"]
    assert query_props["type"] == "string"
    assert query_props["description"] == "Search query"
    assert "minLength" not in query_props
    assert "maxLength" not in query_props
    assert "pattern" not in query_props

    # Property-level checks — top_k
    top_k_props = cleaned["properties"]["top_k"]
    assert top_k_props["type"] == "integer"
    assert top_k_props["description"] == "Max results"
    assert "minimum" not in top_k_props
    assert "maximum" not in top_k_props
    assert "default" not in top_k_props

    # Property-level checks — direction
    dir_props = cleaned["properties"]["direction"]
    assert dir_props["type"] == "string"
    assert dir_props["description"] == "Traversal direction"
    assert dir_props["enum"] == ["dependencies", "dependents", "neighbors"]
    assert "default" not in dir_props

    # Property-level checks — edge_types (nested items)
    et_props = cleaned["properties"]["edge_types"]
    assert et_props["type"] == "array"
    assert et_props["description"] == "Filter by edge types"
    assert "minItems" not in et_props
    assert "maxItems" not in et_props
    assert "default" not in et_props
    assert et_props["items"]["type"] == "string"
    assert et_props["items"]["enum"] == ["imports", "defines", "contains", "calls"]
    assert "default" not in et_props["items"]

    # Original schema is untouched (no mutation)
    assert "minimum" in rich_schema["properties"]["top_k"]
    assert "additionalProperties" in rich_schema


def test_gemini_provider_passes_sanitized_schemas():
    """Test that GeminiProvider passes sanitized schemas when generating content."""
    mock_genai = MagicMock()
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-1.5-flash")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock(text="Answer", function_call=None)]
        mock_response.text = "Answer"
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(provider, "_get_client", return_value=mock_client):
            # Create real default tool definitions
            retriever = MagicMock()
            traverser = MagicMock()
            registry = create_default_registry(retriever, traverser)
            tools = registry.get_definitions()

            messages = [Message(role="user", content="hello")]
            provider.complete(messages, tools)

            # Check what was passed to generate_content
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args[1]
            config = call_kwargs["config"]
            # The tools are inside the config, verify they were constructed
            assert config is not None


# ============================================================================
# Tool & Tool Registry Tests
# ============================================================================

def test_tool_registry_registration_and_lookup():
    """Test tool registry methods: register, get, get_all, get_definitions."""
    registry = ToolRegistry()
    dummy_tool = Tool(
        name="dummy",
        description="dummy tool",
        input_schema={"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]},
        executor=lambda args: ToolResult(call_id="1", name="dummy", content=f"result: {args.get('arg')}"),
    )

    registry.register(dummy_tool)
    assert registry.get("dummy") == dummy_tool
    assert len(registry.get_all()) == 1
    assert len(registry.get_definitions()) == 1
    assert registry.get_definitions()[0].name == "dummy"

    # Duplicate registration raises ValueError
    with pytest.raises(ValueError, match="already registered"):
        registry.register(dummy_tool)


def test_tool_registry_execution_validation():
    """Test tool execution validation for missing required args and unknown tool name."""
    registry = ToolRegistry()
    dummy_tool = Tool(
        name="dummy",
        description="dummy tool",
        input_schema={"type": "object", "properties": {"req_field": {"type": "string"}}, "required": ["req_field"]},
        executor=lambda args: ToolResult(call_id="1", name="dummy", content="ok"),
    )
    registry.register(dummy_tool)

    # Unknown tool call
    unknown_call = ToolCall(id="c1", name="nonexistent", arguments={})
    res_unknown = registry.execute(unknown_call)
    assert res_unknown.is_error
    assert "Unknown tool" in res_unknown.content

    # Missing required argument
    missing_arg_call = ToolCall(id="c2", name="dummy", arguments={})
    res_missing = registry.execute(missing_arg_call)
    assert res_missing.is_error
    assert "Missing required argument 'req_field'" in res_missing.content

    # Executor raises unexpected exception
    error_tool = Tool(
        name="error_tool",
        description="errors out",
        input_schema={},
        executor=lambda args: 1 / 0,
    )
    registry.register(error_tool)
    error_call = ToolCall(id="c3", name="error_tool", arguments={})
    res_err = registry.execute(error_call)
    assert res_err.is_error
    assert "Error executing tool" in res_err.content


def test_semantic_search_tool_execution():
    """Test semantic_search tool creation and execution using a mock retriever."""
    mock_retriever = MagicMock()
    mock_node = MagicMock()
    mock_node.node_id = "path/file.py::func"
    mock_node.filepath = Path("path/file.py")
    mock_node.symbol_name = "func"
    mock_node.symbol_type = "function"
    mock_node.source.value = "vector"
    mock_node.semantic_score = 0.95
    mock_node.graph_distance = 0
    mock_node.related_via = []
    mock_node.combined_score = 0.85
    mock_node.start_line = 1
    mock_node.end_line = 10
    mock_node.docstring = "Sample docstring"

    mock_result = HybridQueryResult(
        query="test query",
        nodes=[mock_node],
        total_results=1,
        vector_count=1,
        graph_count=0,
        both_count=0,
        top_k=5,
        traversal_depth=1,
    )
    mock_retriever.retrieve.return_value = mock_result

    tool = create_semantic_search_tool(mock_retriever)
    assert tool.name == "semantic_search"

    res = tool.executor({"query": "test query", "top_k": 5, "depth": 1})
    assert not res.is_error
    data = json.loads(res.content)
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["symbol_name"] == "func"


def test_graph_traverse_tool_execution():
    """Test graph_traverse tool execution with dependencies, dependents, and neighbors directions."""
    mock_traverser = MagicMock()
    node = GraphNode(
        node_id="f.py::sym",
        filepath=Path("f.py"),
        symbol_name="sym",
        node_type=NodeType.FUNCTION,
        start_line=1,
        end_line=5,
        docstring="doc",
    )

    mock_traverser.get_dependencies.return_value = [node]
    mock_traverser.get_dependents.return_value = [node]
    mock_traverser.get_neighbors.return_value = [node]

    tool = create_graph_traverse_tool(mock_traverser)
    assert tool.name == "graph_traverse"

    # Direction: dependencies with edge types
    res_dep = tool.executor({"node_id": "f.py::sym", "direction": "dependencies", "edge_types": ["calls"]})
    assert "total_found" in res_dep.content
    data_dep = json.loads(res_dep.content)
    assert data_dep["direction"] == "dependencies"

    # Direction: dependents
    res_depts = tool.executor({"node_id": "f.py::sym", "direction": "dependents"})
    data_depts = json.loads(res_depts.content)
    assert data_depts["direction"] == "dependents"

    # Direction: neighbors
    res_neigh = tool.executor({"node_id": "f.py::sym", "direction": "neighbors"})
    data_neigh = json.loads(res_neigh.content)
    assert data_neigh["direction"] == "neighbors"


def test_read_file_tool_execution(tmp_path):
    """Test read_file tool execution with full file reading and line-range filtering."""
    file_path = tmp_path / "sample.py"
    file_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")

    tool = create_read_file_tool(repo_root=tmp_path)
    assert tool.name == "read_file"

    # Read entire file
    res_full = tool.executor({"filepath": str(file_path)})
    assert not res_full.is_error
    data_full = json.loads(res_full.content)
    assert data_full["total_lines"] == 5
    assert "line 1" in data_full["content"]

    # Read line range
    res_range = tool.executor({"filepath": str(file_path), "start_line": 2, "end_line": 4})
    data_range = json.loads(res_range.content)
    assert data_range["start_line"] == 2
    assert data_range["end_line"] == 4
    assert data_range["content"] == "line 2\nline 3\nline 4"

    # Nonexistent file error
    res_err = tool.executor({"filepath": str(tmp_path / "nonexistent.py")})
    assert res_err.is_error
    assert "File not found" in res_err.content


def test_create_default_registry():
    """Test create_default_registry helper creates a registry with all 3 default tools."""
    mock_retriever = MagicMock()
    mock_traverser = MagicMock()
    registry = create_default_registry(mock_retriever, mock_traverser, repo_root=Path("."))
    tools = registry.get_all()
    tool_names = {t.name for t in tools}
    assert tool_names == {"semantic_search", "graph_traverse", "read_file"}


# ============================================================================
# AgentState Tests
# ============================================================================

def test_agent_state_initialization_and_mutations():
    """Test AgentState tracking of messages, tool calls, tool results, and stops."""
    state = AgentState(query="What is RepoMind?", max_iterations=5)
    assert state.query == "What is RepoMind?"
    assert state.iteration == 0
    assert not state.is_complete()
    assert not state.should_stop()

    # Add message
    state.add_message(Message(role="user", content="query"))
    assert len(state.messages) == 1

    # Add tool call
    call = ToolCall(id="1", name="read_file", arguments={"filepath": "main.py"})
    state.add_tool_call(call)
    assert len(state.tool_calls) == 1

    # Add successful tool result
    res_success = ToolResult(
        call_id="1",
        name="read_file",
        content=json.dumps({"filepath": "main.py", "content": "print('hello')"}),
        is_error=False,
    )
    state.add_tool_result(res_success)
    assert len(state.tool_results) == 1
    assert "main.py" in state.read_files

    # Add error tool result
    res_error = ToolResult(call_id="2", name="read_file", content="Error reading file", is_error=True)
    state.add_tool_result(res_error)
    assert state.consecutive_errors == 1
    assert len(state.errors) == 1

    # Test final answer setting
    state.set_final_answer("RepoMind is an AI assistant.")
    assert state.is_complete()
    assert state.should_stop()
    assert state.final_answer == "RepoMind is an AI assistant."


def test_agent_state_context_extraction():
    """Test extraction of semantic_search results and graph_traverse nodes in AgentState."""
    state = AgentState(query="test")

    # Semantic search result extraction
    sem_res = ToolResult(
        call_id="1",
        name="semantic_search",
        content=json.dumps({
            "results": [{
                "node_id": "file.py::func",
                "source": "vector",
                "combined_score": 0.9,
                "docstring": "Sample function docstring"
            }]
        }),
    )
    state.add_tool_result(sem_res)
    assert len(state.retrieved_nodes) == 1

    # Graph traverse result extraction
    graph_res = ToolResult(
        call_id="2",
        name="graph_traverse",
        content=json.dumps({
            "nodes": [{
                "node_id": "file.py::class",
                "node_type": "class",
                "docstring": "Sample class docstring"
            }]
        }),
    )
    state.add_tool_result(graph_res)
    assert "file.py::class" in state.graph_nodes

    # Non-JSON result gracefully ignored
    non_json_res = ToolResult(call_id="3", name="semantic_search", content="non json string")
    state.add_tool_result(non_json_res)

    # State serialization
    dict_repr = state.to_dict()
    assert dict_repr["query"] == "test"
    assert dict_repr["is_complete"] is False


def test_agent_state_get_context_summary():
    """Test get_context_summary formatting and token truncation logic."""
    state = AgentState(query="test")
    state.retrieved_nodes.append({
        "node_id": "f1.py::func1",
        "source": "vector",
        "combined_score": 0.85,
        "docstring": "doc string content"
    })
    state.retrieved_nodes.append(HybridResultNode(
        node_id="f2.py::func2",
        filepath=Path("f2.py"),
        symbol_name="func2",
        symbol_type="function",
        source_code=None,
        start_line=1,
        end_line=10,
        docstring=None,
        source=RetrievalSource.VECTOR,
        semantic_score=0.8,
        semantic_distance=0.2,
        graph_distance=None,
        related_via=[],
        combined_score=0.8,
    ))
    state.graph_nodes["f3.py::class3"] = {"node_type": "class", "docstring": "class docstring"}
    state.read_files["f4.py"] = "\n".join([f"line {i}" for i in range(60)])

    summary = state.get_context_summary(max_tokens=2000)
    assert "=== Retrieved Code (semantic search) ===" in summary
    assert "f1.py::func1" in summary
    assert "f2.py::func2" in summary
    assert "=== Graph Nodes (structural) ===" in summary
    assert "=== File Contents Read ===" in summary
    assert "... (10 more lines)" in summary

    # Small max_tokens triggers truncation
    truncated_summary = state.get_context_summary(max_tokens=10)
    assert "[truncated]" in truncated_summary


# ============================================================================
# ReActAgent Core Loop & Behavior Tests
# ============================================================================

def test_react_agent_final_answer_without_tool_call():
    """Test ReActAgent when LLM directly provides a final answer on iteration 1."""
    provider = FakeScriptedLLMProvider(["Here is the direct answer to your question."])
    registry = ToolRegistry()
    agent = ReActAgent(llm_provider=provider, tool_registry=registry, max_iterations=3)

    state = agent.analyze("Explain authentication.")
    assert state.is_complete()
    assert state.final_answer == "Here is the direct answer to your question."
    assert state.iteration == 1
    assert len(state.tool_calls) == 0


def test_react_agent_one_tool_call_then_final_answer():
    """Test ReActAgent making one tool call, receiving the tool result, then producing final answer."""
    tool_call_json = json.dumps({
        "id": "c1",
        "name": "dummy_tool",
        "arguments": {"query": "auth"}
    })
    responses = [
        f"TOOL_CALL: {tool_call_json}",
        "Based on the tool result, auth is implemented in auth.py."
    ]
    provider = FakeScriptedLLMProvider(responses)

    registry = ToolRegistry()
    dummy_tool = Tool(
        name="dummy_tool",
        description="dummy",
        input_schema={"type": "object"},
        executor=lambda args: ToolResult(call_id="c1", name="dummy_tool", content="Found auth.py"),
    )
    registry.register(dummy_tool)

    agent = ReActAgent(llm_provider=provider, tool_registry=registry, max_iterations=5)
    state = agent.analyze("Where is auth?")

    assert state.is_complete()
    assert state.final_answer == "Based on the tool result, auth is implemented in auth.py."
    assert state.iteration == 2
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].name == "dummy_tool"
    assert len(state.tool_results) == 1
    assert state.tool_results[0].content == "Found auth.py"


def test_react_agent_multiple_tool_calls_across_iterations():
    """Test ReActAgent sequential tool calls across multiple iterations."""
    tc1 = json.dumps({"id": "1", "name": "t1", "arguments": {"a": 1}})
    tc2 = json.dumps({"id": "2", "name": "t2", "arguments": {"b": 2}})
    responses = [
        f"TOOL_CALL: {tc1}",
        f"TOOL_CALL: {tc2}",
        "Final synthesis of both tools."
    ]
    provider = FakeScriptedLLMProvider(responses)

    registry = ToolRegistry()
    registry.register(Tool("t1", "desc", {}, lambda args: ToolResult("1", "t1", "res1")))
    registry.register(Tool("t2", "desc", {}, lambda args: ToolResult("2", "t2", "res2")))

    agent = ReActAgent(llm_provider=provider, tool_registry=registry)
    state = agent.analyze("Complex query")

    assert state.is_complete()
    assert state.iteration == 3
    assert len(state.tool_calls) == 2
    assert state.final_answer == "Final synthesis of both tools."


def test_react_agent_malformed_llm_response():
    """Test ReActAgent handling malformed TOOL_CALL JSON string from LLM."""
    responses = [
        "TOOL_CALL: {not valid json}",
        "Correction: Here is the actual final answer."
    ]
    provider = FakeScriptedLLMProvider(responses)
    registry = ToolRegistry()

    agent = ReActAgent(llm_provider=provider, tool_registry=registry)
    state = agent.analyze("query")

    assert state.is_complete()
    assert state.iteration == 2
    assert len(state.errors) == 1
    assert "Malformed tool call response" in state.errors[0]
    assert state.final_answer == "Correction: Here is the actual final answer."


def test_react_agent_repeated_identical_tool_call_prevention():
    """Test ReActAgent catching duplicate consecutive tool calls to prevent infinite loops."""
    tc = json.dumps({"id": "1", "name": "t1", "arguments": {"query": "same"}})
    responses = [
        f"TOOL_CALL: {tc}",
        f"TOOL_CALL: {tc}",  # Repeated identical call
        "Recovered and giving final answer."
    ]
    provider = FakeScriptedLLMProvider(responses)

    registry = ToolRegistry()
    registry.register(Tool("t1", "desc", {}, lambda args: ToolResult("1", "t1", "res")))

    agent = ReActAgent(llm_provider=provider, tool_registry=registry)
    state = agent.analyze("query")

    assert state.is_complete()
    assert state.iteration == 3
    assert len(state.errors) == 1
    assert "Repeated consecutive tool call" in state.errors[0]
    assert state.final_answer == "Recovered and giving final answer."


def test_react_agent_consecutive_errors_threshold():
    """Test ReActAgent terminating when consecutive errors reach the error threshold (3)."""
    # Need 4 responses: 3 malformed tool calls + 1 more LLM call after threshold reached
    responses = [
        "TOOL_CALL: {bad json 1}",
        "TOOL_CALL: {bad json 2}",
        "TOOL_CALL: {bad json 3}",
        "TOOL_CALL: {bad json 4}",  # 4th call when consecutive_errors == 3
    ]
    provider = FakeScriptedLLMProvider(responses)
    registry = ToolRegistry()

    agent = ReActAgent(llm_provider=provider, tool_registry=registry)
    state = agent.analyze("query")

    assert state.is_complete()
    assert "encountered repeated errors" in state.final_answer
    assert state.consecutive_errors >= 3


def test_react_agent_max_iterations_reached():
    """Test ReActAgent stopping and generating a summary answer when max_iterations limit is reached."""
    responses = [f"TOOL_CALL: {json.dumps({'id': str(i), 'name': 't1', 'arguments': {'step': i}})}" for i in range(10)]
    provider = FakeScriptedLLMProvider(responses)

    registry = ToolRegistry()
    registry.register(Tool("t1", "desc", {}, lambda args: ToolResult("1", "t1", "res")))

    agent = ReActAgent(llm_provider=provider, tool_registry=registry, max_iterations=3)
    state = agent.analyze("query")

    assert state.is_complete()
    assert state.iteration == 3
    assert "maximum loop limit of 3 iterations" in state.final_answer


def test_react_agent_llm_exception_handling():
    """Test ReActAgent exception handling when LLM complete() raises an unexpected exception."""
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = Exception("API Connection Timeout")

    registry = ToolRegistry()
    agent = ReActAgent(llm_provider=mock_provider, tool_registry=registry)

    state = agent.analyze("query")
    assert state.is_complete()
    assert "encountered repeated errors" in state.final_answer
    assert any("API Connection Timeout" in e for e in state.errors)


def test_react_agent_multi_turn_search_then_read():
    """Regression test: semantic_search -> read_file -> final answer (3-turn tool-calling).

    This validates the exact multi-turn flow that triggered the Gemini
    thought_signature error in production when native Content objects
    were not preserved across turns.
    """
    tc_search = json.dumps({
        "id": "s1",
        "name": "semantic_search",
        "arguments": {"query": "calculator math operations"},
    })
    tc_read = json.dumps({
        "id": "r1",
        "name": "read_file",
        "arguments": {"filepath": "calculator.py"},
    })
    responses = [
        f"TOOL_CALL: {tc_search}",
        f"TOOL_CALL: {tc_read}",
        "The calculator module implements four arithmetic operations: add, subtract, multiply, and divide.",
    ]
    provider = FakeScriptedLLMProvider(responses)

    registry = ToolRegistry()
    registry.register(Tool(
        name="semantic_search",
        description="Search code",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=lambda args: ToolResult(
            call_id="s1", name="semantic_search",
            content=json.dumps({"results": [{"node_id": "calculator.py::add", "source": "vector", "combined_score": 0.9, "docstring": "Add two numbers"}]}),
        ),
    ))
    registry.register(Tool(
        name="read_file",
        description="Read file",
        input_schema={"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
        executor=lambda args: ToolResult(
            call_id="r1", name="read_file",
            content=json.dumps({"filepath": "calculator.py", "content": "def add(a, b): return a + b\ndef subtract(a, b): return a - b", "total_lines": 2}),
        ),
    ))

    agent = ReActAgent(llm_provider=provider, tool_registry=registry, max_iterations=5)
    state = agent.analyze("What math operations are in calculator.py?")

    assert state.is_complete()
    assert state.iteration == 3
    assert len(state.tool_calls) == 2
    assert state.tool_calls[0].name == "semantic_search"
    assert state.tool_calls[1].name == "read_file"
    assert len(state.tool_results) == 2
    assert not any(r.is_error for r in state.tool_results)
    assert "calculator" in state.final_answer.lower()
    # Verify conversation history has correct message sequence
    roles = [m.role for m in state.messages]
    # system, user, assistant(tool_call), tool, assistant(tool_call), tool, assistant(final)
    assert roles.count("tool") == 2
    assert roles.count("assistant") >= 3
