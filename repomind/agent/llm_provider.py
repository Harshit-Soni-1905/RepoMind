"""LLM Provider abstraction for RepoMind agent.

This module defines the interface that all LLM providers must implement,
allowing the agent to work with different LLM backends without modification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json


@dataclass(frozen=True)
class ToolDefinition:
    """Represents a tool that can be called by the LLM."""
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema for tool arguments


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call request from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """Represents a message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool result messages
    name: Optional[str] = None  # Optional tool or entity name


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete implementations should handle communication with specific
    LLM APIs (Gemini, OpenAI, Anthropic, local models, etc.)
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a completion given conversation history and available tools.

        Args:
            messages: Conversation history including system prompt, user messages,
                     assistant responses, and tool results
            tools: List of available tool definitions
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Assistant's response text (may include tool calls in a specific format)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured correctly."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for logging/debugging."""
        pass


class FreeLocalProvider(LLMProvider):
    """Free/local LLM provider using a simple pattern-matching approach.

    This provider implements a basic ReAct pattern without requiring any
    external API. It uses keyword matching and simple heuristics to decide
    which tools to call. This is intended as a zero-cost fallback for
    development and testing.

    For production use, replace with a real LLM provider (Gemini, etc.)
    """

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "free-local"

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response using simple heuristics.

        This is a placeholder implementation that demonstrates the ReAct
        pattern without requiring an actual LLM. It looks at the latest
        user message and decides which tool to call based on keywords.
        """
        self._call_count += 1

        # Get the latest user message
        latest_user_msg = ""
        for msg in reversed(messages):
            if msg.role == "user":
                latest_user_msg = msg.content.lower()
                break

        # Check if we have tool results to incorporate
        has_tool_results = any(msg.role == "tool" for msg in messages)

        # If we already have tool results, provide a final answer
        if has_tool_results:
            return self._generate_final_answer(messages, latest_user_msg)

        # Decide which tool to call based on keywords
        tool_choice = self._choose_tool(latest_user_msg, tools)

        if tool_choice:
            return self._format_tool_call(tool_choice, latest_user_msg)

        # Default: provide a generic answer
        return "I'll help you explore the repository. Let me search for relevant code."

    def _choose_tool(self, query: str, tools: List[ToolDefinition]) -> Optional[str]:
        """Choose a tool based on query keywords."""
        query = query.lower()

        # Semantic search keywords
        if any(kw in query for kw in ["search", "find", "look for", "what is", "how does", "explain", "where"]):
            for tool in tools:
                if tool.name == "semantic_search":
                    return "semantic_search"

        # Graph traversal keywords
        if any(kw in query for kw in ["depend", "import", "call", "relationship", "connected", "structure"]):
            for tool in tools:
                if tool.name == "graph_traverse":
                    return "graph_traverse"

        # File reading keywords
        if any(kw in query for kw in ["read", "show", "view", "content", "source code", "file"]):
            for tool in tools:
                if tool.name == "read_file":
                    return "read_file"

        # Default to semantic search
        for tool in tools:
            if tool.name == "semantic_search":
                return "semantic_search"

        return None

    def _format_tool_call(self, tool_name: str, query: str) -> str:
        """Format a tool call in a parseable way."""
        import uuid
        call_id = str(uuid.uuid4())[:8]

        # Extract a reasonable query for the tool
        if tool_name == "semantic_search":
            args = {"query": query, "top_k": 5}
        elif tool_name == "graph_traverse":
            args = {"node_id": "", "depth": 1}
        elif tool_name == "read_file":
            args = {"filepath": "", "start_line": None, "end_line": None}
        else:
            args = {}

        tool_call = {
            "id": call_id,
            "name": tool_name,
            "arguments": args
        }

        return f"TOOL_CALL: {json.dumps(tool_call)}"

    def _generate_final_answer(self, messages: List[Message], query: str) -> str:
        """Generate a final answer incorporating tool results."""
        # Collect tool results
        tool_results = []
        for msg in messages:
            if msg.role == "tool":
                tool_results.append(f"Tool '{msg.content}' returned: {msg.content[:200]}...")

        if not tool_results:
            return "I've gathered some information. Let me provide a summary based on what I found."

        return f"Based on my investigation, here's what I found:\n\n" + "\n".join(tool_results[:3])


class GeminiProvider(LLMProvider):
    """Google Gemini API provider using the official google-genai SDK.

    Requires GEMINI_API_KEY environment variable.
    Preserves native Gemini Content objects (including thought signatures)
    across multi-turn tool-calling conversations.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
        # Cache native Content objects keyed by tool-call ID so that
        # thought signatures survive across the ReAct loop turns.
        self._content_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        if self.model_name.startswith("gemini-"):
            return self.model_name
        return f"gemini-{self.model_name}"

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_client(self):
        """Lazy-load the google-genai Client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # Fields accepted by Gemini's Schema (OpenAPI-ish subset).
    # The google-genai SDK handles most of these natively via
    # types.Schema, but we still sanitize when building from dicts
    # to avoid unsupported fields that the REST API rejects.
    _GEMINI_SUPPORTED_SCHEMA_FIELDS = frozenset({
        "type", "description", "enum", "items", "properties",
        "required", "format", "nullable",
    })

    @classmethod
    def _sanitize_schema_for_gemini(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Strip JSON Schema fields that Gemini's function-calling API rejects.

        Gemini supports a small subset of OpenAPI 3.0 Schema fields.
        Standard JSON Schema validation fields (minimum, maximum, default,
        minLength, maxLength, pattern, additionalProperties, etc.) cause
        ``Unknown field for Schema`` errors and must be removed before the
        schema is sent to the API.

        The original schema dict is not mutated — a cleaned copy is returned.
        Nested ``properties`` and ``items`` are recursively sanitized.
        """
        cleaned: Dict[str, Any] = {}
        for key, value in schema.items():
            if key not in cls._GEMINI_SUPPORTED_SCHEMA_FIELDS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    prop_name: cls._sanitize_schema_for_gemini(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                cleaned[key] = cls._sanitize_schema_for_gemini(value)
            else:
                cleaned[key] = value
        return cleaned

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion using google-genai SDK with function calling.

        Builds a contents list of ``types.Content`` objects.  For assistant
        turns that contained a tool call, the *original* ``types.Content``
        returned by the API is re-used verbatim so that opaque thought
        signatures are never stripped.
        """
        from google.genai import types
        import uuid

        client = self._get_client()

        system_instructions: List[str] = []
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_instructions.append(msg.content)
            elif msg.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
            elif msg.role == "assistant":
                if msg.tool_calls:
                    # Re-use the cached native Content that preserves
                    # thought signatures emitted by Gemini 2.0/3.x models.
                    cached_content = None
                    for tc in msg.tool_calls:
                        cached_content = self._content_cache.get(tc.id)
                        if cached_content is not None:
                            break

                    if cached_content is not None:
                        contents.append(cached_content)
                    else:
                        # Fallback: reconstruct (may lose thought signatures
                        # on Gemini 3.x but works for Gemini 2.x and tests).
                        parts = []
                        if msg.content and not msg.content.startswith("TOOL_CALL:"):
                            parts.append(types.Part.from_text(text=msg.content))
                        for tc in msg.tool_calls:
                            parts.append(
                                types.Part.from_function_call(
                                    name=tc.name, args=tc.arguments,
                                )
                            )
                        contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg.content)],
                        )
                    )
            elif msg.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name or "tool",
                                response={"result": msg.content},
                            )
                        ],
                    )
                )

        # Build function declarations from the provider-agnostic ToolDefinitions.
        function_declarations = []
        for tool in tools:
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=self._sanitize_schema_for_gemini(tool.input_schema),
                )
            )

        tools_list = (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else None
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=(
                "\n\n".join(system_instructions) if system_instructions else None
            ),
            tools=tools_list,
        )

        import time

        # Try configured model first; if 503/high-demand error occurs, fallback to 3.1-flash-lite
        models_to_try = [self.model_name]
        if self.model_name == "gemini-3.5-flash-lite":
            models_to_try.append("gemini-3.1-flash-lite")

        max_attempts = 3
        last_error = None
        response = None

        for model in models_to_try:
            for attempt in range(max_attempts):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # Retry on transient rate limit / backoff
                    if attempt < max_attempts - 1 and any(
                        code in err_str for code in ("429", "ResourceExhausted")
                    ):
                        backoff = 2 ** (attempt + 1)
                        time.sleep(backoff)
                        continue
                    # On 503 / UNAVAILABLE / high demand, break out to try fallback model if available
                    if any(code in err_str for code in ("503", "UNAVAILABLE", "high demand")):
                        break
            if response is not None:
                break

        if response is None:
            if last_error is not None:
                raise last_error
            return "I don't have a response."

        # Check for function calls in the response
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            candidate_content = response.candidates[0].content
            for part in candidate_content.parts:
                if part.function_call:
                    fc = part.function_call
                    call_id = str(uuid.uuid4())[:8]
                    # Cache the complete native Content so thought signatures
                    # survive when this turn is replayed in future requests.
                    self._content_cache[call_id] = candidate_content
                    args = dict(fc.args) if fc.args else {}
                    return f"TOOL_CALL: {json.dumps({'id': call_id, 'name': fc.name, 'arguments': args})}"

        return response.text or "I don't have a response."