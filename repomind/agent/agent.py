"""ReAct Agent implementation for RepoMind.

This module orchestrates the ReAct (Reasoning and Action) loop, coordinating
the LLM provider, tool execution, state updates, and stop conditions.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging

from repomind.config import config
from repomind.agent.llm_provider import LLMProvider, Message, ToolCall, ToolResult
from repomind.agent.tools import ToolRegistry
from repomind.agent.state import AgentState

logger = logging.getLogger(__name__)


class ReActAgent:
    """Orchestrates ReAct agent loop for codebase understanding."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        max_iterations: Optional[int] = None,
    ):
        """Initialize the ReAct agent.

        Args:
            llm_provider: Implementation of LLMProvider
            tool_registry: Registry of tools the agent can use
            max_iterations: Max loop iterations (defaults to config.AGENT_MAX_ITERATIONS)
        """
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations or config.AGENT_MAX_ITERATIONS

    def analyze(self, query: str) -> AgentState:
        """Run the ReAct loop to analyze the query.

        Args:
            query: Repository question/query

        Returns:
            Final AgentState instance containing the answer, tools used, etc.
        """
        # 1. Initialize state
        state = AgentState(
            query=query,
            max_iterations=self.max_iterations,
        )

        # 2. Add system prompt with instructions and tool descriptions
        system_prompt = self._build_system_prompt()
        state.add_message(Message(role="system", content=system_prompt))

        # 3. Add user query
        state.add_message(Message(role="user", content=query))

        # 4. Start the loop
        while not state.should_stop():
            state.increment_iteration()
            logger.info(
                f"Agent Iteration {state.iteration}/{state.max_iterations} for query: {query}"
            )

            try:
                # Ask LLM for the next step (Thought + Action or Thought + Final Answer)
                response_text = self.llm_provider.complete(
                    messages=state.messages,
                    tools=[t.to_definition() for t in self.tool_registry.get_all()],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )

                # Parse LLM response (checks for tool calls or final answer)
                self._process_llm_response(response_text, state)

            except Exception as e:
                error_msg = f"LLM error during generation: {str(e)}"
                logger.error(error_msg)
                state.consecutive_errors += 1
                state.errors.append(error_msg)

                # Stop if consecutive errors exceed threshold (e.g., 3)
                if state.consecutive_errors >= 3:
                    state.set_final_answer(
                        "I encountered repeated errors while calling the language model backend. "
                        "Please verify your connection and config."
                    )
                    break

        # 5. Handle case where iteration limit was reached
        if not state.is_complete():
            state.set_final_answer(
                f"I have reached the maximum loop limit of {self.max_iterations} iterations "
                "without reaching a final answer. Here is the summary of what I retrieved:\n\n"
                f"{state.get_context_summary(1000)}"
            )

        return state

    def _build_system_prompt(self) -> str:
        """Build the system prompt containing instructions and tool guidelines."""
        tools_desc = []
        for tool in self.tool_registry.get_all():
            tools_desc.append(
                f"- Name: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Schema: {json.dumps(tool.input_schema)}\n"
            )
        tools_str = "\n".join(tools_desc)

        return (
            "You are RepoMind, an AI codebase-understanding assistant. Your goal is to answer developer "
            "questions about the repository code accurately by retrieving and examining code structure and files.\n\n"
            "You have access to the following tools to explore the codebase:\n"
            f"{tools_str}\n"
            "Use the ReAct pattern (Reasoning and Action) to solve the task:\n"
            "1. Analyze the current state and what information you still need.\n"
            "2. Think about what tool to call and with what arguments.\n"
            "3. Issue a tool call using the function-calling format.\n"
            "4. Observe the tool result.\n"
            "5. Repeat until you have enough context to answer the question.\n"
            "6. Once you have enough context, respond with your final answer.\n\n"
            "Response Formatting Guidelines:\n"
            "- If you need to run a tool, respond with a TOOL_CALL prefix followed by the tool call JSON, e.g.:\n"
            "  TOOL_CALL: {\"id\": \"call_id\", \"name\": \"semantic_search\", \"arguments\": {\"query\": \"user auth\"}}\n"
            "- If you have all the information needed, do not prefix with TOOL_CALL. Simply write your final answer directly.\n"
            "- Keep your final answer comprehensive and trace code elements by their filepath and line numbers where appropriate.\n"
            "- Be concise and precise."
        )

    def _process_llm_response(self, response_text: str, state: AgentState) -> None:
        """Parse LLM response for tool calls or final answer, and execute tools.

        Args:
            response_text: Text output from LLM
            state: Mutable AgentState to update
        """
        # Parse for tool call
        if response_text.startswith("TOOL_CALL:"):
            # Strip prefix and parse json
            try:
                call_json = response_text[len("TOOL_CALL:"):].strip()
                call_data = json.loads(call_json)

                tool_call = ToolCall(
                    id=call_data.get("id", "call_id"),
                    name=call_data.get("name"),
                    arguments=call_data.get("arguments", {}),
                )

                # Check for duplicate consecutive tool calls to prevent loops
                if state.tool_calls and state.tool_calls[-1].name == tool_call.name and state.tool_calls[-1].arguments == tool_call.arguments:
                    raise ValueError(
                        f"Repeated consecutive tool call to '{tool_call.name}' with same arguments. "
                        "Terminating to avoid infinite loop."
                    )

                # Record the tool call
                state.add_tool_call(tool_call)
                # Add assistant message containing the tool call to history
                state.add_message(Message(
                    role="assistant",
                    content=response_text,
                    tool_calls=[tool_call],
                ))

                # Execute tool
                logger.info(f"Executing tool '{tool_call.name}' with args {tool_call.arguments}")
                tool_result = self.tool_registry.execute(tool_call)

                # Record the result
                state.add_tool_result(tool_result)
                # Add tool result message to history
                state.add_message(Message(
                    role="tool",
                    name=tool_call.name,
                    content=tool_result.content,
                    tool_call_id=tool_call.id
                ))

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"Malformed tool call response: {str(e)}"
                logger.error(error_msg)
                state.consecutive_errors += 1
                state.errors.append(error_msg)

                # Stop if consecutive errors exceed threshold (e.g., 3)
                if state.consecutive_errors >= 3:
                    state.set_final_answer(
                        "I encountered repeated errors while parsing tool calls. "
                        "Please verify your response format."
                    )
                    return

                # Report error back to LLM so it can retry
                state.add_message(Message(
                    role="assistant",
                    content=response_text
                ))
                state.add_message(Message(
                    role="user",
                    content=f"Error: Could not parse or validate tool call: {str(e)}. Please correct your format."
                ))
        else:
            # LLM didn't request a tool call -> Treat it as the Final Answer
            # Add message to history
            state.add_message(Message(
                role="assistant",
                content=response_text
            ))
            # Mark status as completed
            state.set_final_answer(response_text)