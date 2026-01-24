"""LangGraph state machine for the agent workflow."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.modes import AgentMode
from src.agent.nodes import (
    execute_tools,
    generate_fallback,
    generate_image,
    generate_local,
    get_grading_decision,
    get_routing_decision,
    get_selector,
    grade_documents,
    prepare_tool_calls,
    retrieve_documents,
    rewrite_query,
    route_query,
    should_update_kb,
    update_knowledge_base,
    web_search,
)
from src.agent.state import AgentStateDict, create_initial_state

logger = logging.getLogger(__name__)


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph agent workflow.

    The workflow follows this pattern:
    1. Route query to appropriate handler
    2. If retrieval needed: retrieve -> grade -> generate or rewrite/fallback
    3. If tool needed: prepare tools -> execute -> generate
    4. If direct generate: generate response
    5. If external fallback used: optionally update KB

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create the graph with our state schema
    workflow = StateGraph(AgentStateDict)

    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_local)
    workflow.add_node("fallback", generate_fallback)
    workflow.add_node("prepare_tools", prepare_tool_calls)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("update_kb", update_knowledge_base)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_image", generate_image)

    # Set entry point
    workflow.set_entry_point("route")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "route",
        get_routing_decision,
        {
            "retrieve": "retrieve",
            "prepare_tools": "prepare_tools",
            "generate": "generate",
            "web_search": "web_search",
            "generate_image": "generate_image",
        },
    )

    # Retrieve -> Grade
    workflow.add_edge("retrieve", "grade")

    # Conditional edges from grader
    workflow.add_conditional_edges(
        "grade",
        get_grading_decision,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    # Rewrite -> Retrieve (loop back)
    workflow.add_edge("rewrite", "retrieve")

    # Tool execution flow
    workflow.add_edge("prepare_tools", "execute_tools")

    # Conditional edge from execute_tools: explicit commands go to END, others to generate
    def should_skip_generation(state: AgentStateDict) -> str:
        """Check if we should skip generation (for explicit !commands)."""
        if state.get("route") == "end":
            return "end"
        return "generate"

    workflow.add_conditional_edges(
        "execute_tools",
        should_skip_generation,
        {
            "generate": "generate",
            "end": END,
        },
    )

    # Generation completion - check if KB update needed
    workflow.add_conditional_edges(
        "generate",
        should_update_kb,
        {
            "update_kb": "update_kb",
            "end": END,
        },
    )

    # Fallback completion - check if KB update needed
    workflow.add_conditional_edges(
        "fallback",
        should_update_kb,
        {
            "update_kb": "update_kb",
            "end": END,
        },
    )

    # KB update -> end
    workflow.add_edge("update_kb", END)

    # Web search -> end
    workflow.add_edge("web_search", END)

    # Image generation -> end
    workflow.add_edge("generate_image", END)

    return workflow.compile()


class Agent:
    """High-level agent interface wrapping the LangGraph workflow."""

    def __init__(self) -> None:
        """Initialize the agent with compiled graph."""
        self._graph = create_agent_graph()
        self._conversation_history: list = []
        self._current_mode: AgentMode = AgentMode.CHAT

    def query(
        self,
        query: str,
        include_history: bool = True,
    ) -> dict[str, Any]:
        """
        Process a query through the agent workflow.

        Args:
            query: The user's question
            include_history: Whether to include conversation history

        Returns:
            Dict with response and metadata
        """
        # Create initial state with current mode
        messages = self._conversation_history if include_history else []
        initial_state = create_initial_state(query, messages, mode=self._current_mode)

        # Run the graph
        logger.info(f"Processing query [{self._current_mode.value}]: {query[:50]}...")
        final_state = self._graph.invoke(initial_state)

        # Update conversation history
        if include_history and final_state.get("messages"):
            self._conversation_history = final_state["messages"]

        return {
            "response": final_state.get("response", ""),
            "provider": final_state.get("provider_used", ""),
            "documents_used": len(final_state.get("documents", [])),
            "tool_results": final_state.get("tool_results", []),
            "mode": self._current_mode.value,
            "tokens_used": final_state.get("tokens_used", 0),
            "input_tokens": final_state.get("input_tokens", 0),
            "output_tokens": final_state.get("output_tokens", 0),
        }

    async def aquery(
        self,
        query: str,
        include_history: bool = True,
    ) -> dict[str, Any]:
        """
        Async version of query.

        Args:
            query: The user's question
            include_history: Whether to include conversation history

        Returns:
            Dict with response and metadata
        """
        messages = self._conversation_history if include_history else []
        initial_state = create_initial_state(query, messages, mode=self._current_mode)

        logger.info(f"Processing query (async) [{self._current_mode.value}]: {query[:50]}...")
        final_state = await self._graph.ainvoke(initial_state)

        if include_history and final_state.get("messages"):
            self._conversation_history = final_state["messages"]

        return {
            "response": final_state.get("response", ""),
            "provider": final_state.get("provider_used", ""),
            "documents_used": len(final_state.get("documents", [])),
            "tool_results": final_state.get("tool_results", []),
            "mode": self._current_mode.value,
            "tokens_used": final_state.get("tokens_used", 0),
            "input_tokens": final_state.get("input_tokens", 0),
            "output_tokens": final_state.get("output_tokens", 0),
        }

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []

    def get_history_length(self) -> int:
        """Get number of messages in history."""
        return len(self._conversation_history)

    def set_model(self, model_name: str) -> None:
        """Switch the local LLM to a different Ollama model.

        Args:
            model_name: Name of the model to use (e.g., 'mistral:7b', 'deepseek-r1:7b')
        """
        selector = get_selector()
        selector.set_local_model(model_name)

    def get_model(self) -> str:
        """Get the current local model name."""
        selector = get_selector()
        return selector.get_current_model()

    def list_models(self) -> list[str]:
        """List all available Ollama models."""
        selector = get_selector()
        return selector.list_models()

    def get_mode(self) -> AgentMode:
        """Get the current agent mode."""
        return self._current_mode

    def set_mode(self, mode: AgentMode | str) -> None:
        """Set the agent mode.

        Args:
            mode: Mode to switch to (AgentMode enum or string name)
        """
        if isinstance(mode, str):
            try:
                mode = AgentMode(mode.lower())
            except ValueError:
                raise ValueError(f"Unknown mode: {mode}")
        self._current_mode = mode

    def cycle_mode(self) -> AgentMode:
        """Cycle to the next mode and return it."""
        from src.agent.modes import get_next_mode
        self._current_mode = get_next_mode(self._current_mode)
        return self._current_mode

    def set_context_window(self, num_ctx: int) -> None:
        """Set the context window size.

        Args:
            num_ctx: Context window size in tokens
        """
        selector = get_selector()
        selector._local_llm.set_context_window(num_ctx)

    def get_context_window(self) -> int:
        """Get the current context window size.

        Returns:
            Context window size in tokens
        """
        selector = get_selector()
        return selector._local_llm.get_context_window()


# Module-level instance for convenience
_default_agent: Agent | None = None


def get_agent() -> Agent:
    """Get or create the default agent instance."""
    global _default_agent
    if _default_agent is None:
        _default_agent = Agent()
    return _default_agent
