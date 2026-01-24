"""Agent state schema for LangGraph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.agent.modes import AgentMode


class RouteDecision(Enum):
    """Routing decisions for the agent."""

    RETRIEVE = "retrieve"
    TOOL = "tool"
    GENERATE = "generate"
    REWRITE = "rewrite"
    FALLBACK = "fallback"
    WEB = "web"
    IMAGE = "image"


@dataclass
class ToolCall:
    """Represents a tool call request."""

    tool_name: str
    arguments: dict[str, Any]
    requires_approval: bool = False


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    output: str
    success: bool
    error: str | None = None


class AgentState:
    """
    State schema for the LangGraph agent.

    Attributes:
        messages: Conversation history (auto-accumulated)
        query: Current user query
        documents: Retrieved documents
        context: Formatted context for generation
        route: Current routing decision
        rewrite_count: Number of query rewrites attempted
        tool_calls: Pending tool calls
        tool_results: Results from tool executions
        response: Final generated response
        provider_used: Which LLM provider was used
        should_update_kb: Whether to update knowledge base
        metadata: Additional state metadata
        mode: Current agent mode
    """

    def __init__(
        self,
        messages: list[BaseMessage] | None = None,
        query: str = "",
        documents: list[Document] | None = None,
        context: str = "",
        route: RouteDecision | None = None,
        rewrite_count: int = 0,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        response: str = "",
        provider_used: str = "",
        should_update_kb: bool = False,
        metadata: dict[str, Any] | None = None,
        mode: AgentMode = AgentMode.CHAT,
    ) -> None:
        self.messages = messages or []
        self.query = query
        self.documents = documents or []
        self.context = context
        self.route = route
        self.rewrite_count = rewrite_count
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.response = response
        self.provider_used = provider_used
        self.should_update_kb = should_update_kb
        self.metadata = metadata or {}
        self.mode = mode


# TypedDict version for LangGraph compatibility
from typing import TypedDict


class AgentStateDict(TypedDict, total=False):
    """TypedDict version of AgentState for LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[Document]
    context: str
    route: str | None
    rewrite_count: int
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    response: str
    provider_used: str
    should_update_kb: bool
    metadata: dict[str, Any]
    mode: str  # AgentMode value
    tokens_used: int
    input_tokens: int
    output_tokens: int


def create_initial_state(
    query: str,
    messages: list[BaseMessage] | None = None,
    mode: AgentMode | str = AgentMode.CHAT,
) -> AgentStateDict:
    """Create initial state for a new agent run."""
    mode_value = mode.value if isinstance(mode, AgentMode) else mode
    return AgentStateDict(
        messages=messages or [],
        query=query,
        documents=[],
        context="",
        route=None,
        rewrite_count=0,
        tool_calls=[],
        tool_results=[],
        response="",
        provider_used="",
        should_update_kb=False,
        metadata={},
        mode=mode_value,
    )
