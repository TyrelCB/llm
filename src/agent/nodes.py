"""Workflow node implementations for LangGraph."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import settings
from src.agent.state import AgentStateDict, RouteDecision
from src.knowledge import DocumentGrader, KnowledgeBaseUpdater, Retriever, VectorStore
from src.llm import ProviderSelector

logger = logging.getLogger(__name__)


# Initialize shared components
_vectorstore = VectorStore()
_retriever = Retriever(_vectorstore)
_grader = DocumentGrader()
_selector = ProviderSelector()
_updater = KnowledgeBaseUpdater(_vectorstore)


ROUTER_SYSTEM_PROMPT = """You are a query router. Analyze the user's query and decide the best action.

Respond with ONLY one of these exact words:
- "retrieve" - Query asks about specific topics, projects, people, APIs, documentation, or anything that might be in a knowledge base. PREFER THIS for any informational query. Examples:
  * Questions about specific projects, products, or systems (e.g., "tell me about ACME", "what is Project X")
  * Questions about APIs, endpoints, configurations
  * Questions about people, teams, or organizations
  * Questions about documentation, procedures, or how things work
  * Any query mentioning specific names, acronyms, or proper nouns
- "tool" - Query requires executing a system command or checking system state. Examples:
  * Checking disk space, memory, CPU usage
  * Listing files or directories
  * Getting system information (OS, hostname, etc.)
  * Running any bash/shell command
  * File operations (reading, writing, searching files on disk)
  * Any query about "this system", "this machine", "my computer"
- "generate" - ONLY for queries that clearly don't need KB lookup:
  * Simple greetings ("hello", "hi", "thanks")
  * Generic coding help ("how do I write a for loop in Python")
  * Math or logic questions ("what is 2+2")
  * Requests about the conversation itself ("summarize our chat")

When in doubt, choose "retrieve" - it's better to check the KB and find nothing than to miss relevant information.

Do not include any other text."""


REWRITE_SYSTEM_PROMPT = """You are a query rewriter. Your task is to reformulate a query to improve retrieval results.

Given the original query and the fact that initial retrieval didn't find relevant documents, rewrite the query to:
1. Use different keywords or synonyms
2. Be more specific or more general as appropriate
3. Break down complex queries into simpler terms

Respond with ONLY the rewritten query, nothing else."""


COMMAND_GENERATOR_PROMPT = """You are a Linux command generator. Given a user's natural language request, generate the appropriate bash command(s) to fulfill it.

Rules:
1. Output ONLY the bash command(s), nothing else
2. Use standard Linux commands that work on most distributions
3. For multiple commands, separate them with && or ;
4. Prefer informative output (e.g., use -h for human-readable sizes)
5. Keep commands safe - avoid destructive operations unless explicitly requested
6. If the request is ambiguous, choose the most common interpretation

Examples:
- "how much disk space" -> "df -h"
- "what OS am I running" -> "uname -a && cat /etc/os-release"
- "list files in current directory" -> "ls -la"
- "show memory usage" -> "free -h"
- "what processes are running" -> "ps aux | head -20"
- "show network interfaces" -> "ip addr"
- "what's my IP address" -> "hostname -I"
- "show CPU info" -> "lscpu | head -20"

Respond with ONLY the command, no explanations or markdown."""


def route_query(state: AgentStateDict) -> AgentStateDict:
    """
    Route the query to the appropriate handler.

    Determines whether to retrieve documents, execute tools, or generate directly.
    """
    query = state["query"]

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}"),
    ]

    try:
        result = _selector.generate(messages, force_local=True)
        response = result.content.strip().lower()

        if "tool" in response:
            route = RouteDecision.TOOL.value
        elif "generate" in response:
            route = RouteDecision.GENERATE.value
        else:
            route = RouteDecision.RETRIEVE.value

        logger.info(f"Query routed to: {route}")

    except Exception as e:
        logger.warning(f"Router failed, defaulting to retrieve: {e}")
        route = RouteDecision.RETRIEVE.value

    return {"route": route}


def retrieve_documents(state: AgentStateDict) -> AgentStateDict:
    """
    Retrieve relevant documents from the knowledge base.
    """
    query = state["query"]

    documents = _retriever.retrieve(
        query=query,
        k=settings.retriever_k,
        min_score=settings.relevance_threshold * 0.5,  # Lower threshold initially
    )

    logger.info(f"Retrieved {len(documents)} documents for query")

    return {"documents": documents}


def grade_documents(state: AgentStateDict) -> AgentStateDict:
    """
    Grade retrieved documents for relevance.

    Updates state with relevant documents and determines next route.
    """
    query = state["query"]
    documents = state.get("documents", [])
    rewrite_count = state.get("rewrite_count", 0)

    if not documents:
        # No documents found
        if rewrite_count < settings.max_rewrite_attempts:
            return {"route": RouteDecision.REWRITE.value}
        else:
            return {"route": RouteDecision.FALLBACK.value}

    # Grade documents
    relevant_docs = _grader.filter_relevant(
        documents=documents,
        query=query,
        include_partial=True,
    )

    if not relevant_docs:
        if rewrite_count < settings.max_rewrite_attempts:
            return {"route": RouteDecision.REWRITE.value, "documents": []}
        else:
            return {"route": RouteDecision.FALLBACK.value, "documents": []}

    # Format context from relevant documents
    context = _retriever.format_context(relevant_docs)

    logger.info(f"Graded {len(relevant_docs)} relevant documents")

    return {
        "documents": relevant_docs,
        "context": context,
        "route": RouteDecision.GENERATE.value,
    }


def rewrite_query(state: AgentStateDict) -> AgentStateDict:
    """
    Rewrite the query to improve retrieval.
    """
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original query: {original_query}\n\n"
            f"This is rewrite attempt {rewrite_count + 1}. "
            "Please reformulate the query."
        ),
    ]

    try:
        result = _selector.generate(messages, force_local=True)
        new_query = result.content.strip()

        # Remove quotes if present
        new_query = new_query.strip('"\'')

        logger.info(f"Query rewritten: '{original_query}' -> '{new_query}'")

    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
        new_query = original_query

    return {
        "query": new_query,
        "rewrite_count": rewrite_count + 1,
        "route": RouteDecision.RETRIEVE.value,
    }


def generate_local(state: AgentStateDict) -> AgentStateDict:
    """
    Generate response using local LLM with context.
    """
    query = state["query"]
    context = state.get("context", "")
    messages = state.get("messages", [])

    result = _selector.generate_with_context(
        query=query,
        context=context if context else "No specific context available.",
        force_local=True,
        conversation_history=messages if messages else None,
    )

    # Add to conversation history
    new_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=result.content),
    ]

    logger.info(f"Generated local response (confidence: {result.confidence:.2f})")

    return {
        "response": result.content,
        "provider_used": result.provider,
        "messages": new_messages,
        "should_update_kb": False,
    }


def generate_fallback(state: AgentStateDict) -> AgentStateDict:
    """
    Generate response using external provider fallback.
    """
    query = state["query"]
    messages = state.get("messages", [])

    # Try to generate with fallback
    try:
        result = _selector.generate_with_context(
            query=query,
            context="No relevant local documents found. Please provide a comprehensive answer.",
            force_local=False,
            conversation_history=messages if messages else None,
        )

        provider_used = result.provider
        should_update = provider_used != "ollama"

        logger.info(f"Generated fallback response from {provider_used}")

    except Exception as e:
        logger.error(f"Fallback generation failed: {e}")
        result_content = (
            "I apologize, but I couldn't find relevant information in the knowledge base "
            "and external providers are unavailable. Please try rephrasing your question "
            "or check if the relevant documents have been ingested."
        )
        provider_used = "none"
        should_update = False

        return {
            "response": result_content,
            "provider_used": provider_used,
            "should_update_kb": False,
        }

    new_messages = messages + [
        HumanMessage(content=query),
        AIMessage(content=result.content),
    ]

    return {
        "response": result.content,
        "provider_used": provider_used,
        "messages": new_messages,
        "should_update_kb": should_update,
    }


def update_knowledge_base(state: AgentStateDict) -> AgentStateDict:
    """
    Update knowledge base with information from external response.
    """
    query = state["query"]
    response = state.get("response", "")
    provider = state.get("provider_used", "")

    if not state.get("should_update_kb", False):
        return {}

    if _updater.should_update_kb(query, response, confidence=0.9):
        added_count = _updater.update_from_response(
            query=query,
            response=response,
            provider=provider,
        )
        logger.info(f"Added {added_count} facts to knowledge base from {provider}")
    else:
        logger.debug("Skipping KB update - response not suitable")

    return {"should_update_kb": False}


def prepare_tool_calls(state: AgentStateDict) -> AgentStateDict:
    """
    Use LLM to generate appropriate bash command for the user's query.
    """
    query = state["query"]

    # Use LLM to generate the appropriate command
    messages = [
        SystemMessage(content=COMMAND_GENERATOR_PROMPT),
        HumanMessage(content=query),
    ]

    try:
        result = _selector.generate(messages, force_local=True)
        command = result.content.strip()

        # Clean up the command (remove markdown code blocks if present)
        if command.startswith("```"):
            lines = command.split("\n")
            command = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Remove any leading/trailing quotes
        command = command.strip('"\'`')

        if command:
            logger.info(f"Generated command for query '{query}': {command}")
            tool_calls = [{
                "tool_name": "bash",
                "arguments": {"command": command},
                "requires_approval": True,
            }]
            return {"tool_calls": tool_calls}

    except Exception as e:
        logger.warning(f"Command generation failed: {e}")

    # Fallback to generate if command generation fails
    return {"route": RouteDecision.GENERATE.value, "tool_calls": []}


def execute_tools(state: AgentStateDict) -> AgentStateDict:
    """
    Execute pending tool calls.

    Note: Actual tool execution is handled in tools module.
    This node orchestrates the calls and collects results.
    """
    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        return {"route": RouteDecision.GENERATE.value}

    # Import here to avoid circular dependency
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()
    results = []

    for call in tool_calls:
        tool_name = call.get("tool_name", "")
        arguments = call.get("arguments", {})

        try:
            output = registry.execute(tool_name, arguments)
            results.append({
                "tool_name": tool_name,
                "output": output,
                "success": True,
                "error": None,
            })
        except Exception as e:
            results.append({
                "tool_name": tool_name,
                "output": "",
                "success": False,
                "error": str(e),
            })

    # Format tool results as context
    context_parts = []
    for result in results:
        if result["success"]:
            context_parts.append(
                f"Tool '{result['tool_name']}' output:\n{result['output']}"
            )
        else:
            context_parts.append(
                f"Tool '{result['tool_name']}' failed: {result['error']}"
            )

    context = "\n\n".join(context_parts)

    return {
        "tool_results": results,
        "context": context,
        "route": RouteDecision.GENERATE.value,
    }


def get_routing_decision(state: AgentStateDict) -> str:
    """
    Conditional edge function to determine next node.

    Returns the name of the next node based on current route.
    """
    route = state.get("route")

    if route == RouteDecision.RETRIEVE.value:
        return "retrieve"
    elif route == RouteDecision.TOOL.value:
        return "prepare_tools"
    elif route == RouteDecision.GENERATE.value:
        return "generate"
    elif route == RouteDecision.REWRITE.value:
        return "rewrite"
    elif route == RouteDecision.FALLBACK.value:
        return "fallback"
    else:
        return "generate"


def get_grading_decision(state: AgentStateDict) -> str:
    """
    Conditional edge function after grading.
    """
    route = state.get("route")

    if route == RouteDecision.REWRITE.value:
        return "rewrite"
    elif route == RouteDecision.FALLBACK.value:
        return "fallback"
    else:
        return "generate"


def should_update_kb(state: AgentStateDict) -> str:
    """
    Conditional edge to check if KB should be updated.
    """
    if state.get("should_update_kb", False):
        return "update_kb"
    return "end"


def get_selector() -> ProviderSelector:
    """Get the shared ProviderSelector instance.

    This allows external code to access model switching functionality.
    """
    return _selector
