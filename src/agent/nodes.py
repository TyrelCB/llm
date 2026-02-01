"""Workflow node implementations for LangGraph."""

import logging
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import settings
from src.agent.modes import AgentMode, get_mode_config
from src.agent.state import AgentStateDict, RouteDecision
if TYPE_CHECKING:
    from src.knowledge import DocumentGrader, KnowledgeBaseUpdater, Retriever, VectorStore
    from src.llm import ProviderSelector

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_vectorstore() -> "VectorStore":
    from src.knowledge.vectorstore import VectorStore

    return VectorStore()


@lru_cache(maxsize=1)
def _get_retriever() -> "Retriever":
    from src.knowledge.retriever import Retriever

    return Retriever(_get_vectorstore())


@lru_cache(maxsize=1)
def _get_grader() -> "DocumentGrader":
    from src.knowledge.grader import DocumentGrader

    return DocumentGrader()


@lru_cache(maxsize=1)
def _get_selector() -> "ProviderSelector":
    from src.llm.selector import ProviderSelector

    return ProviderSelector()


@lru_cache(maxsize=1)
def _get_updater() -> "KnowledgeBaseUpdater":
    from src.knowledge.updater import KnowledgeBaseUpdater

    return KnowledgeBaseUpdater(_get_vectorstore())


ROUTER_SYSTEM_PROMPT = """You are a query router. Analyze the user's query and conversation history to decide the best action.

Respond with ONLY one of these exact words:
- "tool" - User explicitly wants to EXECUTE a system command. Choose this ONLY when:
  * User explicitly says "run this command: <command>"
  * Query starts with action verbs + specific system operations: "check disk space", "list running processes", "show network connections"
  * User references previous shell output and asks to run related commands
  * IMPORTANT: Generic questions about technologies, frameworks, or best practices are NOT tool requests
  * Example tool requests: "check if docker is running", "list files in /var/log", "kill process 1234"
  * Example NOT tool: "how to use docker", "flutter best practices", "explain kubernetes"
- "generate" - Use this for knowledge/explanation requests:
  * User asks "what is X?", "how does X work?", "explain Y"
  * Questions about programming concepts, frameworks, languages, best practices
  * Asking for guides, tutorials, recommendations, or advice
  * Generic coding help ("how do I write a for loop", "flutter best practices")
  * Simple greetings ("hello", "hi", "thanks")
  * Math or logic questions ("what is 2+2")
  * Requests about the conversation itself ("summarize our chat")
- "retrieve" - Query asks about INTERNAL/PROPRIETARY topics in a private knowledge base:
  * Questions about specific internal projects (e.g., "tell me about ACME project")
  * Questions about internal APIs, proprietary configurations
  * Questions about people, teams, or organizations in your company

CRITICAL: Default to "generate" unless the user explicitly wants to execute a system command.
Questions like "flutter best practices" or "how to use X" are "generate", NOT "tool".

Do not include any other text."""


REWRITE_SYSTEM_PROMPT = """You are a query rewriter. Your task is to reformulate a query to improve retrieval results.

Given the original query and the fact that initial retrieval didn't find relevant documents, rewrite the query to:
1. Use different keywords or synonyms
2. Be more specific or more general as appropriate
3. Break down complex queries into simpler terms

Respond with ONLY the rewritten query, nothing else."""


COMMAND_GENERATOR_PROMPT = """You are a Linux command generator. Given a user's request and conversation context, generate the appropriate bash command(s).

Rules:
1. Output ONLY the bash command(s), nothing else - no explanations
2. Use standard Linux commands that work on most distributions
3. For multiple commands, separate them with && or ;
4. Prefer informative output (e.g., use -h for human-readable sizes)
5. Keep commands safe - avoid destructive operations unless explicitly requested
6. If user says "run these commands" or "execute these", look at the conversation history for the commands they're referring to
7. If commands were suggested in previous messages, extract and combine them

Examples:
- "how much disk space" -> "df -h"
- "what OS am I running" -> "uname -a && cat /etc/os-release"
- "check jellyfin status" -> "systemctl status jellyfin"
- "examine jellyfin and related services" -> "dpkg -l | grep jellyfin && systemctl status jellyfin && pgrep -f jellyfin"

Respond with ONLY the command(s), no explanations or markdown."""


def route_query(state: AgentStateDict) -> AgentStateDict:
    """
    Route the query to the appropriate handler.

    Determines whether to retrieve documents, execute tools, or generate directly.
    Takes the current mode into account for routing bias.

    Special syntax:
    - !command - Execute the command directly (e.g., "!ls -la")
    """
    query = state["query"]
    conversation_history = state.get("messages", [])
    current_mode = state.get("mode", AgentMode.CHAT.value)

    # Get mode configuration
    try:
        mode_enum = AgentMode(current_mode)
        mode_config = get_mode_config(mode_enum)
    except ValueError:
        mode_enum = AgentMode.CHAT
        mode_config = get_mode_config(AgentMode.CHAT)

    # Log mode info in debug mode
    if mode_config.verbose:
        logger.info(f"[DEBUG] Current mode: {mode_enum.value}, routing_bias: {mode_config.routing_bias}")

    # Check for explicit command syntax: !command
    if query.strip().startswith("!"):
        command = query.strip()[1:].strip()  # Remove the ! prefix
        if command:
            logger.info(f"Explicit command detected: {command}")
            # Store the command directly in tool_calls to skip command generation
            return {
                "route": RouteDecision.TOOL.value,
                "tool_calls": [{
                    "tool_name": "bash",
                    "arguments": {"command": command},
                    "requires_approval": True,
                    "_explicit": True,  # Flag to skip command generation
                }],
            }

    # If mode has a strong routing bias, use it directly for certain modes
    if mode_config.routing_bias == "image":
        logger.info(f"Query routed to: image (mode bias)")
        return {"route": RouteDecision.IMAGE.value}

    if mode_config.routing_bias == "web":
        logger.info(f"Query routed to: web (mode bias)")
        return {"route": RouteDecision.WEB.value}

    # Build context for router including recent conversation history
    context_parts = []
    if conversation_history:
        # Include last few exchanges for context (limit to avoid token bloat)
        recent = conversation_history[-4:]  # Last 2 exchanges
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            context_parts.append(f"{role}: {content}")

    history_context = ""
    if context_parts:
        history_context = "Recent conversation:\n" + "\n".join(context_parts) + "\n\n"

    # Add mode context to router prompt
    mode_hint = ""
    if mode_config.routing_bias:
        mode_hint = f"\n\nNote: The current mode is '{mode_enum.value}' which prefers '{mode_config.routing_bias}' actions."

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT + mode_hint),
        HumanMessage(content=f"{history_context}Current query: {query}"),
    ]

    try:
        result = _get_selector().generate(messages, force_local=True)
        response = result.content.strip().lower()

        # Determine base route from LLM response
        if "tool" in response:
            base_route = RouteDecision.TOOL.value
        elif "generate" in response:
            base_route = RouteDecision.GENERATE.value
        else:
            base_route = RouteDecision.RETRIEVE.value

        # Apply mode bias adjustment
        route = _apply_mode_bias(base_route, mode_config)

        if mode_config.verbose:
            logger.info(f"[DEBUG] LLM route: {base_route}, final route: {route}")
        else:
            logger.info(f"Query routed to: {route}")

    except Exception as e:
        logger.warning(f"Router failed, defaulting to retrieve: {e}")
        route = RouteDecision.RETRIEVE.value

    return {"route": route}


def _apply_mode_bias(base_route: str, mode_config) -> str:
    """Apply mode routing bias to the base route decision."""
    bias = mode_config.routing_bias

    if not bias:
        return base_route

    # Strong bias modes override the base route
    if bias == "tool" and mode_config.name == AgentMode.EXECUTE:
        # Execute mode strongly prefers tools
        return RouteDecision.TOOL.value

    if bias == "retrieve" and mode_config.name == AgentMode.ASK:
        # Ask mode strongly prefers retrieval
        return RouteDecision.RETRIEVE.value

    if bias == "generate":
        # Code, Plan, and Creative modes prefer generation
        return RouteDecision.GENERATE.value

    return base_route


def retrieve_documents(state: AgentStateDict) -> AgentStateDict:
    """
    Retrieve relevant documents from the knowledge base.
    """
    query = state["query"]

    documents = _get_retriever().retrieve(
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
    relevant_docs = _get_grader().filter_relevant(
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
    context = _get_retriever().format_context(relevant_docs)

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
        result = _get_selector().generate(messages, force_local=True)
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
    current_mode = state.get("mode", AgentMode.CHAT.value)

    # Get mode configuration for system prompt
    try:
        mode_enum = AgentMode(current_mode)
        mode_config = get_mode_config(mode_enum)
        custom_system_prompt = mode_config.system_prompt_modifier
    except ValueError:
        mode_enum = AgentMode.CHAT
        custom_system_prompt = None

    # If the user asks to create/update files outside code mode, force code prompt.
    def looks_like_file_request(text: str) -> bool:
        lowered = text.lower()
        if not any(
            keyword in lowered
            for keyword in ("create", "write", "update", "edit", "add", "generate", "make")
        ):
            return False
        if "docs/" in lowered or "doc/" in lowered:
            return True
        if ".md" in lowered or "markdown" in lowered:
            return True
        import re
        return bool(re.search(r"\b[\w\-/]+\.[a-z0-9]{1,5}\b", lowered))

    if mode_enum != AgentMode.CODE and looks_like_file_request(query):
        code_config = get_mode_config(AgentMode.CODE)
        custom_system_prompt = code_config.system_prompt_modifier

    result = _get_selector().generate_with_context(
        query=query,
        context=context if context else "No specific context available.",
        force_local=True,
        conversation_history=messages if messages else None,
        system_prompt=custom_system_prompt,
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
        "tokens_used": result.tokens_used or 0,
        "input_tokens": result.input_tokens or 0,
        "output_tokens": result.output_tokens or 0,
    }


def generate_fallback(state: AgentStateDict) -> AgentStateDict:
    """
    Generate response using external provider fallback.
    """
    query = state["query"]
    messages = state.get("messages", [])

    # Try to generate with fallback
    try:
        result = _get_selector().generate_with_context(
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
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
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
        "tokens_used": result.tokens_used or 0,
        "input_tokens": result.input_tokens or 0,
        "output_tokens": result.output_tokens or 0,
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

    updater = _get_updater()
    if updater.should_update_kb(query, response, confidence=0.9):
        added_count = updater.update_from_response(
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

    If tool_calls are already present with _explicit flag (from !command syntax),
    skip command generation and use them directly.
    """
    # Check if we already have explicit tool calls (from !command syntax)
    existing_calls = state.get("tool_calls", [])
    if existing_calls and any(call.get("_explicit") for call in existing_calls):
        logger.info("Using explicit command from !syntax, skipping generation")
        return {}  # Keep existing tool_calls

    query = state["query"]
    conversation_history = state.get("messages", [])

    # Build context from conversation history
    context_parts = []
    if conversation_history:
        # Include recent exchanges for context (commands may have been suggested)
        recent = conversation_history[-6:]  # Last 3 exchanges
        for msg in recent:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = msg.content[:1000] if len(msg.content) > 1000 else msg.content
            context_parts.append(f"{role}: {content}")

    history_context = ""
    if context_parts:
        history_context = "Recent conversation:\n" + "\n".join(context_parts) + "\n\n"

    # Use LLM to generate the appropriate command
    messages = [
        SystemMessage(content=COMMAND_GENERATOR_PROMPT),
        HumanMessage(content=f"{history_context}Current request: {query}"),
    ]

    try:
        result = _get_selector().generate(messages, force_local=True)
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

    For explicit commands (from !syntax), returns raw output directly
    without LLM interpretation.
    """
    tool_calls = state.get("tool_calls", [])

    if not tool_calls:
        return {"route": RouteDecision.GENERATE.value}

    # Check if this is an explicit command (from !syntax)
    is_explicit = any(call.get("_explicit") for call in tool_calls)

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

    # For explicit commands (!syntax), return raw output directly without LLM interpretation
    if is_explicit:
        # Format raw output for display
        raw_output_parts = []
        for call, result in zip(tool_calls, results):
            cmd = call.get("arguments", {}).get("command", "")
            if result["success"]:
                raw_output_parts.append(f"$ {cmd}\n{result['output']}")
            else:
                raw_output_parts.append(f"$ {cmd}\nError: {result['error']}")

        response_text = "\n\n".join(raw_output_parts)

        # Add to conversation history so follow-up questions work
        query = state.get("query", "")
        messages = state.get("messages", [])
        new_messages = messages + [
            HumanMessage(content=query),
            AIMessage(content=f"[Shell command executed]\n{response_text}"),
        ]

        return {
            "tool_results": results,
            "context": context,
            "response": response_text,
            "provider_used": "shell",
            "should_update_kb": False,
            "messages": new_messages,
            "route": "end",  # Skip generation, go directly to end
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    return {
        "tool_results": results,
        "context": context,
        "route": RouteDecision.GENERATE.value,
        "tokens_used": 0,
        "input_tokens": 0,
        "output_tokens": 0,
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
    elif route == RouteDecision.WEB.value:
        return "web_search"
    elif route == RouteDecision.IMAGE.value:
        return "generate_image"
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


def web_search(state: AgentStateDict) -> AgentStateDict:
    """
    Perform web search and synthesize results.
    """
    query = state["query"]
    messages = state.get("messages", [])
    current_mode = state.get("mode", AgentMode.CHAT.value)

    # Get mode config for verbose logging
    try:
        mode_enum = AgentMode(current_mode)
        mode_config = get_mode_config(mode_enum)
    except ValueError:
        mode_config = get_mode_config(AgentMode.CHAT)

    if mode_config.verbose:
        logger.info(f"[DEBUG] Web search for: {query}")

    try:
        from src.tools.web import WebSearchTool
        web_tool = WebSearchTool()

        # Enhance query to prioritize official documentation
        enhanced_query = query
        if any(tech in query.lower() for tech in ['flutter', 'dart', 'python', 'react', 'vue', 'angular', 'django', 'fastapi']):
            # Add "official documentation" to prioritize official sources
            enhanced_query = f"{query} official documentation"

        search_results = web_tool.search(enhanced_query, max_results=8)  # Get more, then filter

        if not search_results:
            logger.warning("No web search results found")
            return {
                "response": "I couldn't find any relevant web results for your query.",
                "provider_used": "web",
                "should_update_kb": False,
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        # Prioritize official documentation sources
        official_domains = [
            'flutter.dev', 'dart.dev', 'python.org', 'docs.python.org',
            'reactjs.org', 'react.dev', 'vuejs.org', 'angular.io',
            'djangoproject.com', 'fastapi.tiangolo.com', 'docs.microsoft.com',
            'developer.mozilla.org', 'golang.org', 'rust-lang.org',
            'kotlinlang.org', 'swift.org', 'developer.apple.com',
            'docs.oracle.com', 'kubernetes.io', 'docker.com'
        ]

        # Separate official and non-official sources
        official_sources = []
        other_sources = []

        for result in search_results:
            url = result.get('url', '').lower()
            if any(domain in url for domain in official_domains):
                official_sources.append(result)
            else:
                other_sources.append(result)

        # Combine: prioritize official sources, then others (max 5 total)
        prioritized_results = (official_sources + other_sources)[:5]

        if not prioritized_results:
            prioritized_results = search_results[:5]

        search_results = prioritized_results

        # Format search results as context
        context_parts = []
        url_list = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            snippet = result.get('snippet', '')

            context_parts.append(
                f"[{i}] {title}\n"
                f"URL: {url}\n"
                f"{snippet}"
            )
            url_list.append(f"[{i}] {title} - {url}")

        context = "\n\n".join(context_parts)

        # Append URL reference list at the end
        url_reference = "\n\nURL Reference List:\n" + "\n".join(url_list)
        full_context = context + url_reference

        if mode_config.verbose:
            logger.info(f"[DEBUG] Found {len(search_results)} web results")

        # Optionally fetch full page content for top results (for deeper research)
        enhanced_results = []
        if len(search_results) > 0:
            # Fetch content from top 2 results for richer context
            for i, result in enumerate(search_results[:2]):
                url = result.get('url', '')
                if url:
                    try:
                        page_content = web_tool.fetch_page_content(url, max_length=3000)
                        if page_content:
                            enhanced_results.append({
                                'index': i + 1,
                                'url': url,
                                'content': page_content,
                            })
                    except Exception as e:
                        logger.debug(f"Failed to fetch content from {url}: {e}")

        # Add enhanced content to context if available
        enhanced_context = ""
        if enhanced_results:
            enhanced_parts = []
            for item in enhanced_results:
                enhanced_parts.append(
                    f"[Full content from source {item['index']}]:\n{item['content'][:2000]}"
                )
            enhanced_context = "\n\n" + "\n\n".join(enhanced_parts)

        # Build complete answer with proper structure
        # Show exact example format to avoid confusion
        example_format = """
Example output format:

[Your comprehensive answer here with citations [1], [2], [3]...]

---
Sources:
[1] Title - URL
[2] Title - URL
[3] Title - URL

---
Next steps:
• Specific action from content
• Related topic to explore
• Practical example to try
• Would you like me to save this to KB / see examples / go deeper?"""

        synthesis_prompt = f"""Synthesize these web search results to answer: {query}

SEARCH RESULTS:
{full_context}{enhanced_context}

INSTRUCTIONS:
1. Write a comprehensive answer using inline citations [1], [2], [3] etc.
2. After your answer, add two sections separated by "---" lines
3. First section: List ALL sources with their URLs
4. Second section: List 4 next steps (3 content + 1 interactive question)

{example_format}

Now provide your answer following this exact format."""

        synthesis_messages = [
            SystemMessage(content=(
                "You are a technical research assistant. "
                "Follow the output format EXACTLY as shown in the example. "
                "Use proper line breaks (2 newlines between sections). "
                "Cite ALL sources. Include 4 next steps. "
                "NEVER output literal text like '[blank line]' or include meta-instructions in your answer."
            )),
            HumanMessage(content=synthesis_prompt),
        ]

        result = _get_selector().generate(synthesis_messages, force_local=True)

        # Post-process output to clean up formatting issues
        cleaned_content = result.content

        # Remove literal "[blank line]" text that LLM might output
        cleaned_content = cleaned_content.replace("[blank line]", "")

        # Ensure proper spacing around --- separators
        import re
        cleaned_content = re.sub(r'\n*---\n*', '\n\n---\n\n', cleaned_content)

        # Ensure Sources: and Next steps: have proper spacing
        cleaned_content = re.sub(r'\n*Sources:\n*', '\n\nSources:\n', cleaned_content)
        cleaned_content = re.sub(r'\n*Next steps:\n*', '\n\nNext steps:\n', cleaned_content)

        # Remove excessive blank lines (more than 2 consecutive)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)

        # Clean up trailing/leading whitespace
        cleaned_content = cleaned_content.strip()

        new_messages = messages + [
            HumanMessage(content=query),
            AIMessage(content=cleaned_content),
        ]

        logger.info(f"Generated web search response with {len(search_results)} sources")

        return {
            "response": cleaned_content,
            "provider_used": "web",
            "messages": new_messages,
            "context": full_context,
            "should_update_kb": False,
            "tokens_used": result.tokens_used or 0,
            "input_tokens": result.input_tokens or 0,
            "output_tokens": result.output_tokens or 0,
        }

    except ImportError:
        logger.error("Web search tool not available")
        return {
            "response": "Web search is not available. Please install duckduckgo-search: pip install duckduckgo-search",
            "provider_used": "error",
            "should_update_kb": False,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "response": f"Web search failed: {e}",
            "provider_used": "error",
            "should_update_kb": False,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def generate_image(state: AgentStateDict) -> AgentStateDict:
    """
    Generate image using local Stable Diffusion.
    """
    query = state["query"]
    messages = state.get("messages", [])
    current_mode = state.get("mode", AgentMode.CHAT.value)

    # Get mode config for verbose logging
    try:
        mode_enum = AgentMode(current_mode)
        mode_config = get_mode_config(mode_enum)
    except ValueError:
        mode_config = get_mode_config(AgentMode.CHAT)

    if mode_config.verbose:
        logger.info(f"[DEBUG] Image generation for: {query}")

    try:
        from src.tools.image import ImageGenerationTool
        image_tool = ImageGenerationTool()

        # Enhance the prompt using LLM
        enhance_messages = [
            SystemMessage(content=(
                "You are an expert at writing Stable Diffusion prompts. "
                "Convert the user's request into an optimized SD prompt. "
                "Output ONLY the prompt, no explanations. "
                "Include style keywords, quality boosters like '4k, detailed, masterpiece'."
            )),
            HumanMessage(content=f"Create a prompt for: {query}"),
        ]

        enhanced = _get_selector().generate(enhance_messages, force_local=True)
        enhanced_prompt = enhanced.content.strip()

        if mode_config.verbose:
            logger.info(f"[DEBUG] Enhanced prompt: {enhanced_prompt}")

        # Generate the image
        result = image_tool.generate(enhanced_prompt)

        if result.get("success"):
            image_path = result.get("path", "")
            response = f"Image generated successfully!\n\nPrompt: {enhanced_prompt}\nSaved to: {image_path}"
        else:
            error = result.get("error", "Unknown error")
            response = f"Image generation failed: {error}"

        new_messages = messages + [
            HumanMessage(content=query),
            AIMessage(content=response),
        ]

        logger.info(f"Image generation completed: {result.get('success', False)}")

        return {
            "response": response,
            "provider_used": "image",
            "messages": new_messages,
            "should_update_kb": False,
            "tokens_used": enhanced.tokens_used or 0,
            "input_tokens": enhanced.input_tokens or 0,
            "output_tokens": enhanced.output_tokens or 0,
        }

    except ImportError:
        logger.error("Image generation tool not available")
        return {
            "response": (
                "Image generation is not available. Please ensure:\n"
                "1. ComfyUI or Automatic1111 is running locally\n"
                "2. Configure the SD URL in settings"
            ),
            "provider_used": "error",
            "should_update_kb": False,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {
            "response": f"Image generation failed: {e}",
            "provider_used": "error",
            "should_update_kb": False,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def get_selector() -> "ProviderSelector":
    """Get the shared ProviderSelector instance.

    This allows external code to access model switching functionality.
    """
    return _get_selector()


def reset_model_dependent_caches() -> None:
    """Clear cached components that bake in model settings."""
    _get_grader.cache_clear()
    _get_updater.cache_clear()
