"""Main entry point for the LLM agent."""

import logging
import sys
import time
import threading
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

__version__ = "0.2.0"


class StatusHandler(logging.Handler):
    """Custom log handler that captures status messages for display."""

    def __init__(self):
        super().__init__()
        self.current_status = ""
        self.setLevel(logging.INFO)

    def emit(self, record):
        msg = record.getMessage()
        # Capture relevant status messages
        if any(keyword in msg.lower() for keyword in [
            "routed to", "retrieved", "graded", "generated", "rewritten",
            "command for query", "switching model"
        ]):
            # Clean up the message for display
            self.current_status = msg

app = typer.Typer(
    name="llm-agent",
    help="Local-first LLM agent with RAG-based knowledge base",
)
console = Console()


@app.command()
def chat(
    query: str = typer.Argument(None, help="Single query to process"),
) -> None:
    """
    Start an interactive chat session with the agent.

    If a query is provided, process it and exit.
    Otherwise, start an interactive REPL.
    """
    from src.agent import Agent
    from src.llm.local import LocalLLM

    # Check Ollama availability
    if not LocalLLM.check_availability():
        console.print(
            "[yellow]Warning:[/yellow] Ollama is not available. "
            f"Ensure Ollama is running at {settings.ollama_base_url}"
        )

    agent = Agent()

    if query:
        # Single query mode
        _process_query(agent, query)
        return

    # Interactive mode
    current_model = agent.get_model()
    console.print(
        Panel(
            "[bold cyan]LLM Agent[/bold cyan] [dim]v" + __version__ + "[/dim]\n"
            "Local-first AI assistant with RAG knowledge base\n\n"
            f"Model: [green]{current_model}[/green]\n\n"
            "Commands:\n"
            "  /model         - Show current model\n"
            "  /model <name>  - Switch to a different model\n"
            "  /models        - List available Ollama models\n"
            "  /clear         - Clear conversation history\n"
            "  /stats         - Show knowledge base stats\n"
            "  /help          - Show help\n"
            "  /quit          - Exit",
            title="Welcome",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                if _handle_command(user_input, agent):
                    continue
                else:
                    break

            _process_query(agent, user_input)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
        except EOFError:
            break

    console.print("\n[cyan]Goodbye![/cyan]")


@app.command()
def serve(
    host: str = typer.Option(settings.api_host, "--host", "-h"),
    port: int = typer.Option(settings.api_port, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
) -> None:
    """Start the REST API server."""
    import uvicorn

    console.print(f"Starting API server at [cyan]http://{host}:{port}[/cyan]")
    console.print("API docs available at [cyan]/docs[/cyan]")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def query(
    text: str = typer.Argument(..., help="Query to process"),
    no_history: bool = typer.Option(
        False, "--no-history", help="Don't use conversation history"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Ollama model to use (e.g., mistral:7b, deepseek-r1:7b)"
    ),
) -> None:
    """Process a single query and exit."""
    from src.agent import Agent

    agent = Agent()

    if model:
        agent.set_model(model)
        console.print(f"[dim]Using model: {model}[/dim]")

    result = agent.query(text, include_history=not no_history)

    console.print(Panel(result["response"], title="Response"))
    console.print(f"[dim]Provider: {result['provider']} | Documents: {result['documents_used']}[/dim]")


@app.command()
def check() -> None:
    """Check system status and dependencies."""
    from src.llm.local import LocalLLM
    from src.llm.selector import ProviderSelector
    from src.knowledge import VectorStore

    console.print("[bold]System Status Check[/bold]\n")

    # Check Ollama
    ollama_ok = LocalLLM.check_availability()
    status = "[green]OK[/green]" if ollama_ok else "[red]NOT AVAILABLE[/red]"
    console.print(f"Ollama ({settings.ollama_base_url}): {status}")

    # Check ChromaDB
    try:
        vs = VectorStore()
        stats = vs.get_collection_stats()
        console.print(f"ChromaDB: [green]OK[/green] ({stats['document_count']} documents)")
    except Exception as e:
        console.print(f"ChromaDB: [red]ERROR[/red] ({e})")

    # Check available providers
    selector = ProviderSelector()
    providers = selector.list_available_providers()
    console.print(f"Available providers: {', '.join(providers)}")

    # Check settings
    console.print(f"\nModel: {settings.ollama_model}")
    console.print(f"Embedding model: {settings.ollama_embedding_model}")
    console.print(f"Fallback enabled: {settings.fallback_enabled}")


def _process_query(agent, query: str) -> None:
    """Process a query and display the response."""
    result = None
    start_time = time.time()

    # Set up status handler to capture processing steps
    status_handler = StatusHandler()
    agent_logger = logging.getLogger("src.agent.nodes")
    llm_logger = logging.getLogger("src.llm")
    agent_logger.addHandler(status_handler)
    llm_logger.addHandler(status_handler)
    agent_logger.setLevel(logging.INFO)
    llm_logger.setLevel(logging.INFO)

    def run_query():
        nonlocal result
        result = agent.query(query)

    # Run query in background thread
    thread = threading.Thread(target=run_query)
    thread.start()

    # Show spinner with elapsed time and status
    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    frame_idx = 0
    with Live(console=console, refresh_per_second=10) as live:
        while thread.is_alive():
            elapsed = time.time() - start_time
            spinner_text = Text()
            spinner_text.append(f"{spinner_frames[frame_idx]} ", style="cyan")
            spinner_text.append(f"Thinking... ", style="cyan")
            spinner_text.append(f"({elapsed:.1f}s)", style="dim")

            # Show current processing step
            if status_handler.current_status:
                spinner_text.append(f"\n  {status_handler.current_status}", style="dim italic")

            live.update(spinner_text)
            frame_idx = (frame_idx + 1) % len(spinner_frames)
            time.sleep(0.1)

    thread.join()
    elapsed_total = time.time() - start_time

    # Clean up handlers
    agent_logger.removeHandler(status_handler)
    llm_logger.removeHandler(status_handler)

    response_text = result["response"]

    # Check if agent is asking for clarification
    if response_text.strip().startswith("CLARIFY:"):
        clarify_question = response_text.strip()[8:].strip()
        console.print(f"\n[bold yellow]Clarification needed:[/bold yellow] {clarify_question}")
        clarification = Prompt.ask("[bold cyan]Your answer[/bold cyan]")

        # Run follow-up query with clarification
        follow_up = f"{query} (Clarification: {clarification})"
        console.print()
        _process_query(agent, follow_up)
        return

    console.print("\n[bold green]Agent[/bold green]")
    console.print(Markdown(response_text))

    # Determine grounding status
    is_local_provider = result["provider"] in ("ollama", "local")
    has_kb_docs = result["documents_used"] > 0
    has_tools = len(result["tool_results"]) > 0

    if has_kb_docs or has_tools:
        # Grounded response (KB or tools)
        if has_kb_docs and has_tools:
            grounding = "[green]Grounded[/green] (KB + Tools)"
        elif has_kb_docs:
            grounding = "[green]Grounded[/green] (KB)"
        else:
            grounding = "[green]Grounded[/green] (Tools)"
    elif is_local_provider:
        grounding = "[yellow]Local[/yellow] (no grounding)"
    else:
        grounding = "[blue]External[/blue] (" + result["provider"] + ")"

    # Show metadata
    meta_parts = [grounding]
    if result["documents_used"]:
        meta_parts.append(f"Docs: {result['documents_used']}")
    if result["tool_results"]:
        meta_parts.append(f"Tools: {len(result['tool_results'])}")
    meta_parts.append(f"Time: {elapsed_total:.1f}s")

    console.print(f"[dim]{' | '.join(meta_parts)}[/dim]")


def _handle_command(command: str, agent) -> bool:
    """
    Handle a CLI command.

    Returns True to continue, False to exit.
    """
    cmd = command.strip()
    cmd_lower = cmd.lower()

    if cmd_lower in ("/quit", "/exit", "/q"):
        return False

    if cmd_lower == "/clear":
        agent.clear_history()
        console.print("[green]Conversation history cleared.[/green]")
        return True

    if cmd_lower == "/models":
        models = agent.list_models()
        current = agent.get_model()
        if models:
            console.print("[bold]Available Ollama models:[/bold]")
            for model in models:
                if model == current:
                    console.print(f"  [green]* {model}[/green] (current)")
                else:
                    console.print(f"    {model}")
        else:
            console.print("[yellow]No models found. Is Ollama running?[/yellow]")
        return True

    if cmd_lower == "/model":
        current = agent.get_model()
        console.print(f"Current model: [cyan]{current}[/cyan]")
        return True

    if cmd_lower.startswith("/model "):
        model_name = cmd[7:].strip()
        if not model_name:
            console.print("[yellow]Usage: /model <model_name>[/yellow]")
            return True

        # Check if model exists
        available = agent.list_models()
        if available and model_name not in available:
            # Check for partial match (e.g., "mistral" matches "mistral:7b")
            matches = [m for m in available if model_name in m]
            if len(matches) == 1:
                model_name = matches[0]
            elif matches:
                console.print(f"[yellow]Ambiguous model name. Did you mean one of: {', '.join(matches)}?[/yellow]")
                return True
            else:
                console.print(f"[yellow]Model '{model_name}' not found. Use /models to see available models.[/yellow]")
                return True

        agent.set_model(model_name)
        console.print(f"[green]Switched to model: {model_name}[/green]")
        return True

    if cmd_lower == "/stats":
        from src.knowledge import VectorStore

        vs = VectorStore()
        stats = vs.get_collection_stats()
        console.print(f"Documents in KB: {stats['document_count']}")
        console.print(f"Conversation history: {agent.get_history_length()} messages")
        console.print(f"Current model: {agent.get_model()}")
        return True

    if cmd_lower == "/help":
        console.print(
            Panel(
                "Available commands:\n"
                "  /model         - Show current model\n"
                "  /model <name>  - Switch to a different model\n"
                "  /models        - List available Ollama models\n"
                "  /clear         - Clear conversation history\n"
                "  /stats         - Show knowledge base and conversation stats\n"
                "  /help          - Show this help message\n"
                "  /quit          - Exit the chat",
                title="Help",
            )
        )
        return True

    console.print(f"[yellow]Unknown command: {command}[/yellow]")
    return True


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
