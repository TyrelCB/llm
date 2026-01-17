"""Main entry point for the LLM agent."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

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
    console.print(
        Panel(
            "[bold cyan]LLM Agent[/bold cyan]\n"
            "Local-first AI assistant with RAG knowledge base\n\n"
            "Commands:\n"
            "  /clear - Clear conversation history\n"
            "  /stats - Show knowledge base stats\n"
            "  /help  - Show help\n"
            "  /quit  - Exit",
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
) -> None:
    """Process a single query and exit."""
    from src.agent import Agent

    agent = Agent()
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
    with console.status("[cyan]Thinking...[/cyan]"):
        result = agent.query(query)

    console.print("\n[bold green]Agent[/bold green]")
    console.print(Markdown(result["response"]))

    # Show metadata
    meta_parts = []
    if result["provider"]:
        meta_parts.append(f"Provider: {result['provider']}")
    if result["documents_used"]:
        meta_parts.append(f"Docs: {result['documents_used']}")
    if result["tool_results"]:
        meta_parts.append(f"Tools: {len(result['tool_results'])}")

    if meta_parts:
        console.print(f"[dim]{' | '.join(meta_parts)}[/dim]")


def _handle_command(command: str, agent) -> bool:
    """
    Handle a CLI command.

    Returns True to continue, False to exit.
    """
    cmd = command.lower().strip()

    if cmd in ("/quit", "/exit", "/q"):
        return False

    if cmd == "/clear":
        agent.clear_history()
        console.print("[green]Conversation history cleared.[/green]")
        return True

    if cmd == "/stats":
        from src.knowledge import VectorStore

        vs = VectorStore()
        stats = vs.get_collection_stats()
        console.print(f"Documents in KB: {stats['document_count']}")
        console.print(f"Conversation history: {agent.get_history_length()} messages")
        return True

    if cmd == "/help":
        console.print(
            Panel(
                "Available commands:\n"
                "  /clear - Clear conversation history\n"
                "  /stats - Show knowledge base and conversation stats\n"
                "  /help  - Show this help message\n"
                "  /quit  - Exit the chat",
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
