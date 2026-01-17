"""CLI for document ingestion."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import IngestionPipeline
from src.knowledge import VectorStore

app = typer.Typer(
    name="llm-ingest",
    help="Document ingestion CLI for the LLM agent knowledge base",
)
console = Console()


@app.command()
def file(
    path: Path = typer.Argument(..., help="Path to the file to ingest"),
) -> None:
    """Ingest a single file into the knowledge base."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    if not path.is_file():
        console.print(f"[red]Error:[/red] Not a file: {path}")
        raise typer.Exit(1)

    console.print(f"Ingesting file: [cyan]{path}[/cyan]")

    pipeline = IngestionPipeline()
    stats = pipeline.ingest_file(path)

    _print_stats(stats)


@app.command()
def directory(
    path: Path = typer.Argument(..., help="Path to the directory to ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R"),
    exclude: list[str] = typer.Option(
        None, "--exclude", "-e", help="Patterns to exclude"
    ),
) -> None:
    """Ingest all documents from a directory."""
    if not path.exists():
        console.print(f"[red]Error:[/red] Directory not found: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {path}")
        raise typer.Exit(1)

    console.print(f"Ingesting directory: [cyan]{path}[/cyan]")
    console.print(f"Recursive: {recursive}")

    pipeline = IngestionPipeline()
    stats = pipeline.ingest_directory(
        path,
        recursive=recursive,
        exclude_patterns=exclude,
    )

    _print_stats(stats)


@app.command()
def text(
    content: str = typer.Argument(..., help="Text content to ingest"),
    source: str = typer.Option("cli-input", "--source", "-s", help="Source identifier"),
) -> None:
    """Ingest raw text into the knowledge base."""
    console.print(f"Ingesting text from source: [cyan]{source}[/cyan]")

    pipeline = IngestionPipeline()
    stats = pipeline.ingest_text(content, source=source)

    _print_stats(stats)


@app.command()
def stats() -> None:
    """Show knowledge base statistics."""
    vectorstore = VectorStore()
    kb_stats = vectorstore.get_collection_stats()

    table = Table(title="Knowledge Base Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Collection Name", kb_stats["collection_name"])
    table.add_row("Document Count", str(kb_stats["document_count"]))
    table.add_row("Storage Path", kb_stats["path"])

    console.print(table)


@app.command()
def clear(
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Confirm clearing the knowledge base"
    ),
) -> None:
    """Clear all documents from the knowledge base."""
    if not confirm:
        confirm = typer.confirm("Are you sure you want to clear the knowledge base?")

    if not confirm:
        console.print("Aborted.")
        raise typer.Exit(0)

    vectorstore = VectorStore()
    vectorstore.clear()
    console.print("[green]Knowledge base cleared.[/green]")


@app.command("delete-source")
def delete_source(
    source: str = typer.Argument(..., help="Source to delete"),
) -> None:
    """Delete all documents from a specific source."""
    vectorstore = VectorStore()
    count = vectorstore.delete_by_source(source)
    console.print(f"[green]Deleted {count} documents from source: {source}[/green]")


def _print_stats(stats) -> None:
    """Print ingestion statistics."""
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(stats.files_processed))
    table.add_row("Documents Loaded", str(stats.documents_loaded))
    table.add_row("Chunks Created", str(stats.chunks_created))
    table.add_row("Chunks Stored", str(stats.chunks_stored))

    console.print(table)

    if stats.errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for error in stats.errors:
            console.print(f"  - {error}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
