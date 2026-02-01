"""Main entry point for the LLM agent."""

import logging
import os
import re
import shlex
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if "chat" in sys.argv:
    sys.stderr.write("Launching chat (loading CLI)...\n")
    sys.stderr.flush()

from config.settings import settings, configure_project_root
from src.agent.modes import AgentMode, get_mode_config, get_next_mode, list_modes, get_mode_by_name

__version__ = "0.8.11"
logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Track session statistics."""

    start_time: float = field(default_factory=time.time)
    queries: int = 0
    shell_commands: int = 0
    plans_executed: int = 0
    kb_retrievals: int = 0
    documents_used: int = 0
    tools_executed: int = 0
    llm_calls: int = 0
    total_response_time: float = 0.0
    clarifications: int = 0
    errors: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    context_window_size: int = 0
    max_tokens_per_query: int = 0
    max_input_tokens: int = 0
    max_output_tokens: int = 0

    def record_query(self, result: dict, response_time: float) -> None:
        """Record stats from a query result."""
        self.queries += 1
        self.total_response_time += response_time
        self.llm_calls += 1

        # Track token usage
        tokens = result.get("tokens_used", 0)
        input_tokens = result.get("input_tokens", 0)
        output_tokens = result.get("output_tokens", 0)

        if tokens:
            self.total_tokens += tokens
            self.max_tokens_per_query = max(self.max_tokens_per_query, tokens)

        if input_tokens:
            self.total_input_tokens += input_tokens
            self.max_input_tokens = max(self.max_input_tokens, input_tokens)

        if output_tokens:
            self.total_output_tokens += output_tokens
            self.max_output_tokens = max(self.max_output_tokens, output_tokens)

        if result.get("provider") == "shell":
            self.shell_commands += 1
        if result.get("documents_used", 0) > 0:
            self.kb_retrievals += 1
            self.documents_used += result["documents_used"]
        if result.get("tool_results"):
            self.tools_executed += len(result["tool_results"])

    def record_plan(self) -> None:
        """Record a plan execution."""
        self.plans_executed += 1

    def record_clarification(self) -> None:
        """Record a clarification request."""
        self.clarifications += 1

    def record_error(self) -> None:
        """Record an error."""
        self.errors += 1

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return time.time() - self.start_time

    @property
    def avg_response_time(self) -> float:
        """Average response time per query."""
        return self.total_response_time / self.queries if self.queries > 0 else 0

    def display(self, console: Console) -> None:
        """Display session statistics."""
        duration = self.duration
        hours, remainder = divmod(int(duration), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"

        table = Table(title="Session Statistics", show_header=False, box=None)
        table.add_column("Stat", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Duration", duration_str)
        table.add_row("Queries", str(self.queries))
        if self.shell_commands > 0:
            table.add_row("Shell commands", str(self.shell_commands))
        if self.plans_executed > 0:
            table.add_row("Plans executed", str(self.plans_executed))
        if self.kb_retrievals > 0:
            table.add_row("KB retrievals", str(self.kb_retrievals))
        if self.documents_used > 0:
            table.add_row("Documents used", str(self.documents_used))
        if self.tools_executed > 0:
            table.add_row("Tools executed", str(self.tools_executed))
        if self.clarifications > 0:
            table.add_row("Clarifications", str(self.clarifications))
        if self.errors > 0:
            table.add_row("Errors", str(self.errors))
        if self.queries > 0:
            table.add_row("Avg response time", f"{self.avg_response_time:.1f}s")

        # Context window usage stats
        if self.total_tokens > 0 and self.context_window_size > 0:
            table.add_row("", "")  # Spacer
            table.add_row("Total tokens", f"{self.total_tokens:,}")
            if self.total_input_tokens > 0:
                table.add_row("  Input tokens", f"↓ {self.total_input_tokens:,}")
            if self.total_output_tokens > 0:
                table.add_row("  Output tokens", f"↑ {self.total_output_tokens:,}")
            table.add_row("Max tokens/query", f"{self.max_tokens_per_query:,}")
            if self.max_input_tokens > 0:
                table.add_row("  Max input", f"↓ {self.max_input_tokens:,}")
            if self.max_output_tokens > 0:
                table.add_row("  Max output", f"↑ {self.max_output_tokens:,}")
            table.add_row("Avg tokens/query", f"{self.total_tokens // self.queries:,}")
            table.add_row("Context window", f"{self.context_window_size:,}")
            max_usage_pct = (self.max_tokens_per_query / self.context_window_size) * 100
            table.add_row("Peak usage", f"{max_usage_pct:.1f}%")

        console.print()
        console.print(table)


# Global session stats
_session_stats: SessionStats | None = None


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


@app.callback()
def _configure_project(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root for file tools and local data (defaults to cwd if not set).",
    ),
) -> None:
    """Configure project root for CLI runs."""
    root = project_root
    if root is None:
        if os.getenv("PROJECT_ROOT"):
            root = settings.project_root
        else:
            root = Path.cwd()
    configure_project_root(root)


def _extract_file_blocks(response: str) -> list[dict[str, str]]:
    """Extract FILE blocks from a code-mode response."""
    lines = response.splitlines()
    blocks: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        match = re.match(r"^\s*FILE:\s*(.+)\s*$", lines[i])
        if not match:
            i += 1
            continue
        path = match.group(1).strip()
        i += 1
        if i >= len(lines) or not lines[i].lstrip().startswith("```"):
            continue
        i += 1
        content_lines: list[str] = []
        while i < len(lines) and not lines[i].lstrip().startswith("```"):
            content_lines.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith("```"):
            i += 1
        blocks.append({"path": path, "content": "\n".join(content_lines)})
    return blocks


def _looks_like_file_request(query: str) -> bool:
    """Heuristic to detect file creation/update intents."""
    lowered = query.lower()
    if not any(
        keyword in lowered
        for keyword in ("create", "write", "update", "edit", "add", "generate", "make")
    ):
        return False
    if "docs/" in lowered or "doc/" in lowered:
        return True
    if ".md" in lowered or "markdown" in lowered:
        return True
    if re.search(r"\b[\w\-/]+\.[a-z0-9]{1,5}\b", lowered):
        return True
    return False


def _apply_file_blocks(blocks: list[dict[str, str]]) -> list[dict[str, str | bool]]:
    """Apply file blocks using the write_file tool."""
    if not blocks:
        return []
    from src.tools.registry import ToolRegistry
    registry = ToolRegistry()
    results: list[dict[str, str | bool]] = []
    for block in blocks:
        output = registry.execute(
            "write_file",
            {"path": block["path"], "content": block["content"], "mode": "overwrite"},
        )
        success = output.startswith("[OK]")
        results.append(
            {
                "path": block["path"],
                "success": success,
                "message": output,
            }
        )
    return results


def _render_apply_summary(results: list[dict[str, str | bool]]) -> str:
    """Render a short summary of applied file changes."""
    if not results:
        return ""
    applied = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    lines = [f"Applied {len(applied)} file(s):"]
    for item in applied:
        lines.append(f"- `{item['path']}`")
    if failed:
        lines.append("\nErrors:")
        for item in failed:
            lines.append(f"- `{item['path']}`: {item['message']}")
    return "\n".join(lines)


def _merge_ingest_stats(stats_list: list) -> object:
    """Merge multiple ingestion stats objects into one."""
    from src.ingestion.pipeline import IngestionStats

    merged = IngestionStats(
        files_processed=0,
        documents_loaded=0,
        chunks_created=0,
        chunks_stored=0,
        errors=[],
    )
    for stats in stats_list:
        merged.files_processed += stats.files_processed
        merged.documents_loaded += stats.documents_loaded
        merged.chunks_created += stats.chunks_created
        merged.chunks_stored += stats.chunks_stored
        merged.errors.extend(stats.errors or [])
    return merged


def _print_ingest_stats(stats: object) -> None:
    """Print ingestion statistics."""
    table = Table(title="Ingestion Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(getattr(stats, "files_processed", 0)))
    table.add_row("Documents Loaded", str(getattr(stats, "documents_loaded", 0)))
    table.add_row("Chunks Created", str(getattr(stats, "chunks_created", 0)))
    table.add_row("Chunks Stored", str(getattr(stats, "chunks_stored", 0)))

    console.print(table)

    errors = getattr(stats, "errors", []) or []
    if errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for error in errors:
            console.print(f"  - {error}")


def _get_mode_color(mode: AgentMode) -> str:
    """Get the display color for a mode."""
    colors = {
        AgentMode.CHAT: "cyan",
        AgentMode.PLAN: "yellow",
        AgentMode.AGENTIC: "bright_magenta",
        AgentMode.ASK: "green",
        AgentMode.EXECUTE: "red",
        AgentMode.CODE: "magenta",
        AgentMode.IMAGE: "blue",
        AgentMode.RESEARCH: "bright cyan",
        AgentMode.DEBUG: "bright_yellow",
        AgentMode.CREATIVE: "bright_magenta",
    }
    return colors.get(mode, "white")


def _get_prompt_color(mode: AgentMode) -> str:
    """Get a prompt_toolkit-safe color for a mode."""
    colors = {
        AgentMode.CHAT: "ansicyan",
        AgentMode.PLAN: "ansiyellow",
        AgentMode.AGENTIC: "ansibrightmagenta",
        AgentMode.ASK: "ansigreen",
        AgentMode.EXECUTE: "ansired",
        AgentMode.CODE: "ansimagenta",
        AgentMode.IMAGE: "ansiblue",
        AgentMode.RESEARCH: "ansibrightcyan",
        AgentMode.DEBUG: "ansibrightyellow",
        AgentMode.CREATIVE: "ansibrightmagenta",
    }
    return colors.get(mode, "ansiwhite")


def _get_state_file() -> Path:
    """Get path to state persistence file."""
    state_dir = Path.home() / ".config" / "llm-agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "state.json"


def _save_state(model: str, mode: str) -> None:
    """Save current model and mode to disk."""
    import json
    state_file = _get_state_file()
    try:
        state = {"model": model, "mode": mode}
        state_file.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")


def _load_state() -> dict:
    """Load last model and mode from disk."""
    import json
    state_file = _get_state_file()
    try:
        if state_file.exists():
            return json.loads(state_file.read_text())
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
    return {}


def _agentic_step_approval(action: dict) -> bool:
    """Auto-approve agentic actions."""
    return True


def _agentic_bash_approval(command: str) -> bool:
    """Auto-approve bash commands for agentic runs."""
    return True


def _agentic_ask_user(question: str) -> str:
    """Agentic runs should not prompt for user input."""
    return ""


def _agentic_step_result(action: dict, result: dict) -> None:
    """Display the output from an agentic step."""
    action_spec = action.get("action") or {}
    tool = action_spec.get("tool", "unknown")
    summary = result.get("summary", "") or result.get("output", "")
    if summary:
        tool_ms = result.get("tool_duration_ms", 0)
        step_ms = result.get("step_duration_ms", 0)
        timing = ""
        if tool_ms or step_ms:
            timing = f" [tool {tool_ms}ms, step {step_ms}ms]"
        console.print(f"[dim]Step result ({tool}){timing}:[/dim]")
        console.print(summary)


def _interactive_model_selector(agent) -> str | None:
    """Interactive model selection with numbered menu."""
    models = agent.list_models()
    current = agent.get_model()

    if not models:
        console.print("[yellow]No models found. Is Ollama running?[/yellow]")
        return None

    console.print("\n[bold]Available Ollama Models:[/bold]")
    for i, model in enumerate(models, 1):
        marker = "[green]*[/green]" if model == current else " "
        console.print(f"{marker} {i:2}. {model}")

    console.print("\n[dim]Enter number to select, or press Enter to cancel[/dim]")
    choice = Prompt.ask("[bold cyan]Select model[/bold cyan]", default="")

    if not choice.strip():
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
        else:
            console.print("[yellow]Invalid selection[/yellow]")
            return None
    except ValueError:
        console.print("[yellow]Invalid input[/yellow]")
        return None


def _create_prompt_session(agent):
    """Create a prompt_toolkit session with key bindings and tab completion."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.completion import Completer, Completion

        complete_style = None
        completion_style = None
        try:
            from prompt_toolkit.completion import CompleteStyle
            from prompt_toolkit.styles import Style

            complete_style = CompleteStyle.MULTI_COLUMN
            completion_style = Style.from_dict(
                {
                    "completion": "fg:#7a7a7a",
                    "completion.match": "fg:#d2d2d2",
                    "completion.current": "bg:#444444 fg:#ffffff",
                    "completion.menu.completion": "fg:#7a7a7a",
                    "completion.menu.completion.current": "bg:#444444 fg:#ffffff",
                    "completion.scrollbar": "bg:#333333",
                    "completion.scrollbar.arrow": "fg:#aaaaaa bg:#333333",
                }
            )
        except Exception:
            complete_style = None
            completion_style = None

        command_meta = {
            "/clear": "Clear conversation history",
            "/help": "Show help and command reference",
            "/stats": "Show KB stats, model, and context",
            "/model": "Show current model or switch: /model <name>",
            "/models": "List available Ollama models",
            "/mode": "Show or set mode: /mode <name>",
            "/modes": "List available modes",
            "/context": "Show or set context window: /context <size>",
            "/plan": "Plan and execute a multi-step task",
            "/ingest": "Ingest files/dirs/globs into the KB",
            "/quit": "Exit the chat",
            "/exit": "Exit the chat",
            "/q": "Exit the chat",
        }

        mode_meta = {
            mode.value: get_mode_config(mode).description for mode in AgentMode
        }

        ingest_flags = {
            "--recursive": "Recurse into subdirectories (default)",
            "--no-recursive": "Only ingest the top-level directory",
            "--exclude": "Exclude matching paths (repeatable)",
            "--glob": "Ingest files matching glob pattern",
            "--text": "Ingest raw text content",
            "--source": "Source label for --text",
        }

        class SlashCompleter(Completer):
            def __init__(self) -> None:
                try:
                    self._models = agent.list_models()
                except Exception:
                    self._models = []

            def _get_prefix(self, document) -> tuple[str, list[str], bool]:
                text = document.text_before_cursor
                if not text.startswith("/"):
                    return "", [], False
                parts = text.split()
                if not parts:
                    return "", [], False
                has_trailing_space = text.endswith(" ")
                current = "" if has_trailing_space else parts[-1]
                return current, parts, has_trailing_space

            def get_completions(self, document, complete_event):
                prefix, parts, has_trailing_space = self._get_prefix(document)
                if not parts:
                    return

                # Command name completion
                if len(parts) == 1 and not has_trailing_space:
                    for cmd, meta in command_meta.items():
                        if cmd.startswith(prefix):
                            yield Completion(
                                cmd,
                                start_position=-len(prefix),
                                display=cmd,
                                display_meta=meta,
                            )
                    return

                command = parts[0]
                if command == "/mode":
                    for name, meta in mode_meta.items():
                        if name.startswith(prefix):
                            yield Completion(
                                name,
                                start_position=-len(prefix),
                                display=name,
                                display_meta=meta,
                            )
                    return

                if command == "/model":
                    for name in self._models:
                        if name.startswith(prefix):
                            yield Completion(
                                name,
                                start_position=-len(prefix),
                                display=name,
                            )
                    return

                if command == "/ingest":
                    if prefix.startswith("-") or (has_trailing_space or prefix == ""):
                        for flag, meta in ingest_flags.items():
                            if flag.startswith(prefix):
                                yield Completion(
                                    flag,
                                    start_position=-len(prefix),
                                    display=flag,
                                    display_meta=meta,
                                )

        completer = SlashCompleter()

        bindings = KeyBindings()

        @bindings.add(Keys.BackTab)  # Shift+Tab
        def _(event):
            """Cycle to next mode on Shift+Tab."""
            new_mode = agent.cycle_mode()
            # Save state
            _save_state(agent.get_model(), new_mode.value)
            event.app.exit(result='__MODE_SWITCH__')

        session_kwargs = {
            "key_bindings": bindings,
            "completer": completer,
            "complete_while_typing": True,
        }
        if complete_style is not None:
            session_kwargs["complete_style"] = complete_style
        if completion_style is not None:
            session_kwargs["style"] = completion_style

        session = PromptSession(**session_kwargs)
        return session
    except ImportError:
        return None


@app.command()
def chat(
    query: str = typer.Argument(None, help="Single query to process"),
    mode: str = typer.Option(None, "--mode", "-m", help="Initial mode (chat, plan, ask, execute, code, image, research, debug, creative)"),
) -> None:
    """
    Start an interactive chat session with the agent.

    If a query is provided, process it and exit.
    Otherwise, start an interactive REPL.

    Use Shift+Tab to cycle between modes.
    """
    import importlib
    import os
    import sys

    sys.stderr.write("Starting chat session (loading modules)...\n")
    sys.stderr.flush()
    global _session_stats

    timing_enabled = os.getenv("LLM_IMPORT_TIMING", "1") != "0" or bool(os.getenv("LLM_IMPORT_TIMING_VERBOSE"))

    def _timed_import(module_name: str, attr: str | None = None):
        if not timing_enabled:
            module = importlib.import_module(module_name)
            return getattr(module, attr) if attr else module

        start = time.perf_counter()
        module = importlib.import_module(module_name)
        duration = time.perf_counter() - start
        sys.stderr.write(f"Imported {module_name} in {duration:.2f}s\n")
        sys.stderr.flush()
        return getattr(module, attr) if attr else module

    if os.getenv("LLM_IMPORT_TIMING_VERBOSE"):
        for module_name in [
            "src.knowledge.vectorstore",
            "src.llm.selector",
            "src.agent.nodes",
            "src.agent.graph",
        ]:
            _timed_import(module_name)

    Agent = _timed_import("src.agent", "Agent")

    sys.stderr.write("Core modules loaded. Initializing components...\n")
    sys.stderr.flush()

    # Initialize session stats
    _session_stats = SessionStats()

    # Check Ollama availability without blocking startup
    def _check_ollama_async() -> None:
        def _worker() -> None:
            try:
                LocalLLM = _timed_import("src.llm.local", "LocalLLM")
                if not LocalLLM.check_availability():
                    console.print(
                        "[yellow]Warning:[/yellow] Ollama is not available. "
                        f"Ensure Ollama is running at {settings.ollama_base_url}"
                    )
            except Exception as exc:
                console.print(f"[yellow]Warning:[/yellow] Ollama check failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    _check_ollama_async()

    init_start = time.perf_counter()
    console.print("[dim]Initializing agent...[/dim]")
    agent = Agent()
    console.print(f"[dim]Initialization complete in {time.perf_counter() - init_start:.1f}s[/dim]")
    agent.set_agentic_callbacks(
        step_approval_callback=_agentic_step_approval,
        step_result_callback=_agentic_step_result,
        bash_approval_callback=_agentic_bash_approval,
        ask_user_callback=_agentic_ask_user,
    )

    # Load saved state (last model and mode)
    saved_state = _load_state()

    # Set initial mode - priority: CLI arg > saved state > default
    if mode:
        mode_enum = get_mode_by_name(mode)
        if mode_enum:
            agent.set_mode(mode_enum)
        else:
            console.print(f"[yellow]Unknown mode '{mode}', using default (chat)[/yellow]")
    elif saved_state.get("mode"):
        mode_enum = get_mode_by_name(saved_state["mode"])
        if mode_enum:
            agent.set_mode(mode_enum)
            console.print(f"[dim]Restored mode: {mode_enum.value}[/dim]")

    # Set model from saved state if available
    if saved_state.get("model"):
        try:
            available = agent.list_models()
            if available and saved_state["model"] in available:
                agent.set_model(saved_state["model"])
                console.print(f"[dim]Restored model: {saved_state['model']}[/dim]")
        except Exception as e:
            logger.debug(f"Failed to restore model: {e}")

    if not query and agent.get_mode() == AgentMode.AGENTIC:
        agent.set_agentic_approval_mode("auto")

    if query:
        # Single query mode
        _process_query(agent, query)
        return

    # Interactive mode
    current_model = agent.get_model()
    current_mode = agent.get_mode()
    mode_color = _get_mode_color(current_mode)

    # Get model context window info
    try:
        from src.llm.local import LocalLLM
        llm = LocalLLM(current_model)
        model_info = llm.get_model_info()
        context_length = model_info.get("context_length", settings.ollama_num_ctx)
        _session_stats.context_window_size = context_length
    except Exception as e:
        logger.debug(f"Failed to get model info: {e}")
        context_length = settings.ollama_num_ctx
        _session_stats.context_window_size = context_length

    console.print(
        Panel(
            "[bold cyan]LLM Agent[/bold cyan] [dim]v" + __version__ + "[/dim]\n"
            "Local-first AI assistant with RAG knowledge base\n\n"
            f"Model: [green]{current_model}[/green]\n"
            f"Mode:  [{mode_color}]{current_mode.value}[/{mode_color}]\n"
            f"Context Window: [cyan]{context_length:,}[/cyan] tokens\n\n"
            "Mode switching:\n"
            "  [bold]Shift+Tab[/bold]  - Cycle between modes\n"
            "  /mode        - Show current mode\n"
            "  /mode <name> - Switch to specific mode\n"
            "  /modes       - List all available modes\n\n"
            "Agentic:\n"
            "  /mode agentic - Persistent multi-step loop (auto-run)\n\n"
            "Commands:\n"
            "  /plan [task]     - Plan and execute a multi-step task\n"
            "  /ingest <path>   - Ingest a file or directory into the KB\n"
            "  /model           - Show current model\n"
            "  /model <name>    - Switch to a different model\n"
            "  /models          - List available Ollama models\n"
            "  /context         - Show context window size\n"
            "  /context <size>  - Set context window size\n"
            "  /clear           - Clear conversation history\n"
            "  /stats           - Show knowledge base stats\n"
            "  /help            - Show help\n"
            "  /quit            - Exit\n\n"
            "Shell commands:\n"
            "  !<command>     - Execute shell command (e.g., !ls, !pwd)",
            title="Welcome",
        )
    )

    last_interrupt_time = 0.0

    # Try to use prompt_toolkit for better key binding support
    prompt_session = _create_prompt_session(agent)

    while True:
        try:
            # Get current mode for prompt
            current_mode = agent.get_mode()
            mode_color = _get_mode_color(current_mode)
            prompt_color = _get_prompt_color(current_mode)

            if prompt_session:
                # Use prompt_toolkit with key bindings
                from prompt_toolkit.formatted_text import HTML
                prompt_text = HTML(f'<b><style fg="{prompt_color}">[{current_mode.value}]</style></b> > ')
                user_input = prompt_session.prompt(prompt_text)
            else:
                # Fallback to basic input
                console.print(f"[bold {mode_color}][{current_mode.value}][/bold {mode_color}] > ", end="")
                user_input = input()

            # Handle mode switch special case
            if user_input == '__MODE_SWITCH__':
                if agent.get_mode() == AgentMode.AGENTIC:
                    agent.set_agentic_approval_mode("auto")
                continue

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
            current_time = time.time()
            if current_time - last_interrupt_time < 2.0:
                # Double Ctrl+C within 2 seconds - exit
                console.print("\n")
                break
            last_interrupt_time = current_time
            console.print("\n[yellow]Press Ctrl+C again to exit.[/yellow]")
        except EOFError:
            break

    # Display session stats
    if _session_stats and _session_stats.queries > 0:
        _session_stats.display(console)

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
    mode: str = typer.Option(
        None, "--mode", help="Agent mode (chat, plan, ask, execute, code, image, research, debug, creative)"
    ),
) -> None:
    """Process a single query and exit."""
    from src.agent import Agent

    agent = Agent()

    if model:
        agent.set_model(model)
        console.print(f"[dim]Using model: {model}[/dim]")

    if mode:
        mode_enum = get_mode_by_name(mode)
        if mode_enum:
            agent.set_mode(mode_enum)
            console.print(f"[dim]Using mode: {mode}[/dim]")
        else:
            console.print(f"[yellow]Unknown mode '{mode}', using default[/yellow]")

    result = agent.query(text, include_history=not no_history)

    console.print(Panel(result["response"], title="Response"))
    console.print(f"[dim]Provider: {result['provider']} | Mode: {result.get('mode', 'chat')} | Documents: {result['documents_used']}[/dim]")


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
    error = None
    start_time = time.time()
    is_agentic_mode = agent.get_mode() == AgentMode.AGENTIC

    # Set up status handler to capture processing steps
    status_handler = StatusHandler()
    agent_logger = logging.getLogger("src.agent.nodes")
    llm_logger = logging.getLogger("src.llm")
    agent_logger.addHandler(status_handler)
    llm_logger.addHandler(status_handler)
    agent_logger.setLevel(logging.INFO)
    llm_logger.setLevel(logging.INFO)

    def run_query():
        nonlocal result, error
        try:
            result = agent.query(query)
        except Exception as e:
            error = e

    if is_agentic_mode:
        run_query()
        elapsed_total = time.time() - start_time
    else:
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

    # Handle errors from the query thread
    if error is not None:
        error_msg = str(error)
        if "not found" in error_msg and "pulling" in error_msg:
            # Missing model error
            console.print(f"\n[bold red]Error:[/bold red] {error_msg}")
            console.print("[yellow]Hint:[/yellow] Run `ollama pull <model-name>` to download the model")
        elif "Connection refused" in error_msg or "not available" in error_msg.lower():
            console.print(f"\n[bold red]Error:[/bold red] Cannot connect to Ollama")
            console.print("[yellow]Hint:[/yellow] Make sure Ollama is running: `ollama serve`")
        else:
            console.print(f"\n[bold red]Error:[/bold red] {error_msg}")
        if _session_stats:
            _session_stats.record_error()
        return

    if result is None:
        console.print("\n[bold red]Error:[/bold red] Query failed with no result")
        if _session_stats:
            _session_stats.record_error()
        return

    response_text = result["response"]

    # Auto-apply file blocks when the request implies file edits
    if result.get("provider") != "shell":
        file_blocks = _extract_file_blocks(response_text)
        if file_blocks and _looks_like_file_request(query):
            apply_results = _apply_file_blocks(file_blocks)
            response_text = _render_apply_summary(apply_results)

    # Check if agent is asking for clarification
    if response_text.strip().startswith("CLARIFY:"):
        clarify_question = response_text.strip()[8:].strip()
        console.print(f"\n[bold yellow]Clarification needed:[/bold yellow] {clarify_question}")
        clarification = Prompt.ask("[bold cyan]Your answer[/bold cyan]")

        # Record clarification stat
        if _session_stats:
            _session_stats.record_clarification()

        # Run follow-up query with clarification
        follow_up = f"{query} (Clarification: {clarification})"
        console.print()
        _process_query(agent, follow_up)
        return

    console.print("\n[bold green]Agent[/bold green]")

    # For shell commands, preserve raw output formatting; for others use Markdown
    if result["provider"] == "shell":
        console.print(response_text)
    else:
        console.print(Markdown(response_text))

    # Determine grounding status
    is_local_provider = result["provider"] in ("ollama", "local", "shell", "agentic")
    has_kb_docs = result["documents_used"] > 0
    has_tools = len(result["tool_results"]) > 0

    if result["provider"] == "shell":
        grounding = "[cyan]Shell[/cyan]"
    elif result["provider"] == "agentic":
        grounding = "[magenta]Agentic[/magenta]"
    elif has_kb_docs or has_tools:
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
    if result.get("steps_executed"):
        meta_parts.append(f"Steps: {result['steps_executed']}")

    # Add model name
    if result["provider"] != "shell":
        if is_local_provider:
            current_model = agent.get_model()
            meta_parts.append(f"Model: {current_model}")
        else:
            # External provider - show provider name as model
            meta_parts.append(f"Model: {result['provider']}")

    # Add token usage and context window info
    tokens = result.get("tokens_used", 0)
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)

    if tokens and _session_stats and _session_stats.context_window_size:
        percentage = (tokens / _session_stats.context_window_size) * 100
        if input_tokens and output_tokens:
            meta_parts.append(f"Tokens: {tokens:,} ({percentage:.1f}%) [↓{input_tokens:,} ↑{output_tokens:,}]")
        else:
            meta_parts.append(f"Tokens: {tokens:,} ({percentage:.1f}%)")

    meta_parts.append(f"Time: {elapsed_total:.1f}s")

    console.print(f"[dim]{' | '.join(meta_parts)}[/dim]")

    # Record session stats
    if _session_stats:
        _session_stats.record_query(result, elapsed_total)


def _execute_command(agent, cmd: str) -> dict:
    """
    Execute a shell command and return the result.
    Returns dict with 'output', 'success', 'blocked', 'error'.
    """
    result = agent.query(f"!{cmd}")
    response = result.get("response", "")

    is_blocked = "[BLOCKED]" in response
    is_error = "[EXIT CODE:" in response and "[EXIT CODE: 0]" not in response
    is_permission_denied = "permission denied" in response.lower() or "Operation not permitted" in response

    return {
        "command": cmd,
        "output": response,
        "success": not is_blocked and not is_error,
        "blocked": is_blocked,
        "permission_denied": is_permission_denied,
        "error": is_error,
    }


def _handle_plan_mode(agent, task: str) -> None:
    """
    Handle planning mode - generate a plan, get approval, then execute.
    """
    import re
    from src.llm.selector import ProviderSelector
    from langchain_core.messages import HumanMessage, SystemMessage

    console.print(f"\n[bold cyan]Planning:[/bold cyan] {task}\n")

    # Track planning time
    plan_start = time.time()

    # Generate the plan
    selector = ProviderSelector()

    plan_prompt = f"""Create a step-by-step plan to accomplish this task: {task}

STRICT FORMAT RULES:
1. Output ONLY a numbered list (1. 2. 3. etc.)
2. For ANY step involving a shell command, format EXACTLY as: $command - description
3. The $ prefix is REQUIRED for all commands - this is how they get executed
4. Keep it concise: 3-7 steps maximum
5. Use actual executable commands, not descriptions of commands

CORRECT format examples:
1. $uptime - check system uptime
2. $free -h - check memory usage
3. $df -h - check disk space
4. $top -bn1 | head -20 - check CPU usage
5. $ip addr - check network interfaces
6. $journalctl -n 50 --no-pager - check recent logs

WRONG format (DO NOT USE):
- "Check CPU with top command" (missing $ prefix)
- "Run the free command" (missing $ prefix)
- "Use df -h to check disk" (missing $ prefix)

Task: {task}"""

    messages = [
        SystemMessage(content="You are a plan generator. Output ONLY numbered steps. For shell commands, ALWAYS use the format: $command - description. Never describe commands without the $ prefix."),
        HumanMessage(content=plan_prompt),
    ]

    console.print("[dim]Generating plan...[/dim]")
    result = selector.generate(messages, force_local=True)
    plan_text = result.content.strip()
    plan_time = time.time() - plan_start

    # Display the plan
    console.print(Panel(plan_text, title=f"[bold]Proposed Plan[/bold] [dim](generated in {plan_time:.1f}s)[/dim]", border_style="cyan"))

    # Get user approval
    console.print("\n[bold]Options:[/bold]")
    console.print("  [green]y/Enter[/green] - Approve and execute (step by step)")
    console.print("  [green]a[/green]       - Approve and execute all (no prompts)")
    console.print("  [red]n[/red]       - Cancel")
    console.print("  [yellow]e[/yellow]       - Edit task and regenerate")

    choice = Prompt.ask("\n[bold cyan]Your choice[/bold cyan]", default="y").strip()

    if choice.lower() == "n":
        console.print("[yellow]Plan cancelled.[/yellow]")
        return

    # Handle edit - either "e" alone or "e <new task>"
    if choice.lower() == "e" or choice.lower().startswith("e "):
        if choice.lower().startswith("e "):
            new_task = choice[2:].strip()
        else:
            new_task = Prompt.ask("[bold cyan]Enter modified task[/bold cyan]")
        if new_task.strip():
            _handle_plan_mode(agent, new_task.strip())
        return

    # Determine execution mode
    run_all = choice.lower() == "a"

    # Execute the plan
    console.print("\n[bold green]Executing plan...[/bold green]\n")
    exec_start = time.time()

    # Parse steps from the plan
    steps = re.findall(r'^\d+\.\s*(.+)$', plan_text, re.MULTILINE)
    if not steps:
        steps = [line.strip() for line in plan_text.split('\n') if line.strip() and line.strip()[0].isdigit()]
        steps = [re.sub(r'^\d+\.\s*', '', s) for s in steps]

    # Collect results for post-execution report
    execution_results = []
    stopped_early = False

    for i, step in enumerate(steps, 1):
        console.print(f"\n[bold cyan]Step {i}/{len(steps)}:[/bold cyan] {step}")

        # Check if step contains a shell command (marked with $)
        cmd_match = re.search(r'\$\s*(.+)', step)
        if cmd_match:
            cmd = cmd_match.group(1).strip()
            if ' - ' in cmd:
                cmd = cmd.split(' - ')[0].strip()
            console.print(f"[dim]Running: {cmd}[/dim]")

            # Execute and capture result
            cmd_result = _execute_command(agent, cmd)

            # Display output
            if cmd_result["blocked"]:
                console.print(f"[yellow]⚠ Command blocked[/yellow]")
            elif cmd_result["success"]:
                console.print(cmd_result["output"])
            else:
                console.print(cmd_result["output"])

            # Handle blocked or permission denied - offer sudo retry
            needs_sudo_retry = (
                cmd_result["blocked"] or
                cmd_result["permission_denied"]
            )
            if needs_sudo_retry and not cmd.startswith("sudo "):
                retry = Prompt.ask(
                    "[yellow]Retry with sudo?[/yellow] [y/n]",
                    default="y"
                ).strip().lower()
                if retry == "y":
                    console.print(f"[dim]Running: sudo {cmd}[/dim]")
                    cmd_result = _execute_command(agent, f"sudo {cmd}")
                    if cmd_result["blocked"]:
                        console.print(f"[yellow]⚠ Command still blocked[/yellow]")
                    else:
                        console.print(cmd_result["output"])

            execution_results.append({
                "step": i,
                "description": step,
                "command": cmd,
                "result": cmd_result,
            })
        else:
            # Non-command step - process as query
            _process_query(agent, step)
            execution_results.append({
                "step": i,
                "description": step,
                "command": None,
                "result": {"success": True, "output": "Processed as query"},
            })

        # Prompt between steps (unless run_all mode)
        if not run_all and i < len(steps):
            continue_choice = Prompt.ask(
                "[dim]Enter=continue, 's'=skip remaining, 'q'=quit[/dim]",
                default=""
            ).strip().lower()
            if continue_choice == "q":
                console.print("[yellow]Plan execution stopped.[/yellow]")
                stopped_early = True
                break
            if continue_choice == "s":
                console.print("[yellow]Skipping remaining steps.[/yellow]")
                stopped_early = True
                break

    exec_time = time.time() - exec_start

    # Generate post-execution report
    console.print("\n" + "=" * 60)
    console.print("[bold]Execution Report[/bold]")
    console.print("=" * 60)

    # Summary stats
    completed = len(execution_results)
    successful = sum(1 for r in execution_results if r["result"].get("success", False))
    blocked = sum(1 for r in execution_results if r["result"].get("blocked", False))
    failed = completed - successful - blocked

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Steps completed: {completed}/{len(steps)}")
    console.print(f"  [green]Successful: {successful}[/green]")
    if blocked > 0:
        console.print(f"  [yellow]Blocked: {blocked}[/yellow]")
    if failed > 0:
        console.print(f"  [red]Failed: {failed}[/red]")
    console.print(f"  Planning time: {plan_time:.1f}s")
    console.print(f"  Execution time: {exec_time:.1f}s")

    # Collect outputs for analysis (truncate to keep analysis fast)
    MAX_OUTPUT_PER_COMMAND = 2000  # chars per command output
    outputs_for_analysis = []
    for r in execution_results:
        if r["command"] and r["result"].get("output"):
            output = r["result"]["output"]
            # Clean up the output (remove the $ command prefix line)
            if output.startswith("$ "):
                lines = output.split("\n", 1)
                if len(lines) > 1:
                    output = lines[1]
            # Truncate long outputs to speed up analysis
            if len(output) > MAX_OUTPUT_PER_COMMAND:
                output = output[:MAX_OUTPUT_PER_COMMAND] + f"\n... (truncated {len(output) - MAX_OUTPUT_PER_COMMAND} chars)"
            outputs_for_analysis.append(f"[{r['description']}]\n{output}")

    # Analyze results if we have command outputs
    if outputs_for_analysis and not stopped_early:
        console.print(f"\n[bold]Analysis:[/bold]")
        analysis_start = time.time()

        # Show spinner while analyzing
        with console.status("[dim]Analyzing results...[/dim]", spinner="dots"):
            analysis_prompt = f"""Based on the following command outputs from a "{task}" task, provide a brief analysis:

{chr(10).join(outputs_for_analysis)}

Provide:
1. A brief overall assessment (1-2 sentences)
2. Any issues or warnings noticed (bullet points)
3. Recommendations if any (bullet points)
4. A single follow-up question to help guide next steps (start with "Would you like to...")

Be concise and focus on actionable insights."""

            analysis_messages = [
                SystemMessage(content="You are a system administrator analyzing command outputs. Be concise and practical. Always end with a helpful follow-up question."),
                HumanMessage(content=analysis_prompt),
            ]

            try:
                analysis_result = selector.generate(analysis_messages, force_local=True)
                analysis_time = time.time() - analysis_start
                console.print(f"[dim](analyzed in {analysis_time:.1f}s)[/dim]\n")
                console.print(Markdown(analysis_result.content))

                # Add plan results to conversation history for follow-up questions
                from langchain_core.messages import HumanMessage as HM, AIMessage as AM
                plan_summary = f"Plan executed: {task}\n\nResults:\n" + chr(10).join(outputs_for_analysis[:3])  # First 3 outputs
                agent._conversation_history.extend([
                    HM(content=f"/plan {task}"),
                    AM(content=f"[Plan executed with {successful} successful, {blocked} blocked steps]\n\nAnalysis:\n{analysis_result.content}"),
                ])
            except Exception as e:
                console.print(f"[yellow]Could not generate analysis: {e}[/yellow]")

    # Record plan execution in session stats
    if _session_stats:
        _session_stats.record_plan()

    console.print("\n[bold green]Plan completed![/bold green]")


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
        # Interactive model selector
        selected = _interactive_model_selector(agent)
        if selected:
            agent.set_model(selected)
            console.print(f"[green]Switched to model: {selected}[/green]")
            # Save state
            _save_state(selected, agent.get_mode().value)
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
        # Save state
        _save_state(model_name, agent.get_mode().value)
        return True

    if cmd_lower == "/stats":
        from src.knowledge import VectorStore

        vs = VectorStore()
        stats = vs.get_collection_stats()
        console.print(f"Documents in KB: {stats['document_count']}")
        console.print(f"Conversation history: {agent.get_history_length()} messages")
        console.print(f"Current model: {agent.get_model()}")
        console.print(f"Current mode: {agent.get_mode().value}")
        console.print(f"Context window: {agent.get_context_window():,} tokens")
        return True

    if cmd_lower == "/ingest" or cmd_lower.startswith("/ingest "):
        args = shlex.split(cmd)
        recursive = True
        exclude: list[str] = []
        glob: str | None = None
        text_content: str | None = None
        source = "cli-input"
        path_arg: str | None = None

        idx = 1
        while idx < len(args):
            token = args[idx]
            if token in ("--recursive", "-r"):
                recursive = True
                idx += 1
                continue
            if token in ("--no-recursive", "-R"):
                recursive = False
                idx += 1
                continue
            if token in ("--exclude", "-e"):
                if idx + 1 >= len(args):
                    console.print("[yellow]Missing value for --exclude[/yellow]")
                    return True
                exclude.append(args[idx + 1])
                idx += 2
                continue
            if token == "--glob":
                if idx + 1 >= len(args):
                    console.print("[yellow]Missing value for --glob[/yellow]")
                    return True
                glob = args[idx + 1]
                idx += 2
                continue
            if token == "--text":
                if idx + 1 >= len(args):
                    console.print("[yellow]Missing value for --text[/yellow]")
                    return True
                text_content = args[idx + 1]
                idx += 2
                continue
            if token == "--source":
                if idx + 1 >= len(args):
                    console.print("[yellow]Missing value for --source[/yellow]")
                    return True
                source = args[idx + 1]
                idx += 2
                continue

            if path_arg is None:
                path_arg = token
                idx += 1
                continue

            console.print(f"[yellow]Unexpected argument: {token}[/yellow]")
            return True

        from src.ingestion import IngestionPipeline

        pipeline = IngestionPipeline()

        if text_content is not None:
            stats = pipeline.ingest_text(text_content, source=source)
            _print_ingest_stats(stats)
            return True

        if not path_arg and not glob:
            console.print("[yellow]Usage:[/yellow] /ingest <path> [--recursive|--no-recursive] [--exclude PATTERN]")
            console.print("[yellow]   or:[/yellow] /ingest --glob \"<pattern>\" [path]")
            console.print("[yellow]   or:[/yellow] /ingest --text \"<content>\" [--source name]")
            console.print("[yellow]Examples:[/yellow]")
            console.print("  /ingest README.md")
            console.print("  /ingest docs --recursive")
            console.print("  /ingest --glob \"*.md\" .")
            console.print("  /ingest --text \"notes\" --source scratchpad")
            return True

        base_path = Path(path_arg) if path_arg else settings.project_root
        if not base_path.is_absolute():
            base_path = (settings.project_root / base_path).resolve()

        if glob:
            if not base_path.exists():
                console.print(f"[red]Error:[/red] Path not found: {base_path}")
                return True
            matches = [p for p in base_path.rglob(glob) if p.is_file()]
            if not matches:
                console.print(f"[yellow]No files matched pattern '{glob}' under {base_path}[/yellow]")
                return True
            stats_list = [pipeline.ingest_file(path) for path in matches]
            merged = _merge_ingest_stats(stats_list)
            _print_ingest_stats(merged)
            return True

        if not base_path.exists():
            console.print(f"[red]Error:[/red] Path not found: {base_path}")
            return True

        if base_path.is_file():
            stats = pipeline.ingest_file(base_path)
            _print_ingest_stats(stats)
            return True

        if base_path.is_dir():
            stats = pipeline.ingest_directory(
                base_path,
                recursive=recursive,
                exclude_patterns=exclude or None,
            )
            _print_ingest_stats(stats)
            return True

        console.print(f"[red]Error:[/red] Unsupported path: {base_path}")
        return True

    if cmd_lower == "/context":
        current_ctx = agent.get_context_window()
        console.print(f"Current context window: [cyan]{current_ctx:,}[/cyan] tokens")
        console.print("\nCommon sizes:")
        console.print("  4096   - Small (faster, less memory)")
        console.print("  8192   - Default (balanced)")
        console.print("  16384  - Large (more history)")
        console.print("  32768  - Very large (long conversations)")
        console.print("  65536  - Huge (maximum history)")
        console.print("\n[dim]Use /context <size> to change[/dim]")
        return True

    if cmd_lower.startswith("/context "):
        size_str = cmd[9:].strip()
        if not size_str:
            console.print("[yellow]Usage: /context <size>[/yellow]")
            return True

        try:
            new_size = int(size_str.replace(",", "").replace("_", ""))
            agent.set_context_window(new_size)
            console.print(f"[green]Context window set to {new_size:,} tokens[/green]")
            console.print("[yellow]Note: This will take effect on the next query[/yellow]")
            # Update session stats
            if _session_stats:
                _session_stats.context_window_size = new_size
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
        return True

    if cmd_lower == "/mode":
        current_mode = agent.get_mode()
        mode_config = get_mode_config(current_mode)
        mode_color = _get_mode_color(current_mode)
        console.print(f"Current mode: [{mode_color}]{current_mode.value}[/{mode_color}]")
        console.print(f"Description: {mode_config.description}")
        console.print(f"Routing bias: {mode_config.routing_bias or 'balanced'}")
        console.print(f"Temperature: {mode_config.temperature}")
        console.print(f"Verbose: {mode_config.verbose}")
        if current_mode == AgentMode.AGENTIC:
            console.print(f"Approval mode: {agent.get_agentic_approval_mode()}")
        return True

    if cmd_lower.startswith("/mode "):
        mode_name = cmd[6:].strip()
        if not mode_name:
            console.print("[yellow]Usage: /mode <mode_name>[/yellow]")
            return True

        mode_enum = get_mode_by_name(mode_name)
        if mode_enum:
            agent.set_mode(mode_enum)
            mode_config = get_mode_config(mode_enum)
            mode_color = _get_mode_color(mode_enum)
            console.print(f"[{mode_color}]→ {mode_enum.value}[/{mode_color}] [dim]{mode_config.description}[/dim]")
            if mode_enum == AgentMode.AGENTIC:
                agent.set_agentic_approval_mode("auto")
            # Save state
            _save_state(agent.get_model(), mode_enum.value)
        else:
            console.print(f"[yellow]Unknown mode: {mode_name}[/yellow]")
            console.print("Use /modes to see available modes.")
        return True

    if cmd_lower == "/modes":
        console.print("[bold]Available modes:[/bold]")
        console.print("  Use [bold]Shift+Tab[/bold] to cycle or [bold]/mode <name>[/bold] to switch\n")

        current_mode = agent.get_mode()
        for mode, description in list_modes():
            mode_color = _get_mode_color(mode)
            if mode == current_mode:
                console.print(f"  [{mode_color}]* {mode.value:10}[/{mode_color}] - {description} [dim](current)[/dim]")
            else:
                console.print(f"    [{mode_color}]{mode.value:10}[/{mode_color}] - {description}")
        return True

    if cmd_lower == "/plan" or cmd_lower.startswith("/plan "):
        # Extract task description if provided
        task = cmd[5:].strip() if len(cmd) > 5 else ""
        if not task:
            task = Prompt.ask("[bold cyan]What would you like to plan?[/bold cyan]")
            if not task.strip():
                console.print("[yellow]No task provided.[/yellow]")
                return True

        _handle_plan_mode(agent, task)
        return True

    if cmd_lower == "/help":
        console.print(
            Panel(
                "Mode commands:\n"
                "  [bold]Shift+Tab[/bold]    - Cycle between modes\n"
                "  /mode          - Show current mode\n"
                "  /mode <name>   - Switch to specific mode\n"
                "  /modes         - List all available modes\n\n"
                "Available commands:\n"
                "  /plan [task]     - Enter planning mode to create and execute a plan\n"
                "  /ingest <path>   - Ingest a file or directory into the KB\n"
                "  /model           - Show current model\n"
                "  /model <name>    - Switch to a different model\n"
                "  /models          - List available Ollama models\n"
                "  /context         - Show/modify context window size\n"
                "  /context <size>  - Set context window (e.g., 4096, 8192, 32768)\n"
                "  /clear           - Clear conversation history\n"
                "  /stats           - Show knowledge base and conversation stats\n"
                "  /help            - Show this help message\n"
                "  /quit            - Exit the chat\n\n"
                "Shell commands:\n"
                "  !<command>     - Execute a shell command directly\n"
                "                   Examples: !ls, !pwd, !cat file.txt\n\n"
                "Modes:\n"
                "  chat     - General conversation (default)\n"
                "  plan     - Multi-step task planning\n"
                "  agentic  - Persistent multi-step loop with tools\n"
                "  ask      - Knowledge retrieval\n"
                "  execute  - Tool/bash execution\n"
                "  code     - Programming assistance\n"
                "  image    - Image generation (requires SD)\n"
                "  research - Web search and synthesis\n"
                "  debug    - Verbose tracing\n"
                "  creative - High creativity responses\n\n"
                "Tips:\n"
                "  - The agent will ask for clarification when queries are ambiguous\n"
                "  - Use ! prefix to run commands without LLM interpretation\n"
                "  - Use /plan to break complex tasks into steps before execution\n"
                "  - Different modes optimize routing for specific tasks",
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
