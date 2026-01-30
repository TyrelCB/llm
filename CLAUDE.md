# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"    # with dev dependencies

# Run (scripts or installed entry points)
python scripts/run.py chat          # interactive chat
python scripts/run.py serve         # REST API at :8000
python scripts/run.py check         # verify Ollama/ChromaDB status
llm-agent chat                      # same as above (after pip install)
llm-ingest directory ./docs         # ingest documents (entry point)

# Ingest documents
python scripts/ingest.py directory ./docs
python scripts/ingest.py file ./document.pdf
python scripts/ingest.py stats

# Development
pytest                              # run all tests
pytest tests/test_foo.py            # run single test file
pytest tests/test_foo.py::test_bar  # run single test function
pytest -k "keyword"                 # run tests matching keyword
ruff check .                        # lint
ruff format .                       # format
mypy src                            # type check
```

## External Dependencies

- **Ollama** (required) - Local LLM inference at `localhost:11434`
- **ChromaDB** - Vector store for RAG, created automatically in `data/chroma_db/`
- **DDGS** (required for research mode) - DuckDuckGo search API (`pip install ddgs`)
- **ComfyUI/Automatic1111** (optional) - Local Stable Diffusion for image generation
- **SearXNG** (optional) - Self-hosted search, falls back to DuckDuckGo

## Architecture

This is a local-first LLM agent using LangGraph for workflow orchestration. The agent prioritizes Ollama for local inference and falls back to external providers (Claude, GPT-4, etc.) when local responses are uncertain.

### Mode System

The agent supports 10 modes, each optimizing routing and behavior for specific tasks:

| Mode | Purpose | Routing Bias | Temperature |
|------|---------|--------------|-------------|
| **chat** (default) | General conversation | Balanced | 0.7 |
| **plan** | Multi-step task planning | generate | 0.3 |
| **agentic** | Agentic loop control | Balanced | 0.2 |
| **ask** | KB retrieval + knowledge | retrieve | 0.5 |
| **execute** | Tool/bash execution | tool | 0.3 |
| **code** | Programming assistance | generate | 0.3 |
| **image** | Local Stable Diffusion | image | 0.7 |
| **research** | Web search + synthesis | web | 0.5 |
| **debug** | Verbose tracing | any | 0.5 |
| **creative** | Exploratory responses | generate | 0.9 |

Switch modes with `Shift+Tab` or `/mode <name>`. Mode config in `src/agent/modes.py`.

### Core Flow

1. **Router** (`src/agent/nodes.py:route_query`) - Classifies query as needing retrieval, tool execution, web search, image generation, or direct generation. Mode bias influences routing.
2. **Retriever** (`src/knowledge/retriever.py`) - Fetches documents from ChromaDB vector store
3. **Grader** (`src/knowledge/grader.py`) - LLM-based relevance scoring; triggers query rewrite or external fallback if documents are irrelevant
4. **Generator** (`src/agent/nodes.py:generate_local`) - Produces response using context
5. **Web Search** (`src/agent/nodes.py:web_search`) - DuckDuckGo/SearXNG search with synthesis
6. **Image Gen** (`src/agent/nodes.py:generate_image`) - Local Stable Diffusion via ComfyUI/A1111
7. **KB Updater** (`src/knowledge/updater.py`) - Extracts facts from external provider responses and stores them locally

### Clarification Mechanism

The agent can request clarification when queries are ambiguous. Responses starting with `CLARIFY:` trigger the CLI to prompt the user for more info, then rerun with the original query plus clarification appended.

### Key Components

- **`src/agent/graph.py`** - LangGraph StateGraph definition with conditional edges for routing decisions
- **`src/agent/state.py`** - `AgentStateDict` TypedDict that flows through the graph, includes mode
- **`src/agent/modes.py`** - Mode definitions, configs, and cycling logic
- **`src/agent/agentic_loop.py`** - Agentic loop with persistent state and tool orchestration
- **Agentic health summaries** - Deterministic summary for standard health commands (uptime/df/free/ps)
- **Agentic codebase summaries** - Deterministic overview when reading README/AGENTS/CLAUDE
- **Agentic JSON retry** - Retries once if model output isn't valid JSON
- **Agentic no-match guard** - Stops repeated empty searches and suggests a review workflow
- **Agentic bootstrap** - Reads README/AGENTS/CLAUDE for repo overview questions before LLM
- **Agentic multi-goal** - Splits tasks on "then/and then/after that" and runs goals sequentially
- **Agentic commas** - Splits chained prompts on commas when phrased as multiple tasks
- **Agentic tool validation** - Rejects malformed tool args and retries
- **Agentic review bootstrap** - Starts code review goals with TODO/FIXME/BUG scan
- **Agentic pytest bootstrap** - Runs pytest once for test-related goals and summarizes output
- **Agentic TTS bootstrap** - Generates audio for TTS goals, stores last output path, and plays via available system tools (non-blocking)
- **Agentic timing** - Logs tool/step durations for better visibility
- **`src/llm/selector.py`** - Provider selection logic; tries Ollama first, estimates confidence, triggers fallback chain via LiteLLM
- **`src/llm/local.py`** - Ollama wrapper with confidence estimation based on uncertainty phrases
- **`src/knowledge/vectorstore.py`** - ChromaDB wrapper with content-hash deduplication
- **`src/tools/bash.py`** - Sandboxed bash execution with blocked command patterns and approval flow
- **`src/tools/registry.py`** - Tool registry (bash, read_file, list_dir, search, write_file, tts service fallback)
- **`src/tools/web.py`** - Web search via DuckDuckGo or SearXNG
- **`src/tools/image.py`** - Image generation via ComfyUI or Automatic1111
- **`.agent/`** - Agentic loop state, logs, and scratch outputs

### Configuration

- **`config/settings.py`** - Pydantic Settings loading from `.env`
- **`config/providers.yaml`** - LiteLLM fallback chain configuration

### Interfaces

- **CLI**: `scripts/run.py` (Typer) - chat, serve, query, check commands
  - Runtime model switching: `/model <name>` or `--model` flag (applies to planner/agentic helpers)
  - Project root defaults to cwd for CLI; override with `--project-root` or `PROJECT_ROOT`
  - Code mode auto-applies `FILE:` blocks to disk
  - Mode cycling: `Shift+Tab` or `/mode <name>`
  - Shell commands: `!command` prefix for direct execution
  - Plan mode: `/plan [task]` for multi-step task planning
  - Agentic mode: `/mode agentic` for multi-step tool orchestration (step or auto approval)
- **API**: `src/api/routes.py` (FastAPI) - /api/v1/query, /api/v1/ingest/*, /api/v1/kb/*, /api/v1/model

## Development Workflow

When making changes to this repository:

1. **Sync with remote** - Ensure repo is up to date: `git pull origin master`
2. **Document plan** - Write implementation plan in `plan.md` before starting work
3. **Implement changes** - Make changes and update `plan.md` to track progress
4. **Update version** - Bump version in `pyproject.toml`
5. **Update documentation** - Update README.md, CLAUDE.md, and any other relevant docs
6. **Commit and push** - `git add . && git commit -m "message" && git push`
