# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"    # with dev dependencies

# Run
python scripts/run.py chat          # interactive chat
python scripts/run.py serve         # REST API at :8000
python scripts/run.py check         # verify Ollama/ChromaDB status

# Ingest documents
python scripts/ingest.py directory ./docs
python scripts/ingest.py file ./document.pdf
python scripts/ingest.py stats

# Development
pytest                    # run tests
ruff check .              # lint
ruff format .             # format
mypy src                  # type check
```

## Architecture

This is a local-first LLM agent using LangGraph for workflow orchestration. The agent prioritizes Ollama for local inference and falls back to external providers (Claude, GPT-4, etc.) when local responses are uncertain.

### Core Flow

1. **Router** (`src/agent/nodes.py:route_query`) - Classifies query as needing retrieval, tool execution, or direct generation
2. **Retriever** (`src/knowledge/retriever.py`) - Fetches documents from ChromaDB vector store
3. **Grader** (`src/knowledge/grader.py`) - LLM-based relevance scoring; triggers query rewrite or external fallback if documents are irrelevant
4. **Generator** (`src/agent/nodes.py:generate_local`) - Produces response using context
5. **KB Updater** (`src/knowledge/updater.py`) - Extracts facts from external provider responses and stores them locally

### Key Components

- **`src/agent/graph.py`** - LangGraph StateGraph definition with conditional edges for routing decisions
- **`src/agent/state.py`** - `AgentStateDict` TypedDict that flows through the graph
- **`src/llm/selector.py`** - Provider selection logic; tries Ollama first, estimates confidence, triggers fallback chain via LiteLLM
- **`src/llm/local.py`** - Ollama wrapper with confidence estimation based on uncertainty phrases
- **`src/knowledge/vectorstore.py`** - ChromaDB wrapper with content-hash deduplication
- **`src/tools/bash.py`** - Sandboxed bash execution with blocked command patterns and approval flow

### Configuration

- **`config/settings.py`** - Pydantic Settings loading from `.env`
- **`config/providers.yaml`** - LiteLLM fallback chain configuration

### Interfaces

- **CLI**: `scripts/run.py` (Typer) - chat, serve, query, check commands
- **API**: `src/api/routes.py` (FastAPI) - /api/v1/query, /api/v1/ingest/*, /api/v1/kb/*
