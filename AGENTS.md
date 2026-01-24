# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the core application code (agent graph, LLM selection, tools, ingestion, and API).
- `config/` contains runtime settings (`settings.py`) and provider config (`providers.yaml`).
- `scripts/` provides CLI entry points (`run.py`, `ingest.py`).
- `data/` is the local working storage for documents and the ChromaDB vector store.

## Build, Test, and Development Commands
- `pip install -e .` installs the app in editable mode.
- `pip install -e ".[dev]"` adds dev tools (ruff, mypy, pytest).
- `python scripts/run.py chat` starts the interactive CLI.
- `python scripts/run.py serve` starts the FastAPI server on `:8000`.
- `python scripts/run.py check` validates Ollama/ChromaDB availability.
- `python scripts/ingest.py directory ./docs` ingests content for the KB.

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation.
- Ruff enforces formatting and linting: `ruff format .` and `ruff check .`.
- Mypy is strict: `mypy src`.
- Use descriptive, module-scoped names (e.g., `vectorstore.py`, `retriever.py`).

## Testing Guidelines
- Tests live under `tests/` and run with `pytest`.
- Prefer `test_<feature>.py` and `test_<behavior>` function names.
- Run coverage only when requested; no explicit threshold is configured.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries (e.g., "Add runtime model switching").
- Include context in the PR description: what changed, why, and how to verify.
- Link related issues, and add screenshots or logs for UI/CLI changes when useful.

## Configuration & Security Notes
- Copy `.env.example` to `.env` for local settings; avoid committing secrets.
- External provider API keys are optional and only needed for fallback behavior.

## Agent-Specific Instructions
- When making changes, keep a `plan.md` implementation plan and update it as you go.
- Bump the version in `pyproject.toml` and update docs (`README.md`, `CLAUDE.md`) when behavior changes.
