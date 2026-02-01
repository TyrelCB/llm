# Local LLM Agent

A local-first LLM agent with RAG-based knowledge base, tool execution, and intelligent fallback to external providers.

## What's New in 0.8.10

- **📁 Project-Aware CLI**: Defaults file tools and local data to the current working directory (override with `--project-root` or `PROJECT_ROOT`)
- **🧾 Code Mode File Writes**: Code mode outputs `FILE:` blocks that the CLI applies directly to disk
- **📄 Cross-Mode File Writes**: File-create requests in other modes use the code-mode prompt and apply `FILE:` blocks automatically
- **⚙️ Agentic Auto Mode**: Agentic runs always execute in auto mode without approval prompts (no shift+tab interruption)
- **📥 /ingest Command**: Ingest a file, directory, or glob of files into the KB from chat
- **🧩 Ingest Dedup Fix**: Avoid duplicate IDs when ingesting identical chunks in a single batch
- **🤖 Agentic Mode**: Compact, persistent control loop with one-tool-per-step execution
- **📁 Agent State on Disk**: `.agent/` state, logs, and scratch outputs for long-running tasks
- **🧰 Minimal Tool Set**: Added `search` and `write_file` tools for grounded workflows
- **✅ Dual Execution Modes**: Per-step approvals or auto-run until completion
- **⚡ Agentic Health Summary**: Fast, deterministic summary for common system health checks
- **📦 Codebase Summary**: Deterministic overview when reading `README.md`/`AGENTS.md`
- **🔎 Smarter Search**: `search` now supports file globs like `*.py` and ignores `.agent/` and `.git/`
- **🧾 JSON Retry Guard**: Agentic loop retries once if model output isn't valid JSON
- **🧯 No-Match Guard**: Stops repeated empty searches and suggests a real review path
- **📚 Codebase Bootstrap**: Forces README/AGENTS/CLAUDE read before LLM for repo questions
- **☕ Multi-Goal Runs**: Chain tasks with “then/and then/after that” and auto-advance goals
- **✂️ Comma Chaining**: Split multi-goal prompts on commas for fire-and-forget tasks
- **🧰 Tool Arg Validation**: Rejects malformed tool calls and retries
- **🧪 Review Bootstrap**: Starts code-review goals with a TODO/FIXME/BUG scan
- **🧪 Pytest Bootstrap**: Runs pytest once for test-related goals with a summary
- **🔊 TTS Tool + Service**: Adds a Qwen3-TTS service and `tts` tool (service-first with local fallback)
- **🔈 TTS Bootstrap**: Detects TTS requests, generates audio, and tracks last output for playback
- **🔉 TTS Playback Fallbacks**: Remembers last audio path across runs and plays via available system tools (non-blocking)
- **⏱️ Agentic Timing**: Shows per-step tool/runtime durations in the CLI
- **🎯 10 Agent Modes**: Expanded mode set including agentic workflows

## Features

- **Local-First**: Prioritizes local Ollama models, only falling back to external providers when needed
- **RAG Knowledge Base**: ChromaDB-powered vector store with document ingestion and retrieval
- **Intelligent Routing**: LangGraph state machine routes queries to retrieval, tools, web search, or direct generation
- **10 Agent Modes**: Specialized modes (chat, plan, agentic, ask, execute, code, image, research, debug, creative)
- **Agentic Loop**: Persistent, capped control loop with compact state and strict JSON actions
- **On-Disk Agent State**: `.agent/` logs and summaries keep prompts tiny
- **Web Research**: DuckDuckGo search with page content crawling and source synthesis
- **Document Grading**: LLM-based relevance grading with automatic query rewriting
- **Multi-Provider Fallback**: Automatic fallback chain (Ollama → Claude → GPT-4 → Gemini → Grok)
- **Knowledge Base Updates**: Automatically extracts and stores facts from external provider responses
- **Sandboxed Tool Execution**: Safe bash command execution with validation and approval
- **TTS Integration**: Qwen3-TTS service with agent tool wrapper (service-first, local fallback)
- **Context Window Management**: Track token usage and dynamically adjust context window size
- **Runtime Model Switching**: Switch between Ollama models at runtime with interactive selector (applies to planner/agentic helpers)
- **Persistent State**: Remembers last used model and mode between sessions
- **Tab Completion**: Auto-complete commands, modes, and models
- **Conversation Memory**: Agent remembers previous exchanges within a session
- **Intelligent Suggestions**: Provides contextual follow-up suggestions after each response
- **Natural Language Tools**: LLM interprets queries like "check disk space" and generates appropriate commands
- **Multiple Interfaces**: Interactive CLI and REST API with real-time status display

## Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌───────────┐     ┌─────────┐
│ Router  │────▶│ Retriever │────▶│ Grader  │
└─────────┘     └───────────┘     └─────────┘
    │                                  │
    │ [tool needed]          ┌────────┴────────┐
    ▼                        ▼                 ▼
┌──────────┐         [docs relevant]    [docs irrelevant]
│Tool Exec │                │                  │
└──────────┘                ▼                  ▼
    │              ┌────────────┐       ┌───────────┐
    │              │ Local Gen  │       │ Rewrite   │──┐
    │              └────────────┘       └───────────┘  │
    │                    │                    ▲        │
    │                    │                    └────────┘
    │                    │              [max retries exceeded]
    │                    │                    │
    │                    │                    ▼
    │                    │           ┌────────────────┐
    │                    │           │External Fallback│
    │                    │           └────────────────┘
    │                    │                    │
    │                    │                    ▼
    │                    │           ┌────────────────┐
    │                    │           │  KB Updater    │
    │                    │           └────────────────┘
    │                    │                    │
    └────────────────────┴────────────────────┘
                         │
                         ▼
                    Response
```

## Agent Modes

The agent supports 10 specialized modes optimized for different tasks:

| Mode | Purpose | Routing Bias | Temperature | Use Case |
|------|---------|--------------|-------------|----------|
| **chat** | General conversation | Balanced | 0.7 | Default mode for mixed tasks |
| **plan** | Multi-step planning | generate | 0.3 | Breaking down complex tasks |
| **agentic** | Agentic loop control | Balanced | 0.2 | Multi-step tool orchestration |
| **ask** | Knowledge retrieval | retrieve | 0.5 | Querying the knowledge base |
| **execute** | Tool/bash execution | tool | 0.3 | Running system commands |
| **code** | Programming assistance | generate | 0.3 | Code generation and review |
| **image** | Image generation | image | 0.7 | Stable Diffusion prompts |
| **research** | Web search | web | 0.5 | Research with web crawling |
| **debug** | Verbose tracing | any | 0.5 | Debugging routing decisions |
| **creative** | Uncensored generation | generate | 0.9 | Creative and unrestricted output |

Switch modes with `Shift+Tab` or `/mode <name>`.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) (for local LLM)
- **Optional**: `ddgs` package for web research (`pip install ddgs`)

## Installation

### 1. Clone and Install

```bash
cd /home/tyrel/projects/llm
pip install -e .
```

### 2. Install and Configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mistral:7b
ollama pull nomic-embed-text

# Start Ollama (usually starts automatically)
ollama serve
```

### 3. Verify Installation

```bash
python scripts/run.py check
```

## Usage

### Interactive Chat

```bash
python scripts/run.py chat
```

By default, the CLI treats the current working directory as the project root for file tools and local data storage. Override with `--project-root` or `PROJECT_ROOT`.

Commands in chat:
- `/model` - Show current model
- `/model <name>` - Switch to a different model (e.g., `/model deepseek-r1:7b`)
- `/models` - Interactive model selector with numbered menu
- `/mode` - Show current mode
- `/mode <name>` - Switch to specific mode (chat, plan, ask, execute, code, image, research, debug, creative)
- `/modes` - List all available modes
- `/context` - Show current context window size
- `/context <size>` - Set context window (e.g., `/context 16384`)
- `/plan [task]` - Enter planning mode for multi-step tasks
- `/ingest <path>` - Ingest a file or directory into the knowledge base
- `/clear` - Clear conversation history
- `/stats` - Show knowledge base, model, and context stats
- `/help` - Show help
- `/quit` - Exit
- `Shift+Tab` - Cycle between modes
- `!<command>` - Execute shell command directly (e.g., `!ls -la`)

Code mode supports direct file writes: responses formatted with `FILE: path` blocks are applied to disk automatically.

The CLI shows:
- Current model, mode, and context window on welcome screen
- Persistent state (restores last model/mode on startup)
- Real-time elapsed time while processing
- Processing steps (routing, retrieving, grading, etc.)
- Token usage per query with input/output breakdown
- Context window utilization percentage
- Grounding status: `Grounded (KB)`, `Grounded (Tools)`, `Local`, `External`, or `Shell`
- Follow-up suggestions after each response

### Single Query

```bash
python scripts/run.py query "What is machine learning?"

# Use a specific model
python scripts/run.py query "Explain quantum computing" --model deepseek-r1:7b
```

### REST API

```bash
python scripts/run.py serve
```

API endpoints available at `http://localhost:8000/docs`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Query the agent (supports optional `model` field) |
| `/api/v1/model` | GET | Get current model |
| `/api/v1/model` | POST | Switch model |
| `/api/v1/models` | GET | List available Ollama models |
| `/api/v1/ingest/text` | POST | Ingest raw text |
| `/api/v1/ingest/file` | POST | Ingest a file |
| `/api/v1/ingest/directory` | POST | Ingest a directory |
| `/api/v1/kb/stats` | GET | Get KB statistics |
| `/api/v1/kb/search` | POST | Search the KB |
| `/api/v1/kb/clear` | DELETE | Clear the KB |
| `/api/v1/history/clear` | POST | Clear chat history |

### Document Ingestion

```bash
# Ingest a directory (recursive by default)
python scripts/ingest.py directory ./docs

# Ingest a single file
python scripts/ingest.py file ./document.pdf

# Ingest raw text
python scripts/ingest.py text "Some content to remember" --source "manual-input"

# View KB statistics
python scripts/ingest.py stats

# Clear the knowledge base
python scripts/ingest.py clear --yes
```

Supported file types:
- **Documents**: `.md`, `.txt`, `.pdf`
- **Code**: `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.sh`
- **Config**: `.yaml`, `.yml`, `.json`, `.toml`, `.ini`, `.xml`
- **Web**: `.html`, `.css`

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ROOT` | repo root (CLI defaults to cwd) | Project root for file tools and local data |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral:7b` | Local LLM model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RETRIEVER_K` | `4` | Number of documents to retrieve |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `RELEVANCE_THRESHOLD` | `0.7` | Minimum relevance score |
| `FALLBACK_ENABLED` | `true` | Enable external fallback |
| `BASH_REQUIRE_APPROVAL` | `true` | Require approval for dangerous commands |

### External Provider API Keys (Optional)

For fallback to external providers, add API keys to `.env`:

```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
XAI_API_KEY=your-xai-key
```

Fallback priority: Local Ollama → Claude → GPT-4o → Gemini → Grok

## Project Structure

```
llm/
├── config/
│   ├── settings.py          # Pydantic settings
│   └── providers.yaml       # LiteLLM provider config
├── src/
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine
│   │   ├── nodes.py         # Workflow node implementations
│   │   └── state.py         # Agent state schema
│   ├── llm/
│   │   ├── local.py         # Ollama integration
│   │   └── selector.py      # Local-first provider selection
│   ├── knowledge/
│   │   ├── vectorstore.py   # ChromaDB operations
│   │   ├── retriever.py     # RAG retrieval logic
│   │   ├── grader.py        # Document relevance grading
│   │   └── updater.py       # KB update from external sources
│   ├── ingestion/
│   │   ├── pipeline.py      # Document ingestion orchestrator
│   │   └── loaders.py       # File type loaders
│   ├── tools/
│   │   ├── registry.py      # Tool registration
│   │   └── bash.py          # Sandboxed bash execution
│   └── api/
│       ├── main.py          # FastAPI application
│       └── routes.py        # API endpoints
├── scripts/
│   ├── ingest.py            # CLI for document ingestion
│   └── run.py               # Main entry point
└── data/
    ├── documents/           # Source documents
    └── chroma_db/           # Vector store
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| Local LLM | Ollama (mistral:7b) |
| Vector Database | ChromaDB |
| Multi-Provider | LiteLLM |
| Embeddings | nomic-embed-text |
| API Framework | FastAPI |
| CLI Framework | Typer + Rich |

## Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
```

### Type Checking

```bash
mypy src
```

## License

MIT
