# Local LLM Agent

A local-first LLM agent with RAG-based knowledge base, tool execution, and intelligent fallback to external providers.

## Features

- **Local-First**: Prioritizes local Ollama models, only falling back to external providers when needed
- **RAG Knowledge Base**: ChromaDB-powered vector store with document ingestion and retrieval
- **Intelligent Routing**: LangGraph state machine routes queries to retrieval, tools, or direct generation
- **Document Grading**: LLM-based relevance grading with automatic query rewriting
- **Multi-Provider Fallback**: Automatic fallback chain (Ollama → Claude → GPT-4 → Gemini → Grok)
- **Knowledge Base Updates**: Automatically extracts and stores facts from external provider responses
- **Sandboxed Tool Execution**: Safe bash command execution with validation and approval
- **Runtime Model Switching**: Switch between Ollama models at runtime via `/model` command or `--model` flag
- **Conversation Memory**: Agent remembers previous exchanges within a session
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

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) (for local LLM)

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

Commands in chat:
- `/model` - Show current model
- `/model <name>` - Switch to a different model (e.g., `/model deepseek-r1:7b`)
- `/models` - List available Ollama models
- `/clear` - Clear conversation history
- `/stats` - Show knowledge base and model stats
- `/help` - Show help
- `/quit` - Exit

The CLI shows:
- Current model and version on welcome screen
- Real-time elapsed time while processing
- Processing steps (routing, retrieving, grading, etc.)
- Grounding status: `Grounded (KB)`, `Grounded (Tools)`, `Local`, or `External`

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
