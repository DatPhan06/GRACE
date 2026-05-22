# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GRACE is a conversational movie recommendation system using LLMs with hybrid retrieval and reranking. It supports INSPIRED and REDIAL datasets. The backend runs a multi-agent LangGraph pipeline; the frontend streams real-time agent progress to the user.

## Development Commands

### Backend (Python 3.12, `uv`)

```bash
cd backend

uv sync                                           # install dependencies
uvicorn api:app --host 0.0.0.0 --port 8000 --reload   # dev server
uv run pytest                                     # run all tests
uv run pytest path/to/test_file.py               # single test file
uv run ruff check .                               # lint
uv run ruff format .                              # format
```

### Frontend (Node.js + Vite)

```bash
cd frontend

npm install
npm run dev      # dev server at http://localhost:5173
npm run build
npm run lint
```

### Docker Compose (full stack)

```bash
docker-compose up -d       # backend + Neo4j + PostgreSQL
docker-compose logs -f retrieval_service
docker-compose down -v     # stop and wipe volumes
```

### Dataset / evaluation scripts

```bash
cd backend

uv run python scripts/run_argos_inspired_eval.py --sample-size 10
uv run python scripts/run_argos_redial_eval.py --sample-size 10
uv run python scripts/generate_embedding.py
uv run python scripts/graph_builder_redial.py
uv run python scripts/graph_builder_inspired.py
```

## Architecture

### Backend — 4-layer (Infra → Domain → App → API)

| Layer | Path | Role |
|---|---|---|
| **Infra** | `backend/infra/` | External connections: PostgreSQL (SQLAlchemy), Neo4j driver, LLM provider clients, embedding clients |
| **Domain** | `backend/domain/` | Business logic by feature: `agent/`, `retrieval/`, `reranking/`, `generation/`, `evaluation/`, `evaluation_storage/` |
| **App** | `backend/app/` | Use-case orchestration with LangGraph: `chat/` and `evaluation/` workflows |
| **API** | `backend/api/` | FastAPI routers: `/chat/`, `/evaluate/`, `/tracing/` |

### LangGraph Chat Pipeline (`app/chat/`)

State is `ARGOSState` (`app/chat/state.py`). Nodes run sequentially:

```
profiler_node → orchestrator_node → retrieval_node → critic_node
    ↑                                                      ↓
    └──────────── relaxation_node (if critic fails) ←─────┘
                                        ↓
                              reranker_node → generator_node
```

- `requires_relaxation` flag and `attempt` counter on state control the retry loop.
- `critic_node` gates whether candidates satisfy hard constraints; on failure it routes to `relaxation_node` which relaxes constraints then loops back to `retrieval_node`.

### Retrieval — Hybrid WRRF Fusion

Three parallel retrievers run via `asyncio.gather()`:
- **SemanticRetriever** — vector similarity (Chroma)
- **ContentRetriever** — genre/year filtering + heuristic scoring
- **GraphReasoningAgent** — Neo4j graph traversal (collaborative filtering via LLM agent)

Results are fused with Weighted Reciprocal Rank Fusion (WRRF); weights `(w_sem, w_con, w_col)` are set dynamically by `orchestrator_node` based on the profiler output.

### Reranking Factory (`domain/reranking/factory.py`)

Three implementations: `LLMReranker`, `CohereReranker`, `DecoupledReranker`. Selected via config/environment.

### Frontend — React + Vite + TypeScript

Three routes:
- `/` — chat interface with streaming agent progress (`components/ChatInterface.tsx`)
- `/eval` — full evaluation runner
- `/step-eval` — step-based evaluation (profiler → retrieval → reranking individually)

All backend calls are in `lib/api.ts` (Axios + NDJSON streaming for `/chat/stream`).

## Configuration

### Required environment variables (`.env`)

```
# Neo4j
NEO4J_URI=bolt://localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=grace
POSTGRES_SERVER=localhost   # or "postgres" inside Docker

# LLM provider (pick one)
LLM_PROVIDER=azure          # azure | openai | gemini
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_LLM_MODEL=...
AZURE_EMBEDDING_MODEL=...
# or OPENAI_API_KEY / GEMINI_API_KEY

# Cohere (reranking only, optional)
AWS_LLM_ACCESS_KEY_ID=...
AWS_LLM_SECRET_ACCESS_KEY=...
AWS_LLM_REGION=...
```

`config.yaml` (root) holds model names and dataset file paths.

## Database Models (PostgreSQL)

Defined in `infra/db/models.py`:
- `EvaluationRun` — run metadata (dataset, sample_size, avg_recall)
- `BatchStepExecution` — per-step execution config, status, results
- `ConversationLog` — conversation metadata and final results

## Key Conventions

- All services are `async`; use `asyncio.gather()` for parallelism.
- Structured LLM output via Pydantic models (ProfilerAgent → `UserPreference`, CriticAgent → dict, etc.).
- Logging: `from shared.utils.logger import setup_logger; logger = setup_logger(__name__)`.
- Frontend API base URL is hardcoded to `http://localhost:8000` — change in `lib/api.ts` for production.
