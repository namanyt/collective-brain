# 🧠 Qdrant Collective Brain MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP: 2.0](https://img.shields.io/badge/MCP-2.0-green.svg)](https://modelcontextprotocol.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A lightweight, production-grade **Model Context Protocol (MCP)** server that unites multiple AI agents, IDEs, and assistants (Claude Code, Claude Desktop, Cursor, Antigravity, Open-WebUI) into a **single, shared, persistent consciousness**.

Powered by **Qdrant**, **FastEmbed ONNX embeddings (`BAAI/bge-small-en-v1.5`)**, and **Cross-Encoder reranking (`Xenova/ms-marco-MiniLM-L-6-v2`)**.

---

## 🌟 Philosophy: One Mind, Many Agents

When working across multiple AI clients, context usually fragments:
* Your IDE agent doesn't know what decisions you made in terminal sessions.
* Web chat assistants forget your infrastructure constraints and user preferences.
* Important architectural rules are lost when context windows trim.

**Collective Brain** solves this by providing a unified cognitive storage and retrieval layer:
- **Agents check memory on boot** (`recall_memories` / `unified_search`).
- **Agents record decisions and preferences in real time** (`remember_fact_or_preference` / `save_note`).
- **Agents reference long-form documentation** with automatic section-aware chunking and cross-encoder precision (`index_document` / `search_documents`).

---

## 🚀 Key Highlights & Hardware Guardrails

- **🛡️ Anti-OOM Guaranteed**:
  - Combined memory footprint is only **~340 MB RAM**.
  - Enforced Docker hard limit: `1024 MB` (`mem_limit: 1024m`) and throttled CPU cores.
  - Zero risk of starving the host server or crashing co-hosted services (e.g., Jellyfin, databases).
- **🎮 100% GPU / VRAM Preserved**:
  - Runs exclusively on CPU using high-speed SIMD ONNX Runtime.
  - Zero GPU VRAM allocated, leaving discrete GPUs (e.g., NVIDIA GTX 1650) completely free for **Jellyfin hardware transcoding (NVENC)**.
- **⚡ Dual Protocol Support**:
  - **Modern Streamable HTTP** (MCP spec 2024-11-05): `POST /sse`, `POST /mcp`, `POST /`
  - **Legacy SSE**: `GET /sse` + `POST /messages/`
  - **Native OpenAPI 3.1.0 REST**: `GET /openapi.json` + `POST /api/*` (direct import for Open-WebUI Tools).
- **⏱️ Anti-Hang & Thread Safety**:
  - Single-threaded ONNX execution (`OMP_NUM_THREADS=1`, `ORT_INTRA_OP_NUM_THREADS=1`) prevents threadpool deadlocks.
  - Hard query execution timeouts (`TOOL_TIMEOUT=20.0s`).
  - Minimum relevance thresholding (`VECTOR_SCORE_THRESHOLD=0.64`, `MIN_RELEVANCE_CONFIDENCE=0.05`) eliminates irrelevant search hallucinations.
- **🧩 Pure Cognitive Abstraction**:
  - Clients never see raw Qdrant collections, vector dimensions, or cosine mathematics.
  - The MCP server manages chunking, payload indexing, two-stage retrieval, and lifecycle management automatically.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          AI Clients & Agents                           │
│  Claude Code / Desktop  •  Cursor IDE  •  Open-WebUI  •  Antigravity   │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │ MCP (Streamable HTTP / SSE)    │ REST / OpenAPI
                    ▼ :5006                          ▼ :5006
┌────────────────────────────────────────────────────────────────────────┐
│                   Collective Brain MCP Server                          │
│                                                                        │
│   FastEmbed ONNX (CPU)                   Cross-Encoder Reranker        │
│   BAAI/bge-small-en-v1.5 (384d)          ms-marco-MiniLM-L-6-v2        │
│   ~130 MB RAM                            ~150 MB RAM                   │
│                                                                        │
│   • Semantic Vector Retrieval            • Near-Duplicate Detection    │
│   • Multi-stage Cross Reranking          • Markdown Structure Chunker  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP :6333 (Docker Network)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Qdrant Vector Database                          │
│   Collections: collective_memories  •  collective_documents            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Complete MCP Tools Reference

### 1. Ingestion & Memory Storage
| Tool | Description |
| :--- | :--- |
| `remember_fact_or_preference` | Store an important user preference, rule, constraint, or fact (`category='preference'/'fact'/'rule'`). |
| `save_note` | Save a project note, architectural decision, idea, or session summary. |
| `remember_facts_batch` | Batch ingest multiple facts or preferences at once in a single efficient call. |
| `index_document` | Ingest long markdown documentation, guides, or code files. Automatically chunks, embeds, and tracks chapters. |

### 2. Retrieval & Semantic Search (with Reranking)
| Tool | Description |
| :--- | :--- |
| `recall_memories` | Vector similarity search + Cross-Encoder reranking over memories, notes, and preferences. Supports category & tag filters. |
| `search_documents` | Deep search through long-form documentation chunks with relevance scores and parent document titles. |
| `unified_search` | Search across ALL memories, notes, facts, and documents simultaneously with unified cross-encoder ranking. |

### 3. Exploration & Detail Retrieval
| Tool | Description |
| :--- | :--- |
| `list_recent_memories` | Chronological browse of recently stored items without needing a search query. |
| `list_all_documents` | List all registered documents in the knowledge base with chunk counts and tags. |
| `get_memory_details` | Inspect the complete content, raw metadata, and timestamps of a memory by ID. |
| `get_document_full_content` | Reassemble and return the complete text of an indexed document by ID or title. |

### 4. Updates, Housekeeping & Admin
| Tool | Description |
| :--- | :--- |
| `update_memory` | Edit an existing memory's content, title, category, or tags (re-vectorizes automatically). |
| `forget_memory` | Permanently delete a memory or note by ID. |
| `delete_document` | Delete an entire document and all of its chunks by ID or title. |
| `list_topics_and_tags` | Discover all active categories and tags stored in the brain. |
| `inspect_brain_health` | Diagnostic report showing Qdrant connectivity, point counts, and RAM footprint. |
| `get_brain_guide` | In-chat documentation guide for agents explaining best practices for memory retrieval. |
| `clear_brain` | Safe brain reset tool requiring an explicit safety confirmation phrase. |

---

## 📡 Client Configuration

### 1. Claude Desktop / Claude Code (`mcpServers`)
Add the following to your `claude_desktop_config.json` or `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "collective-brain": {
      "url": "http://localhost:5006/sse"
    }
  }
}
```

### 2. Cursor IDE
In **Cursor Settings ➔ Features ➔ MCP Servers ➔ Add New MCP Server**:
* **Name**: `collective-brain`
* **Type**: `sse` (or `http`)
* **URL**: `http://localhost:5006/sse`

### 3. Open-WebUI
Open-WebUI supports two connection methods:

* **Method A (Native MCP via SSE)**:
  In **Admin Settings ➔ External Tools / MCP**, add an SSE endpoint:
  ```
  http://qdrant-brain-mcp:5006/sse
  ```
  *(assuming Open-WebUI and this container share the Docker bridge network `llm_ai-net`).*

* **Method B (OpenAPI Tool Import)**:
  In **Workspace ➔ Tools ➔ + (Add Tool) ➔ Import from OpenAPI**:
  Point to:
  ```
  http://qdrant-brain-mcp:5006/openapi.json
  ```
  Open-WebUI will generate individual UI tool buttons for all 18 cognitive operations.

### 4. Antigravity CLI (`agy`)
In `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "collective-brain": {
      "serverUrl": "http://localhost:5006/sse",
      "url": "http://localhost:5006/sse"
    }
  }
}
```

---

## 🐳 Deployment

### Prerequisites
* Docker & Docker Compose v2+
* Running Qdrant instance on `localhost:6333` (or on a shared Docker network like `llm_ai-net`)

### Running with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/ciderboi/collective-brain-mcp.git
cd collective-brain-mcp

# Copy environment configuration
cp .env.example .env

# Build and start container in background
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Verify health status
curl http://localhost:5006/health
```

### Running Locally with Python (`uv`)

```bash
# Install dependencies and run directly
uv run server.py
```

---

## ⚙️ Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://qdrant:6333` | URL of the running Qdrant instance. |
| `PORT` | `5006` | HTTP / SSE port for the MCP server. |
| `HOST` | `0.0.0.0` | Bind host address. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed ONNX embedding model (384-dim). |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | FastEmbed ONNX cross-encoder reranker model. |
| `VECTOR_DIM` | `384` | Dimension size of embedding vectors. |
| `VECTOR_SCORE_THRESHOLD` | `0.64` | Minimum cosine similarity threshold before candidate passes to reranker. |
| `MIN_RELEVANCE_CONFIDENCE`| `0.05` | Minimum sigmoid confidence score required to return a result. |
| `MAX_CANDIDATES_FOR_RERANK`| `8` | Maximum candidates scored by the cross-encoder per query. |
| `TOOL_TIMEOUT` | `20.0` | Execution timeout in seconds per tool call to prevent hanging. |
| `ONNX_THREADS` | `1` | Intra-op threads allocated to ONNX runtime (preserves CPU). |

---

## 👥 Authors

* **Eve** — [`<eve@ciderboi.xyz>`](mailto:eve@ciderboi.xyz)
* **Naman** — [`<me@ciderboi.xyz>`](mailto:me@ciderboi.xyz)

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).
