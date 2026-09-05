"""Qdrant Collective Brain MCP Server (SSE-only on port 5006).

Provides high-level cognitive abstractions over Qdrant, FastEmbed embeddings (bge-small-en-v1.5),
and cross-encoder reranking (ms-marco-MiniLM-L-6-v2). All backend collection management,
vector math, payload indexing, and chunking are fully encapsulated.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp, StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
import uvicorn

from brain import BrainManager
from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("qdrant-mcp")

# Initialize the Collective Brain
brain = BrainManager()


async def _brain_call(fn, *args, **kwargs):
    """Run blocking ONNX / Qdrant work off the event loop with a hard timeout."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=config.TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Brain operation timed out after {config.TOOL_TIMEOUT}s") from e

# Initialize MCPServer
server = MCPServer(
    name="collective-brain-mcp",
    version="1.0.0",
    instructions=(
        "You are connected to the Collective Brain MCP server. This server provides shared, "
        "persistent memory, conceptual notes, and document knowledge across multiple AI clients "
        "and agents. Always check for relevant context using recall_memories or search_documents "
        "when starting a task, and remember new user preferences, rules, decisions, or facts "
        "to keep the collective intelligence aligned."
    ),
)


# -----------------------------------------------------------------------------
# Custom Starlette HTTP Routes (Health & Status)
# -----------------------------------------------------------------------------


@server.custom_route("/health", methods=["GET"])
async def health_endpoint(request: Request) -> JSONResponse:
    """HTTP health check endpoint for Docker and monitoring."""
    try:
        health_data = brain.inspect_health()
        return JSONResponse({"status": "healthy", **health_data})
    except Exception as e:
        logger.error("Health check error: %s", e)
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=500)


@server.custom_route("/openapi.json", methods=["GET"])
async def openapi_endpoint(request: Request) -> JSONResponse:
    """Serve the OpenAPI 3.1.0 specification for REST clients and Open-WebUI."""
    try:
        with open("openapi.json", "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    except Exception as e:
        logger.error("Failed to read openapi.json: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@server.custom_route("/api/remember_fact_or_preference", methods=["POST"])
async def rest_remember(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.remember(
            content=data["fact"],
            category=data.get("category", "preference"),
            tags=data.get("tags", []),
            source=data.get("source", "rest-client"),
        )
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/save_note", methods=["POST"])
async def rest_save_note(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.remember(
            content=data["content"],
            category="note",
            tags=data.get("tags", []),
            title=data.get("title"),
            source=data.get("source", "rest-client"),
        )
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/remember_facts_batch", methods=["POST"])
async def rest_remember_batch(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        items = [
            {
                "content": f,
                "category": data.get("category", "fact"),
                "tags": data.get("tags", []),
                "source": data.get("source", "rest-client"),
            }
            for f in data.get("facts", [])
        ]
        res = brain.remember_batch(items)
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/index_document", methods=["POST"])
async def rest_index_doc(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.index_document(
            title=data["title"],
            content=data["content"],
            category=data.get("category", "doc"),
            tags=data.get("tags", []),
            source=data.get("source", "rest-client"),
            chunk_size=data.get("chunk_size"),
            chunk_overlap=data.get("chunk_overlap"),
        )
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/recall_memories", methods=["POST"])
async def rest_recall(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = await _brain_call(
            brain.recall_memories,
            data["query"],
            int(data.get("limit", 5)),
            data.get("category") or None,
            data.get("tag") or None,
            bool(data.get("use_reranker", True)),
        )
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/search_documents", methods=["POST"])
async def rest_search_docs(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = await _brain_call(
            brain.search_documents,
            data["query"],
            int(data.get("limit", 5)),
            data.get("tag") or None,
            bool(data.get("use_reranker", True)),
        )
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/unified_search", methods=["POST"])
async def rest_unified(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = await _brain_call(
            brain.unified_search,
            data["query"],
            int(data.get("limit", 5)),
        )
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/list_recent_memories", methods=["GET", "POST"])
async def rest_list_recent(request: Request) -> JSONResponse:
    try:
        limit = 10
        category = None
        tag = None
        if request.method == "POST":
            try:
                data = await request.json()
                limit = int(data.get("limit", 10))
                category = data.get("category") or None
                tag = data.get("tag") or None
            except Exception:
                pass
        else:
            limit = int(request.query_params.get("limit", 10))
            category = request.query_params.get("category") or None
            tag = request.query_params.get("tag") or None

        res = brain.list_recent_memories(limit=limit, category=category, tag=tag)
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/list_all_documents", methods=["GET", "POST"])
async def rest_list_docs(request: Request) -> JSONResponse:
    try:
        limit = 20
        tag = None
        if request.method == "POST":
            try:
                data = await request.json()
                limit = int(data.get("limit", 20))
                tag = data.get("tag") or None
            except Exception:
                pass
        else:
            limit = int(request.query_params.get("limit", 20))
            tag = request.query_params.get("tag") or None

        res = brain.list_documents(limit=limit, tag=tag)
        return JSONResponse({"status": "success", "results": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/get_memory_details", methods=["POST"])
async def rest_get_memory(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.get_memory(data["memory_id"])
        if not res:
            return JSONResponse({"status": "not_found"}, status_code=404)
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/get_document_full_content", methods=["POST"])
async def rest_get_doc_full(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.get_document_full(data["document_id_or_title"])
        if not res:
            return JSONResponse({"status": "not_found"}, status_code=404)
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/update_memory", methods=["POST"])
async def rest_update_memory(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        res = brain.update_memory(
            memory_id=data["memory_id"],
            content=data.get("content") or None,
            title=data.get("title") or None,
            category=data.get("category") or None,
            tags=data.get("tags") or None,
        )
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/forget_memory", methods=["POST"])
async def rest_forget_memory(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        deleted = brain.forget_memory(data["memory_id"])
        if not deleted:
            return JSONResponse({"status": "not_found"}, status_code=404)
        return JSONResponse({"status": "success", "deleted_id": data["memory_id"]})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/delete_document", methods=["POST"])
async def rest_delete_doc(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        brain.delete_document(data["document_id_or_title"])
        return JSONResponse({"status": "success", "deleted": data["document_id_or_title"]})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/list_topics_and_tags", methods=["GET"])
async def rest_topics(request: Request) -> JSONResponse:
    try:
        res = brain.list_topics_and_tags()
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/inspect_brain_health", methods=["GET"])
async def rest_health_info(request: Request) -> JSONResponse:
    try:
        res = brain.inspect_health()
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


@server.custom_route("/api/get_brain_guide", methods=["GET"])
async def rest_guide(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "success",
        "guide": (
            "Collective Brain provides shared cognitive memory and document knowledge. "
            "Use recall_memories or unified_search for context retrieval, and "
            "remember_fact_or_preference for persisting knowledge."
        )
    })


@server.custom_route("/api/clear_brain", methods=["POST"])
async def rest_clear_brain(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        confirm = data.get("confirm", "")
        expected = "I understand this permanently deletes agent brain data"
        if confirm != expected:
            return JSONResponse(
                {"status": "error", "error": f"Confirmation phrase mismatch. Must be: '{expected}'"},
                status_code=400,
            )
        res = brain.clear_brain(target=data.get("target", "memories"))
        return JSONResponse({"status": "success", "result": res})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)


# -----------------------------------------------------------------------------
# MCP Tools: Memory & Knowledge Ingestion
# -----------------------------------------------------------------------------


@server.tool()
def remember_fact_or_preference(
    fact: str,
    category: str = "preference",
    tags: list[str] = [],
    source: str = "agent",
) -> str:
    """Save an important fact, user preference, instruction, rule, or constraint to the collective brain.

    Use this whenever:
    - The user specifies a personal preference (e.g. 'I prefer concise python answers').
    - You learn an important environmental fact (e.g. 'Server has 4GB VRAM GTX 1650, Jellyfin runs on port 8096').
    - An architectural rule or constraint is established that other agents must follow.

    Args:
        fact: The specific statement, preference, rule, or fact to remember.
        category: Category of the memory ('preference', 'fact', 'rule', 'decision'). Defaults to 'preference'.
        tags: List of keywords/tags for topical filtering (e.g. ['server', 'jellyfin', 'python']).
        source: Name or ID of the client/agent storing this memory (e.g. 'antigravity', 'open-webui').

    Returns:
        Confirmation markdown with memory ID, category, and timestamp.
    """
    try:
        res = brain.remember(content=fact, category=category, tags=tags, source=source)
        action = "Updated existing memory" if res.get("updated") else "Memory stored"
        return (
            f"### ✅ {action} successfully\n\n"
            f"- **ID**: `{res['id']}`\n"
            f"- **Category**: `{res['category']}`\n"
            f"- **Tags**: {', '.join(f'`{t}`' for t in res['tags']) if res['tags'] else '_None_'}\n"
            f"- **Source**: `{res['source']}`\n"
            f"- **Recorded At**: `{res.get('created_at')}`\n\n"
            f"> \"{res['text']}\""
        )
    except Exception as e:
        logger.error("remember_fact_or_preference error: %s", e)
        return f"❌ Error storing memory: {e}"


@server.tool()
def save_note(
    title: str,
    content: str,
    tags: list[str] = [],
    source: str = "agent",
) -> str:
    """Save a conceptual note, idea, project discussion, or architectural decision to the collective brain.

    Use this for:
    - Session summaries or project status logs.
    - Architecture decisions and reasoning behind technical choices.
    - Scratchpad ideas or todo lists to be shared across sessions and agents.

    Args:
        title: A descriptive title for the note.
        content: The text/markdown body of the note.
        tags: Topic tags (e.g. ['architecture', 'roadmap', 'meeting']).
        source: Client or agent name storing this note.

    Returns:
        Confirmation markdown with the note ID and metadata.
    """
    try:
        res = brain.remember(content=content, category="note", tags=tags, title=title, source=source)
        action = "Note updated" if res.get("updated") else "Note saved"
        return (
            f"### 📝 {action} successfully\n\n"
            f"- **ID**: `{res['id']}`\n"
            f"- **Title**: **{res['title']}**\n"
            f"- **Tags**: {', '.join(f'`{t}`' for t in res['tags']) if res['tags'] else '_None_'}\n"
            f"- **Created At**: `{res.get('created_at')}`\n\n"
            f"**Preview:**\n{content[:200]}..."
        )
    except Exception as e:
        logger.error("save_note error: %s", e)
        return f"❌ Error saving note: {e}"


@server.tool()
def remember_facts_batch(
    facts: list[str],
    category: str = "fact",
    tags: list[str] = [],
    source: str = "agent",
) -> str:
    """Store multiple facts or preferences in a single efficient vector batch.

    Ideal when summarizing a conversation or onboarding knowledge where multiple distinct
    facts were learned simultaneously.

    Args:
        facts: List of fact strings to store.
        category: Common category for these items ('fact', 'preference', 'rule').
        tags: Common tags to attach to all items.
        source: Client/agent identifier.

    Returns:
        Summary of stored items with their assigned IDs.
    """
    try:
        items = [{"content": f, "category": category, "tags": tags, "source": source} for f in facts if f.strip()]
        results = brain.remember_batch(items)
        lines = [f"### 📦 Batch Memories Stored ({len(results)} items)\n"]
        for r in results:
            lines.append(f"- `{r['id']}`: {r['text'][:80]}...")
        return "\n".join(lines)
    except Exception as e:
        logger.error("remember_facts_batch error: %s", e)
        return f"❌ Error in batch memory store: {e}"


@server.tool()
def index_document(
    title: str,
    content: str,
    category: str = "doc",
    tags: list[str] = [],
    source: str = "agent",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> str:
    """Ingest a long-form document, guide, markdown file, or code manual into the collective brain.

    The MCP server automatically splits the document into overlapping chunks, generates
    FastEmbed vectors, and stores them with document-level traceability.

    Args:
        title: Title or filename of the document (e.g. 'Jellyfin GPU Setup Guide').
        content: The full markdown or text content of the document.
        category: Document category ('doc', 'guide', 'spec', 'code', 'manual').
        tags: Categorical tags (e.g. ['jellyfin', 'transcoding', 'docker']).
        source: Ingestion source or author.
        chunk_size: Maximum character count per chunk (default: 800).
        chunk_overlap: Character overlap between consecutive chunks (default: 150).

    Returns:
        Document ID, chunk count, and ingestion confirmation.
    """
    try:
        res = brain.index_document(
            title=title,
            content=content,
            category=category,
            tags=tags,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return (
            f"### 📚 Document Indexed Successfully\n\n"
            f"- **Document ID**: `{res['document_id']}`\n"
            f"- **Title**: **{res['document_title']}**\n"
            f"- **Total Chunks**: `{res['total_chunks']}`\n"
            f"- **Category**: `{res['category']}`\n"
            f"- **Tags**: {', '.join(f'`{t}`' for t in res['tags']) if res['tags'] else '_None_'}\n"
            f"- **Indexed At**: `{res['created_at']}`\n\n"
            f"**First Chunk Preview:**\n> {res['preview']}"
        )
    except Exception as e:
        logger.error("index_document error: %s", e)
        return f"❌ Error indexing document: {e}"


# -----------------------------------------------------------------------------
# MCP Tools: Retrieval & Search (Semantic + Reranked)
# -----------------------------------------------------------------------------


@server.tool()
async def recall_memories(
    query: str,
    limit: int = 5,
    category: str = "",
    tag: str = "",
    use_reranker: bool = True,
) -> str:
    """Search through stored facts, preferences, notes, and decisions using semantic search and cross-encoder reranking.

    Use this tool whenever you need to:
    - Check user preferences or instructions (e.g. 'What are user preferences for coding?').
    - Check system/hardware specs or credentials (e.g. 'What is the Jellyfin port?').
    - Retrieve previous decisions or discussions on a project.

    Args:
        query: Natural language search query or question.
        limit: Number of top results to return (default: 5, max: 20).
        category: Optional category filter ('preference', 'fact', 'note', 'decision'). Empty string for no filter.
        tag: Optional single tag filter. Empty string for no filter.
        use_reranker: If True, uses the ms-marco cross-encoder reranker for maximum precision. Defaults to True.

    Returns:
        Ranked list of memories with relevance confidence scores, tags, and timestamps.
    """
    try:
        cat_filter = category.strip() if category.strip() else None
        tag_filter = tag.strip() if tag.strip() else None

        results = await _brain_call(
            brain.recall_memories,
            query,
            min(limit, 20),
            cat_filter,
            tag_filter,
            use_reranker,
        )

        if not results:
            return (
                f"🔍 No relevant memories found for: \"{query}\". "
                "This information is not in the collective brain. Do not retry the same query."
            )

        lines = [f"### 🧠 Recalled Memories for: \"{query}\"\n"]
        for idx, r in enumerate(results, 1):
            conf = r.get("relevance_confidence", 0.0)
            stars = "⭐⭐⭐" if conf > 0.85 else ("⭐⭐" if conf > 0.65 else "⭐")
            lines.append(
                f"#### {idx}. {r.get('title', 'Memory')} ({stars} Relevance: `{conf:.1%}`)\n"
                f"- **ID**: `{r['id']}` | **Category**: `{r.get('category', 'fact')}` | **Source**: `{r.get('source', 'agent')}`\n"
                f"- **Tags**: {', '.join(f'`{t}`' for t in r.get('tags', [])) or '_None_'}\n"
                f"- **Content**:\n> {r['text']}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("recall_memories error: %s", e)
        return f"❌ Error recalling memories: {e}"


@server.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    tag: str = "",
    use_reranker: bool = True,
) -> str:
    """Search through indexed long-form documentation, manuals, and guides.

    Uses FastEmbed vector search + cross-encoder reranking to locate the exact relevant paragraphs.

    Args:
        query: Natural language question or topic to search for.
        limit: Number of document chunks to return (default: 5).
        tag: Optional tag filter. Empty string for no filter.
        use_reranker: Whether to cross-encoder rerank candidate chunks. Defaults to True.

    Returns:
        Ranked document sections with parent document titles and relevance scores.
    """
    try:
        tag_filter = tag.strip() if tag.strip() else None
        results = await _brain_call(
            brain.search_documents,
            query,
            min(limit, 20),
            tag_filter,
            use_reranker,
        )

        if not results:
            return (
                f"🔍 No relevant document sections found for: \"{query}\". "
                "This information is not in the collective brain. Do not retry the same query."
            )

        lines = [f"### 📖 Document Search Results for: \"{query}\"\n"]
        for idx, r in enumerate(results, 1):
            conf = r.get("relevance_confidence", 0.0)
            chunk_info = f"Chunk {r.get('chunk_index', 0) + 1}/{r.get('total_chunks', 1)}"
            lines.append(
                f"#### {idx}. {r.get('document_title', 'Untitled')} ({chunk_info}) — Relevance: `{conf:.1%}`\n"
                f"- **Doc ID**: `{r.get('document_id')}` | **Tags**: {', '.join(f'`{t}`' for t in r.get('tags', [])) or '_None_'}\n"
                f"```markdown\n{r.get('text', '')}\n```\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("search_documents error: %s", e)
        return f"❌ Error searching documents: {e}"


@server.tool()
async def unified_search(
    query: str,
    limit: int = 5,
) -> str:
    """Search across ALL memories, notes, facts, and documents simultaneously.

    Gathers candidate results from both memory and documentation collections, cross-encoder
    reranks all items together, and outputs the top items regardless of their underlying type.

    Args:
        query: Search query or question.
        limit: Total number of combined results to return (default: 5).

    Returns:
        Ranked list combining memories, notes, and document sections.
    """
    try:
        results = await _brain_call(brain.unified_search, query, min(limit, 20))
        if not results:
            return (
                f"🔍 No relevant memories or documents found for: \"{query}\". "
                "This information is not in the collective brain. Do not retry the same query."
            )

        lines = [f"### 🌐 Unified Brain Search for: \"{query}\"\n"]
        for idx, r in enumerate(results, 1):
            stype = r.get("source_type", "item").upper()
            title = r.get("document_title") if stype == "DOCUMENT" else r.get("title", "Memory")
            conf = r.get("relevance_confidence", 0.0)
            lines.append(
                f"#### {idx}. [{stype}] {title} (Confidence: `{conf:.1%}`)\n"
                f"- **ID**: `{r.get('id') or r.get('document_id')}` | **Category**: `{r.get('category', 'general')}`\n"
                f"- **Text Snippet**:\n> {r.get('text', '')[:300]}...\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("unified_search error: %s", e)
        return f"❌ Error in unified search: {e}"


# -----------------------------------------------------------------------------
# MCP Tools: Exploration & Detail Retrieval
# -----------------------------------------------------------------------------


@server.tool()
def list_recent_memories(
    limit: int = 10,
    category: str = "",
    tag: str = "",
) -> str:
    """Browse recently saved memories, notes, and facts chronologically without needing a search query.

    Args:
        limit: Number of recent items to retrieve (default: 10, max: 50).
        category: Optional category filter ('preference', 'fact', 'note', 'decision').
        tag: Optional tag filter.

    Returns:
        Markdown list of recent memories with IDs, tags, and text previews.
    """
    try:
        cat_filter = category.strip() if category.strip() else None
        tag_filter = tag.strip() if tag.strip() else None
        items = brain.list_recent_memories(limit=limit, category=cat_filter, tag=tag_filter)

        if not items:
            return "📭 No memories found."

        lines = [f"### 🕒 Recent Memories ({len(items)} items)\n"]
        for it in items:
            tags_str = ", ".join(f"`{t}`" for t in it.get("tags", [])) or "None"
            lines.append(
                f"- **{it.get('title', 'Memory')}** (`{it.get('category', 'fact')}`)\n"
                f"  - ID: `{it.get('id')}` | Tags: {tags_str} | Date: `{it.get('created_at', '')[:10]}`\n"
                f"  - Content: {it.get('text', '')[:120]}...\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("list_recent_memories error: %s", e)
        return f"❌ Error listing memories: {e}"


@server.tool()
def list_all_documents(
    limit: int = 20,
    tag: str = "",
) -> str:
    """List all indexed documents in the brain with titles, chunk counts, and metadata.

    Args:
        limit: Maximum number of documents to list (default: 20).
        tag: Optional tag filter.

    Returns:
        Markdown table of all registered documents in the knowledge base.
    """
    try:
        tag_filter = tag.strip() if tag.strip() else None
        docs = brain.list_documents(limit=limit, tag=tag_filter)
        if not docs:
            return "📭 No indexed documents found."

        lines = [
            f"### 📚 Indexed Documents ({len(docs)} found)\n",
            "| Document ID | Title | Chunks | Category | Tags | Created |",
            "| :--- | :--- | :---: | :--- | :--- | :--- |",
        ]
        for d in docs:
            tags_str = ", ".join(d.get("tags", [])) or "-"
            created = str(d.get("created_at", ""))[:10]
            lines.append(
                f"| `{d['document_id']}` | **{d['document_title']}** | {d['total_chunks']} | `{d['category']}` | {tags_str} | {created} |"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("list_all_documents error: %s", e)
        return f"❌ Error listing documents: {e}"


@server.tool()
def get_memory_details(memory_id: str) -> str:
    """Retrieve full content, raw metadata, and timestamps for a specific memory by ID.

    Args:
        memory_id: Unique ID of the memory.

    Returns:
        Complete memory payload in formatted markdown.
    """
    try:
        mem = brain.get_memory(memory_id.strip())
        if not mem:
            return f"❌ Memory ID `{memory_id}` not found."

        return (
            f"### 🔍 Memory Details: `{memory_id}`\n\n"
            f"- **Title**: **{mem.get('title', 'Memory')}**\n"
            f"- **Category**: `{mem.get('category', 'fact')}`\n"
            f"- **Tags**: {', '.join(f'`{t}`' for t in mem.get('tags', [])) or '_None_'}\n"
            f"- **Source**: `{mem.get('source', 'agent')}`\n"
            f"- **Created At**: `{mem.get('created_at')}`\n"
            f"- **Updated At**: `{mem.get('updated_at')}`\n\n"
            f"**Full Content:**\n```\n{mem.get('text', '')}\n```"
        )
    except Exception as e:
        logger.error("get_memory_details error: %s", e)
        return f"❌ Error retrieving memory: {e}"


@server.tool()
def get_document_full_content(document_id_or_title: str) -> str:
    """Reconstruct and return the full text of an indexed document from all its chunks.

    Args:
        document_id_or_title: The document ID (e.g. 'doc_123abc') or exact document title.

    Returns:
        Reassembled full document text.
    """
    try:
        doc = brain.get_document_full(document_id_or_title)
        if not doc:
            return f"❌ Document `{document_id_or_title}` not found."

        return (
            f"# {doc.get('document_title')}\n\n"
            f"> **ID**: `{doc.get('document_id')}` | **Chunks**: {doc.get('total_chunks')} | "
            f"**Tags**: {', '.join(doc.get('tags', [])) or 'None'} | **Date**: `{doc.get('created_at')}`\n\n"
            f"---\n\n"
            f"{doc.get('content')}"
        )
    except Exception as e:
        logger.error("get_document_full_content error: %s", e)
        return f"❌ Error retrieving full document: {e}"


# -----------------------------------------------------------------------------
# MCP Tools: Updating & Deletion
# -----------------------------------------------------------------------------


@server.tool()
def update_memory(
    memory_id: str,
    content: str = "",
    title: str = "",
    category: str = "",
    tags: list[str] = [],
) -> str:
    """Update the content or metadata of an existing memory.

    If new content is supplied, the vector embedding is automatically recomputed.

    Args:
        memory_id: ID of the memory to update.
        content: New memory content (leave empty to keep unchanged).
        title: New title (leave empty to keep unchanged).
        category: New category (leave empty to keep unchanged).
        tags: New tags list (leave empty to keep unchanged).

    Returns:
        Updated memory summary.
    """
    try:
        res = brain.update_memory(
            memory_id=memory_id,
            content=content if content.strip() else None,
            title=title if title.strip() else None,
            category=category if category.strip() else None,
            tags=tags if tags else None,
        )
        return (
            f"### ✏️ Memory Updated Successfully\n\n"
            f"- **ID**: `{res['id']}`\n"
            f"- **Title**: **{res.get('title')}**\n"
            f"- **Category**: `{res.get('category')}`\n"
            f"- **Tags**: {', '.join(f'`{t}`' for t in res.get('tags', [])) or '_None_'}\n"
            f"- **Updated At**: `{res.get('updated_at')}`\n"
        )
    except Exception as e:
        logger.error("update_memory error: %s", e)
        return f"❌ Error updating memory: {e}"


@server.tool()
def forget_memory(memory_id: str) -> str:
    """Permanently delete a specific memory, preference, or note from the brain by ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        Confirmation message.
    """
    try:
        deleted = brain.forget_memory(memory_id)
        if not deleted:
            return f"❌ Memory ID `{memory_id}` not found."
        return f"🗑️ Memory `{memory_id}` deleted successfully from the brain."
    except Exception as e:
        logger.error("forget_memory error: %s", e)
        return f"❌ Error deleting memory: {e}"


@server.tool()
def delete_document(document_id_or_title: str) -> str:
    """Delete an entire indexed document and all of its constituent chunks from the brain.

    Args:
        document_id_or_title: Document ID or exact title to delete.

    Returns:
        Confirmation message.
    """
    try:
        brain.delete_document(document_id_or_title)
        return f"🗑️ Document `{document_id_or_title}` and all its chunks were deleted successfully."
    except Exception as e:
        logger.error("delete_document error: %s", e)
        return f"❌ Error deleting document: {e}"


# -----------------------------------------------------------------------------
# MCP Tools: Brain Insights & Administration
# -----------------------------------------------------------------------------


@server.tool()
def list_topics_and_tags() -> str:
    """Discover all tags, categories, and topics currently stored in the brain.

    Call this tool when you want to explore what knowledge domains or notes currently
    exist in the brain.

    Returns:
        List of all active categories and tags.
    """
    try:
        info = brain.list_topics_and_tags()
        cats = ", ".join(f"`{c}`" for c in info["categories"]) or "_None_"
        tags = ", ".join(f"`{t}`" for t in info["tags"]) or "_None_"
        return (
            f"### 🏷️ Brain Topics & Categories\n\n"
            f"- **Categories ({info['total_categories']})**: {cats}\n"
            f"- **Tags ({info['total_tags']})**: {tags}\n"
        )
    except Exception as e:
        logger.error("list_topics_and_tags error: %s", e)
        return f"❌ Error listing topics: {e}"


@server.tool()
def inspect_brain_health() -> str:
    """Inspect the status, health, collection counts, and memory footprint of the brain server.

    Returns:
        System diagnostic report verifying Qdrant connection and RAM usage.
    """
    try:
        h = brain.inspect_health()
        return (
            f"### 🩺 Collective Brain Health Report\n\n"
            f"- **Status**: `HEALTHY` ✅\n"
            f"- **Qdrant URL**: `{h['qdrant_url']}`\n"
            f"- **Embedding Model**: `{h['embedding_model']}` ({h['vector_dimension']} dims, ONNX CPU)\n"
            f"- **Reranker Model**: `{h['reranker_model']}` (Cross-Encoder ONNX CPU)\n"
            f"- **Stored Memories**: `{h['total_memories']}`\n"
            f"- **Document Chunks**: `{h['total_document_chunks']}`\n"
            f"- **Process RAM Usage**: `{h['process_memory_mb']} MB`\n"
            f"- **Server Available RAM**: `{h['system_available_ram_gb']} GB` (Total: `{h['system_total_ram_gb']} GB`)\n"
            f"- **VRAM Usage**: `0 MB` (100% of GPU reserved for Jellyfin / Host)\n"
        )
    except Exception as e:
        logger.error("inspect_brain_health error: %s", e)
        return f"❌ Error inspecting health: {e}"


@server.tool()
def get_brain_guide() -> str:
    """Comprehensive guide explaining how agents should interact with this shared memory system.

    Returns:
        Full usage guide, best practices for remembering and recalling, and query tips.
    """
    return (
        "### 🧠 Collective Brain MCP Usage Guide\n\n"
        "This MCP server connects multiple AI agents and clients into a single unified intelligence.\n\n"
        "#### Best Practices for Agents:\n"
        "1. **Start of Task**: Run `recall_memories(query='...')` or `unified_search(...)` to check if past "
        "rules, user preferences, server details, or decisions exist.\n"
        "2. **Learning User Preferences**: When the user says 'Always use X' or 'I prefer Y', call "
        "`remember_fact_or_preference(fact='...', category='preference', tags=['...'])` immediately.\n"
        "3. **Document Ingestion**: For long READMEs, architecture files, or guides, use `index_document(...)`. "
        "Do NOT paste huge documents into memories.\n"
        "4. **Reranking**: The cross-encoder reranker runs automatically (`use_reranker=True`), giving high precision "
        "scores for relevant chunks.\n"
        "5. **Housekeeping**: Use `forget_memory` or `delete_document` if information becomes obsolete or incorrect.\n"
    )


@server.tool()
def clear_brain(
    target: str,
    confirm: str,
) -> str:
    """Clear memories or documents. Requires strict confirmation phrase to prevent accidental deletion.

    Args:
        target: Target collection to reset ('memories', 'documents', or 'all').
        confirm: Must be set to 'I understand this permanently deletes agent brain data'.

    Returns:
        Result of the reset operation.
    """
    expected = "I understand this permanently deletes agent brain data"
    if confirm.strip() != expected:
        return f"❌ Safety check failed! You must set confirm exactly to: '{expected}'"

    try:
        res = brain.clear_brain(target=target)
        return f"⚠️ Brain reset complete for target: `{res['cleared']}`."
    except Exception as e:
        logger.error("clear_brain error: %s", e)
        return f"❌ Error clearing brain: {e}"


# -----------------------------------------------------------------------------
# MCP Transports Setup (Hybrid: Streamable HTTP + SSE)
# -----------------------------------------------------------------------------

# JSON responses (not hanging SSE) so Claude Code's HTTP transport can finish initialize.
session_manager = StreamableHTTPSessionManager(
    app=server._lowlevel_server,
    json_response=True,
)
streamable_asgi = StreamableHTTPASGIApp(session_manager)
sse = SseServerTransport("/messages/")


async def handle_sse(scope: Any, receive: Any, send: Any) -> Response:
    async with sse.connect_sse(scope, receive, send) as streams:
        await server._lowlevel_server.run(
            streams[0], streams[1], server._lowlevel_server.create_initialization_options()
        )
    return Response()


async def sse_endpoint(request: Request) -> Response:
    return await handle_sse(request.scope, request.receive, request._send)


class MCPCompatASGI:
    """Normalize tools/call body quirks without breaking streaming MCP responses.

    Only rewrites POST tools/call. Must not drain ``http.disconnect`` or fake it,
    or Streamable HTTP initialize hangs ("stuck in init") / 500s.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if path not in ("/mcp", "/") and not path.startswith("/messages"):
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunks.append(message.get("body") or b"")
            if not message.get("more_body"):
                break

        body = b"".join(chunks)
        rewritten = _normalize_mcp_jsonrpc(body)
        if rewritten != body:
            logger.info("Normalized MCP JSON-RPC body on %s (%d -> %d bytes)", path, len(body), len(rewritten))

        body_emitted = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal body_emitted
            if not body_emitted:
                body_emitted = True
                return {"type": "http.request", "body": rewritten, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _normalize_mcp_jsonrpc(body: bytes) -> bytes:
    if not body:
        return body
    try:
        payload = json.loads(body)
    except Exception:
        return body

    messages = payload if isinstance(payload, list) else [payload]
    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        method = msg.get("method")
        params = msg.get("params")
        if method != "tools/call" or not isinstance(params, dict):
            continue
        if "arguments" not in params and "input" in params:
            params["arguments"] = params.pop("input")
            changed = True
        args = params.get("arguments")
        if isinstance(args, str):
            try:
                params["arguments"] = json.loads(args)
                changed = True
            except Exception:
                params["arguments"] = {}
                changed = True
        elif args is None:
            params["arguments"] = {}
            changed = True
        logger.info(
            "MCP tools/call name=%s arg_keys=%s",
            params.get("name"),
            sorted((params.get("arguments") or {}).keys()) if isinstance(params.get("arguments"), dict) else type(args),
        )
        msg["params"] = params

    if not changed:
        return body
    return json.dumps(payload if isinstance(payload, list) else messages[0]).encode()


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with session_manager.run():
        yield


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Collective Brain MCP Server (Hybrid SSE + Streamable HTTP) on %s:%d", config.HOST, config.PORT)
    logger.info("MCP Streamable HTTP / SSE endpoint: http://%s:%d/sse", config.HOST, config.PORT)
    logger.info("MCP Alternative endpoint: http://%s:%d/mcp", config.HOST, config.PORT)
    logger.info("Health check endpoint: http://%s:%d/health", config.HOST, config.PORT)
    logger.info("OpenAPI specification: http://%s:%d/openapi.json", config.HOST, config.PORT)

    app = MCPCompatASGI(Starlette(
        debug=False,
        routes=[
            # Health & OpenAPI
            Route("/health", endpoint=health_endpoint, methods=["GET"]),
            Route("/openapi.json", endpoint=openapi_endpoint, methods=["GET"]),
            # REST API Routes for Open-WebUI & HTTP clients
            Route("/api/remember_fact_or_preference", endpoint=rest_remember, methods=["POST"]),
            Route("/api/save_note", endpoint=rest_save_note, methods=["POST"]),
            Route("/api/remember_facts_batch", endpoint=rest_remember_batch, methods=["POST"]),
            Route("/api/index_document", endpoint=rest_index_doc, methods=["POST"]),
            Route("/api/recall_memories", endpoint=rest_recall, methods=["POST"]),
            Route("/api/search_documents", endpoint=rest_search_docs, methods=["POST"]),
            Route("/api/unified_search", endpoint=rest_unified, methods=["POST"]),
            Route("/api/list_recent_memories", endpoint=rest_list_recent, methods=["GET", "POST"]),
            Route("/api/list_all_documents", endpoint=rest_list_docs, methods=["GET", "POST"]),
            Route("/api/get_memory_details", endpoint=rest_get_memory, methods=["POST"]),
            Route("/api/get_document_full_content", endpoint=rest_get_doc_full, methods=["POST"]),
            Route("/api/update_memory", endpoint=rest_update_memory, methods=["POST"]),
            Route("/api/forget_memory", endpoint=rest_forget_memory, methods=["POST"]),
            Route("/api/delete_document", endpoint=rest_delete_doc, methods=["POST"]),
            Route("/api/list_topics_and_tags", endpoint=rest_topics, methods=["GET"]),
            Route("/api/inspect_brain_health", endpoint=rest_health_info, methods=["GET"]),
            Route("/api/get_brain_guide", endpoint=rest_guide, methods=["GET"]),
            Route("/api/clear_brain", endpoint=rest_clear_brain, methods=["POST"]),
            # Classic SSE (Claude Code type=sse): GET /sse + POST /messages
            Route("/sse", endpoint=sse_endpoint, methods=["GET"]),
            Mount("/messages", app=sse.handle_post_message),
            # Streamable HTTP (Claude Code type=http): POST/GET /mcp only
            Route("/mcp", endpoint=streamable_asgi, methods=["POST", "GET", "DELETE"]),
            Route("/", endpoint=streamable_asgi, methods=["POST", "GET", "DELETE"]),
        ],
        lifespan=lifespan,
    ))

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
