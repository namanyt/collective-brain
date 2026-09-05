"""Core management of Qdrant, FastEmbed embeddings, cross-encoder reranking, and chunking.

Clients interact exclusively through abstract brain tools; this module encapsulates
all collection setup, schema indexing, vector transformations, and housekeeping.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import psutil
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import config

logger = logging.getLogger("brain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _sigmoid(x: float) -> float:
    """Normalize unbounded cross-encoder logits to a 0.0 - 1.0 confidence score."""
    clamped = max(min(x, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-clamped))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_now() -> int:
    return int(time.time())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:16]


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_URL_RE = re.compile(r"https?://[^\s]+", re.I)
_QUOTED_RE = re.compile(r'"([^"]{2,80})"|\'([^\']{2,80})\'')
_IDENT_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-._:][A-Za-z0-9]+)+\b")
_PORT_RE = re.compile(r"\bport\s+(\d{2,5})\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")
_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "is", "are", "was", "were",
    "what", "who", "where", "when", "why", "how", "does", "do", "did", "can",
    "my", "me", "i", "we", "you", "your", "our", "and", "or", "with", "from",
    "about", "this", "that", "these", "those", "please", "tell", "find",
    "search", "recall", "remember", "memory", "memories", "know", "known",
    "any", "some", "which", "into", "over", "under", "not", "no", "just",
    "like", "have", "has", "been", "will", "would", "should", "could",
}


def extract_search_tokens(query: str) -> list[str]:
    """Distinctive tokens for payload text match (ids, emails, IPs, keywords)."""
    seen: set[str] = set()
    tokens: list[str] = []

    def add(raw: str) -> None:
        tok = (raw or "").strip().lower().rstrip(".,);[]{}")
        if len(tok) >= 2 and tok not in seen and tok not in _STOPWORDS:
            seen.add(tok)
            tokens.append(tok)

    for rx in (_EMAIL_RE, _IPV4_RE, _URL_RE, _IDENT_RE):
        for m in rx.finditer(query):
            add(m.group(0))
    for m in _QUOTED_RE.finditer(query):
        add(m.group(1) or m.group(2) or "")
    for m in _PORT_RE.finditer(query):
        add(m.group(1))
    for m in re.finditer(r"\b(\d{4,5})\b", query):
        n = m.group(1)
        if not n.startswith(("19", "20")):
            add(n)
    # Generic leftover words only if the query has no distinctive token.
    if not tokens:
        for w in _WORD_RE.findall(query):
            add(w)
            if len(tokens) >= 6:
                break
    return tokens[:8]


def _stitch_overlapping_chunks(chunks: list[str], overlap: int) -> str:
    """Join chunked text while dropping the duplicated overlap tails."""
    if not chunks:
        return ""
    out = chunks[0]
    for nxt in chunks[1:]:
        if not nxt:
            continue
        max_ol = min(len(out), len(nxt), max(overlap + 80, 40))
        matched = 0
        for k in range(max_ol, 19, -1):
            if out[-k:] == nxt[:k]:
                matched = k
                break
        out = out + nxt[matched:] if matched else f"{out}\n\n{nxt}"
    return out


def _collapse_near_identical(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop later near-copies from an already-ranked list. Does not delete points."""
    kept: list[dict[str, Any]] = []
    for r in results:
        text = _normalize_text(str(r.get("text") or ""))
        duplicate = False
        for prev in kept:
            other = _normalize_text(str(prev.get("text") or ""))
            if not text or not other:
                continue
            if text == other or text[:160] == other[:160]:
                duplicate = True
                break
            n = min(len(text), len(other), 400)
            if n >= 80:
                same = sum(1 for a, b in zip(text[:n], other[:n]) if a == b)
                if same / n >= 0.92:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(r)
    return kept


def _text_index_params() -> models.TextIndexParams:
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=40,
        lowercase=True,
        phrase_matching=True,
    )


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text intelligently using markdown headers, paragraphs, sentences, and overlap."""
    clean_text = text.strip()
    if len(clean_text) <= chunk_size:
        return [clean_text] if clean_text else []

    # Split into sections by markdown headers or paragraphs
    sections = re.split(r"(\n#{1,6}\s+[^\n]+\n|\n\n+)", clean_text)
    blocks: list[str] = []
    current_block = ""

    for sec in sections:
        if not sec:
            continue
        if len(current_block) + len(sec) <= chunk_size:
            current_block += sec
        else:
            if current_block.strip():
                blocks.append(current_block.strip())
            current_block = sec

    if current_block.strip():
        blocks.append(current_block.strip())

    # Build overlapping chunks from blocks
    chunks: list[str] = []
    accum = ""

    for block in blocks:
        if len(block) > chunk_size:
            # Fallback split for huge unbroken blocks (e.g. large code or continuous lines)
            step = max(chunk_size - chunk_overlap, 100)
            for i in range(0, len(block), step):
                sub = block[i : i + chunk_size].strip()
                if sub:
                    chunks.append(sub)
        else:
            if not accum:
                accum = block
            elif len(accum) + len(block) + 1 <= chunk_size:
                accum = f"{accum}\n\n{block}"
            else:
                chunks.append(accum)
                # Keep tail for overlap
                overlap_text = accum[-chunk_overlap:] if len(accum) > chunk_overlap else accum
                accum = f"{overlap_text}\n\n{block}"

    if accum.strip() and (not chunks or chunks[-1] != accum):
        chunks.append(accum.strip())

    return chunks or [clean_text]


class BrainManager:
    def __init__(self) -> None:
        logger.info("Initializing Qdrant client at %s", config.QDRANT_URL)
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT,
        )

        logger.info("Loading FastEmbed embedding model: %s (CPU ONNX, threads=%d)", config.EMBEDDING_MODEL, config.ONNX_THREADS)
        self.embedding_model = TextEmbedding(
            model_name=config.EMBEDDING_MODEL,
            threads=config.ONNX_THREADS,
        )

        logger.info("Loading FastEmbed cross-encoder reranker: %s (CPU ONNX, threads=%d)", config.RERANKER_MODEL, config.ONNX_THREADS)
        self.reranker_model = TextCrossEncoder(
            model_name=config.RERANKER_MODEL,
            threads=config.ONNX_THREADS,
        )

        # ONNX InferenceSession is not thread-safe; serialize embed + rerank.
        self._infer_lock = threading.Lock()

        self._ensure_collections()
        self._warmup_models()

    def _ensure_payload_indexes(
        self,
        collection_name: str,
        indexes: dict[str, models.PayloadSchemaType | models.TextIndexParams],
    ) -> None:
        """Create payload indexes only if they do not already exist."""
        try:
            col_info = self.client.get_collection(collection_name)
            existing = col_info.payload_schema or {}
        except Exception:
            existing = {}

        for field, schema_type in indexes.items():
            if field not in existing:
                logger.info("Creating index for '%s' in %s", field, collection_name)
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema_type,
                        wait=False,
                    )
                except Exception as e:
                    logger.warning("Index creation for '%s' returned: %s", field, e)

    def _ensure_collections(self) -> None:
        """Create and configure backend collections and payload indexes automatically."""
        # 1. Collective Memories Collection
        if not self.client.collection_exists(config.MEMORY_COLLECTION):
            logger.info("Creating collection %s (vectors=%d)", config.MEMORY_COLLECTION, config.VECTOR_DIM)
            self.client.create_collection(
                collection_name=config.MEMORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=config.VECTOR_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes(
            config.MEMORY_COLLECTION,
            {
                "category": models.PayloadSchemaType.KEYWORD,
                "tags": models.PayloadSchemaType.KEYWORD,
                "source": models.PayloadSchemaType.KEYWORD,
                "created_at_ts": models.PayloadSchemaType.INTEGER,
                "content_hash": models.PayloadSchemaType.KEYWORD,
                "title": _text_index_params(),
                "text": _text_index_params(),
            },
        )

        # 2. Collective Documents Collection
        if not self.client.collection_exists(config.DOCUMENT_COLLECTION):
            logger.info("Creating collection %s (vectors=%d)", config.DOCUMENT_COLLECTION, config.VECTOR_DIM)
            self.client.create_collection(
                collection_name=config.DOCUMENT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=config.VECTOR_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        self._ensure_payload_indexes(
            config.DOCUMENT_COLLECTION,
            {
                "document_id": models.PayloadSchemaType.KEYWORD,
                "document_title": models.PayloadSchemaType.KEYWORD,
                "category": models.PayloadSchemaType.KEYWORD,
                "tags": models.PayloadSchemaType.KEYWORD,
                "source": models.PayloadSchemaType.KEYWORD,
                "chunk_index": models.PayloadSchemaType.INTEGER,
                "text": _text_index_params(),
            },
        )

    # -------------------------------------------------------------------------
    # Embeddings & Reranker Helpers
    # -------------------------------------------------------------------------

    def _warmup_models(self) -> None:
        """Run a tiny inference pass so the first real query does not pay ONNX init cost."""
        try:
            self.embed_text("warmup")
            self.rerank_candidates("warmup", ["warmup candidate"])
            logger.info("Embedding and reranker models warmed up")
        except Exception as e:
            logger.warning("Model warmup failed: %s", e)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute 384d ONNX vectors for a batch of strings."""
        if not texts:
            return []
        batch_size = max(1, min(len(texts), 8))
        with self._infer_lock:
            vectors = list(self.embedding_model.embed(texts, batch_size=batch_size, parallel=None))
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    def embed_text(self, text: str) -> list[float]:
        """Compute 384d ONNX vector for a single string."""
        return self.embed_texts([text])[0]

    def rerank_candidates(self, query: str, candidate_texts: list[str]) -> list[float]:
        """Score candidate texts against query using the cross-encoder model."""
        if not candidate_texts:
            return []
        truncated = [(t or "")[: config.RERANK_MAX_CHARS] for t in candidate_texts]
        with self._infer_lock:
            scores = list(
                self.reranker_model.rerank(
                    query,
                    truncated,
                    batch_size=max(1, config.RERANK_BATCH_SIZE),
                )
            )
        return [float(s) for s in scores]

    def _scroll_all(
        self,
        collection_name: str,
        scroll_filter: models.Filter | None = None,
        page_size: int = 200,
        max_points: int = 10000,
    ) -> list[Any]:
        offset = None
        out: list[Any] = []
        while len(out) < max_points:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=min(page_size, max_points - len(out)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            out.extend(records)
            if offset is None:
                break
        return out

    def _text_match_candidates(
        self,
        collection_name: str,
        query: str,
        query_filter: models.Filter | None,
        text_fields: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        tokens = extract_search_tokens(query)
        if not tokens:
            return []

        should: list[models.FieldCondition] = []
        for tok in tokens:
            for field in text_fields:
                if any(ch in tok for ch in ".-_@:/"):
                    should.append(
                        models.FieldCondition(key=field, match=models.MatchPhrase(phrase=tok))
                    )
                else:
                    should.append(
                        models.FieldCondition(key=field, match=models.MatchText(text=tok))
                    )
        if not should:
            return []

        extra = models.Filter(should=should)
        if query_filter and query_filter.must:
            extra = models.Filter(must=list(query_filter.must), should=should)

        try:
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=extra,
                limit=max(limit, 1),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("text match scroll failed on %s: %s", collection_name, e)
            return []

        results: list[dict[str, Any]] = []
        for r in records:
            payload = dict(r.payload or {})
            blob = " ".join(
                str(payload.get(k) or "")
                for k in ("text", "title", "document_title")
            ).lower()
            if not any(tok in blob for tok in tokens):
                continue
            payload["vector_score"] = 0.0
            payload["text_match"] = True
            if payload.get("id") is None and r.id is not None:
                payload["id"] = str(r.id)
            results.append(payload)
        return results

    def _finalize_ranked_results(self, results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not results:
            return []
        tss = [int(r.get("created_at_ts") or 0) for r in results]
        tmin, tmax = min(tss), max(tss)
        span = max(tmax - tmin, 1)
        boost = config.RECENCY_TIE_BOOST
        for r, ts in zip(results, tss):
            rec = (ts - tmin) / span
            r["_sort"] = float(r.get("relevance_confidence") or 0.0) + boost * rec
        results.sort(key=lambda r: r["_sort"], reverse=True)
        for r in results:
            r.pop("_sort", None)
        return _collapse_near_identical(results)[:limit]

    def _vector_search(
        self,
        collection_name: str,
        query: str,
        limit: int,
        query_filter: models.Filter | None,
        use_reranker: bool,
        text_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense search plus payload text match, then optional rerank.

        Unrelated queries still get nearest neighbors from Qdrant (cosine ~0.55-0.63).
        Without a score floor the cross-encoder then reranks those misses, which is
        slow and can stall this process when the container is already at its RAM cap.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        t0 = time.perf_counter()
        query_vector = self.embed_text(clean_query)

        candidate_limit = min(max(limit, 1) * 4, config.MAX_CANDIDATES_FOR_RERANK) if use_reranker else limit
        score_threshold = config.VECTOR_SCORE_THRESHOLD if config.VECTOR_SCORE_THRESHOLD > 0 else None

        search_res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=max(candidate_limit, 1),
            with_payload=True,
            score_threshold=score_threshold,
            timeout=int(config.QDRANT_TIMEOUT),
        )

        by_id: dict[str, dict[str, Any]] = {}
        for p in search_res.points:
            payload = dict(p.payload or {})
            payload["vector_score"] = round(float(p.score), 4)
            pid = str(payload.get("id") or p.id)
            payload["id"] = pid
            by_id[pid] = payload

        for hit in self._text_match_candidates(
            collection_name=collection_name,
            query=clean_query,
            query_filter=query_filter,
            text_fields=text_fields or ["text", "title"],
            limit=max(candidate_limit, 1),
        ):
            pid = str(hit.get("id") or "")
            if not pid:
                continue
            if pid in by_id:
                by_id[pid]["text_match"] = True
            else:
                by_id[pid] = hit

        results = list(by_id.values())
        if not results:
            logger.info(
                "search %s query=%r candidates=0 elapsed=%.3fs",
                collection_name,
                clean_query[:80],
                time.perf_counter() - t0,
            )
            return []

        if use_reranker and results:
            candidate_texts = [str(r.get("text") or r.get("title") or "") for r in results]
            raw_scores = self.rerank_candidates(clean_query, candidate_texts)
            for idx, r in enumerate(results):
                raw = raw_scores[idx] if idx < len(raw_scores) else -20.0
                r["rerank_raw_score"] = round(raw, 4)
                conf = round(_sigmoid(raw), 4)
                if r.get("text_match"):
                    conf = max(conf, 0.55)
                r["relevance_confidence"] = conf
            min_conf = config.MIN_RELEVANCE_CONFIDENCE
            if min_conf > 0:
                results = [
                    r for r in results
                    if r["relevance_confidence"] >= min_conf or r.get("text_match")
                ]
        else:
            for r in results:
                if r.get("text_match") and not r.get("vector_score"):
                    r["relevance_confidence"] = 0.7
                else:
                    r["relevance_confidence"] = round(float(r.get("vector_score") or 0.0), 4)

        results = self._finalize_ranked_results(results, limit)
        logger.info(
            "search %s query=%r candidates=%d returned=%d rerank=%s elapsed=%.3fs",
            collection_name,
            clean_query[:80],
            len(by_id),
            len(results),
            use_reranker,
            time.perf_counter() - t0,
        )
        return results

    # -------------------------------------------------------------------------
    # Memories, Notes, and Facts
    # -------------------------------------------------------------------------

    def _find_near_duplicate(
        self,
        vector: list[float],
        category: str | None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Return (point_id, payload) if an existing memory is almost the same text."""
        must: list[models.FieldCondition] = []
        if category:
            must.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category.strip().lower()))
            )
        q_filter = models.Filter(must=must) if must else None
        try:
            res = self.client.query_points(
                collection_name=config.MEMORY_COLLECTION,
                query=vector,
                query_filter=q_filter,
                limit=1,
                with_payload=True,
                score_threshold=config.NEAR_DUP_COSINE_THRESHOLD,
                timeout=int(config.QDRANT_TIMEOUT),
            )
        except Exception as e:
            logger.warning("near-dup search failed: %s", e)
            return None
        if not res.points:
            return None
        p = res.points[0]
        payload = dict(p.payload or {})
        return str(payload.get("id") or p.id), payload

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        title: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Store a single memory, fact, note, or preference.

        If an existing memory in the same category is nearly identical (cosine >=
        NEAR_DUP_COSINE_THRESHOLD), update that point instead of inserting a twin.
        """
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")
        if len(clean_content) > config.MAX_MEMORY_CHARS:
            raise ValueError(f"Memory content exceeds {config.MAX_MEMORY_CHARS} characters.")

        clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
        clean_category = category.strip().lower()
        clean_title = (
            title.strip()
            if title
            else (clean_content[:60] + "..." if len(clean_content) > 60 else clean_content)
        )
        clean_source = (source or "agent").strip()
        now_iso = _iso_now()
        now_ts = _timestamp_now()
        vector = self.embed_text(clean_content)

        existing = self._find_near_duplicate(vector, clean_category)
        if existing:
            point_id, current = existing
            payload = {
                **current,
                "id": point_id,
                "text": clean_content,
                "title": clean_title,
                "category": clean_category,
                "tags": clean_tags or current.get("tags", []),
                "source": clean_source or current.get("source", "agent"),
                "content_hash": _content_hash(clean_content),
                "updated_at": now_iso,
            }
            payload.pop("updated", None)
            self.client.upsert(
                collection_name=config.MEMORY_COLLECTION,
                points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            return {**payload, "updated": True}

        point_id = str(uuid.uuid4())
        payload = {
            "id": point_id,
            "text": clean_content,
            "title": clean_title,
            "category": clean_category,
            "tags": clean_tags,
            "source": clean_source,
            "content_hash": _content_hash(clean_content),
            "created_at": now_iso,
            "created_at_ts": now_ts,
            "updated_at": now_iso,
        }
        self.client.upsert(
            collection_name=config.MEMORY_COLLECTION,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return {**payload, "updated": False}

    def remember_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Store multiple facts or memories, updating near-duplicates in place."""
        if not items:
            return []
        if len(items) > config.MAX_BATCH_ITEMS:
            raise ValueError(f"Batch exceeds {config.MAX_BATCH_ITEMS} items.")

        results: list[dict[str, Any]] = []
        for item in items:
            text = str(item.get("content", item.get("fact", ""))).strip()
            if not text:
                raise ValueError("All batch memory items must have non-empty content.")
            results.append(
                self.remember(
                    content=text,
                    category=str(item.get("category", "fact")),
                    tags=item.get("tags") or [],
                    title=item.get("title"),
                    source=item.get("source"),
                )
            )
        return results

    def recall_memories(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        tag: str | None = None,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search with optional Cross-Encoder reranking over memories/notes."""
        must_conditions: list[models.FieldCondition] = []
        if category:
            must_conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category.strip().lower()))
            )
        if tag:
            must_conditions.append(
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag.strip().lower()))
            )
        q_filter = models.Filter(must=must_conditions) if must_conditions else None
        return self._vector_search(
            collection_name=config.MEMORY_COLLECTION,
            query=query,
            limit=limit,
            query_filter=q_filter,
            use_reranker=use_reranker,
            text_fields=["text", "title"],
        )

    def list_recent_memories(
        self,
        limit: int = 10,
        category: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recently saved memories/notes without vector search."""
        must_conditions: list[models.FieldCondition] = []
        if category:
            must_conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category.strip().lower()))
            )
        if tag:
            must_conditions.append(
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag.strip().lower()))
            )
        s_filter = models.Filter(must=must_conditions) if must_conditions else None

        records, _ = self.client.scroll(
            collection_name=config.MEMORY_COLLECTION,
            scroll_filter=s_filter,
            limit=min(limit, 50),
            with_payload=True,
            with_vectors=False,
            order_by=models.OrderBy(key="created_at_ts", direction=models.Direction.DESC),
        )
        return [dict(r.payload or {}) for r in records]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a single memory/note by ID."""
        mid = memory_id.strip()
        if not mid:
            return None
        try:
            records = self.client.retrieve(
                collection_name=config.MEMORY_COLLECTION,
                ids=[mid],
                with_payload=True,
            )
        except Exception as e:
            logger.warning("get_memory(%s) failed: %s", mid, e)
            return None
        if not records:
            return None
        return dict(records[0].payload or {})

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        title: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update existing memory content, metadata, and re-embed if content changed."""
        current = self.get_memory(memory_id)
        if not current:
            raise ValueError(f"Memory with ID {memory_id} not found.")

        updated_text = content.strip() if content is not None else current["text"]
        updated_title = title.strip() if title is not None else current.get("title", "")
        updated_cat = category.strip().lower() if category is not None else current.get("category", "fact")
        updated_tags = [t.strip().lower() for t in tags] if tags is not None else current.get("tags", [])

        payload = {
            **current,
            "text": updated_text,
            "title": updated_title,
            "category": updated_cat,
            "tags": updated_tags,
            "content_hash": _content_hash(updated_text),
            "updated_at": _iso_now(),
        }
        payload.pop("updated", None)

        # If content changed, re-calculate vector
        vector = self.embed_text(updated_text) if content is not None else None

        if vector is not None:
            self.client.upsert(
                collection_name=config.MEMORY_COLLECTION,
                points=[models.PointStruct(id=memory_id, vector=vector, payload=payload)],
            )
        else:
            self.client.set_payload(
                collection_name=config.MEMORY_COLLECTION,
                payload=payload,
                points=[memory_id],
            )

        return payload

    def forget_memory(self, memory_id: str) -> bool:
        """Delete a memory from the brain by ID. Returns False if it was not found."""
        current = self.get_memory(memory_id)
        if not current:
            return False
        self.client.delete(
            collection_name=config.MEMORY_COLLECTION,
            points_selector=models.PointIdsList(points=[memory_id.strip()]),
        )
        return True

    # -------------------------------------------------------------------------
    # Long-Form Documents and Knowledge Base
    # -------------------------------------------------------------------------

    def index_document(
        self,
        title: str,
        content: str,
        category: str = "doc",
        tags: list[str] | None = None,
        source: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Intelligently chunk and index a long-form document."""
        clean_title = title.strip()
        clean_content = content.strip()
        if not clean_title or not clean_content:
            raise ValueError("Document title and content cannot be empty.")
        if len(clean_content) > config.MAX_DOCUMENT_CHARS:
            raise ValueError(f"Document content exceeds {config.MAX_DOCUMENT_CHARS} characters.")

        c_size = chunk_size or config.CHUNK_SIZE_DEFAULT
        c_overlap = chunk_overlap or config.CHUNK_OVERLAP_DEFAULT

        # Split into chunks
        chunks = chunk_text(clean_content, chunk_size=c_size, chunk_overlap=c_overlap)
        if not chunks:
            raise ValueError("No text chunks could be extracted from content.")

        # Batch embed all chunks
        vectors = self.embed_texts(chunks)

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        now_iso = _iso_now()
        now_ts = _timestamp_now()
        clean_tags = [t.strip().lower() for t in (tags or []) if str(t).strip()]
        total_chunks = len(chunks)

        points: list[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            payload = {
                "id": point_id,
                "document_id": document_id,
                "document_title": clean_title,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "text": chunk,
                "category": category.strip().lower(),
                "tags": clean_tags,
                "source": (source or "agent").strip(),
                "created_at": now_iso,
                "created_at_ts": now_ts,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=config.DOCUMENT_COLLECTION, points=points)

        return {
            "document_id": document_id,
            "document_title": clean_title,
            "total_chunks": total_chunks,
            "category": category,
            "tags": clean_tags,
            "created_at": now_iso,
            "preview": chunks[0][:150] + "...",
        }

    def search_documents(
        self,
        query: str,
        limit: int = 5,
        tag: str | None = None,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search with cross-encoder reranking across all indexed document chunks."""
        must_conditions: list[models.FieldCondition] = []
        if tag:
            must_conditions.append(
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag.strip().lower()))
            )
        q_filter = models.Filter(must=must_conditions) if must_conditions else None
        return self._vector_search(
            collection_name=config.DOCUMENT_COLLECTION,
            query=query,
            limit=limit,
            query_filter=q_filter,
            use_reranker=use_reranker,
            text_fields=["text"],
        )

    def list_documents(self, limit: int = 20, tag: str | None = None) -> list[dict[str, Any]]:
        """List distinct indexed documents in the knowledge base."""
        must_conditions: list[models.FieldCondition] = []
        if tag:
            must_conditions.append(
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag.strip().lower()))
            )
        s_filter = models.Filter(must=must_conditions) if must_conditions else None

        records = self._scroll_all(
            config.DOCUMENT_COLLECTION,
            scroll_filter=s_filter,
            page_size=200,
            max_points=5000,
        )

        docs_map: dict[str, dict[str, Any]] = {}
        for r in records:
            p = r.payload or {}
            doc_id = p.get("document_id")
            if not doc_id or doc_id in docs_map:
                continue
            docs_map[doc_id] = {
                "document_id": doc_id,
                "document_title": p.get("document_title", "Untitled"),
                "category": p.get("category", "doc"),
                "tags": p.get("tags", []),
                "total_chunks": p.get("total_chunks", 1),
                "created_at": p.get("created_at", ""),
                "source": p.get("source", "agent"),
            }
            if len(docs_map) >= limit:
                break

        return list(docs_map.values())

    def get_document_full(self, document_id_or_title: str) -> dict[str, Any] | None:
        """Reconstruct the complete text of an indexed document from its chunks."""
        identifier = document_id_or_title.strip()

        filter_doc = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=identifier))]
        )
        records = self._scroll_all(config.DOCUMENT_COLLECTION, scroll_filter=filter_doc)

        if not records:
            filter_title = models.Filter(
                must=[models.FieldCondition(key="document_title", match=models.MatchValue(value=identifier))]
            )
            records = self._scroll_all(config.DOCUMENT_COLLECTION, scroll_filter=filter_title)

        if not records:
            return None

        chunks_data = [r.payload for r in records if r.payload]
        chunks_data.sort(key=lambda x: x.get("chunk_index", 0))
        texts = [c.get("text", "") for c in chunks_data]
        meta = chunks_data[0]
        overlap = config.CHUNK_OVERLAP_DEFAULT

        return {
            "document_id": meta.get("document_id"),
            "document_title": meta.get("document_title"),
            "category": meta.get("category"),
            "tags": meta.get("tags"),
            "total_chunks": len(chunks_data),
            "created_at": meta.get("created_at"),
            "content": _stitch_overlapping_chunks(texts, overlap),
        }

    def delete_document(self, document_id_or_title: str) -> bool:
        """Delete all chunks for a document by its document_id or title."""
        identifier = document_id_or_title.strip()
        looks_like_id = identifier.startswith("doc_") and " " not in identifier
        key = "document_id" if looks_like_id else "document_title"
        self.client.delete(
            collection_name=config.DOCUMENT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key=key, match=models.MatchValue(value=identifier))]
                )
            ),
        )
        return True

    # -------------------------------------------------------------------------
    # Unified Search Across Everything
    # -------------------------------------------------------------------------

    def unified_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search across both memories and documents, cross-encoder rerank the combined results."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # Retrieve candidates from both collections
        memories = self.recall_memories(clean_query, limit=limit * 2, use_reranker=False)
        for m in memories:
            m["source_type"] = "memory"

        doc_chunks = self.search_documents(clean_query, limit=limit * 2, use_reranker=False)
        for d in doc_chunks:
            d["source_type"] = "document"

        combined = memories + doc_chunks
        if not combined:
            return []

        if len(combined) == 1:
            return combined[:limit]

        candidate_texts = [str(item.get("text") or "") for item in combined]
        raw_scores = self.rerank_candidates(clean_query, candidate_texts)

        for idx, item in enumerate(combined):
            raw = raw_scores[idx] if idx < len(raw_scores) else -20.0
            item["rerank_raw_score"] = round(raw, 4)
            conf = round(_sigmoid(raw), 4)
            if item.get("text_match"):
                conf = max(conf, 0.55)
            item["relevance_confidence"] = conf

        min_conf = config.MIN_RELEVANCE_CONFIDENCE
        if min_conf > 0:
            combined = [
                item for item in combined
                if item["relevance_confidence"] >= min_conf or item.get("text_match")
            ]
        return self._finalize_ranked_results(combined, limit)

    # -------------------------------------------------------------------------
    # Brain Insights & Diagnostics
    # -------------------------------------------------------------------------

    def list_topics_and_tags(self) -> dict[str, Any]:
        """Aggregate all tags and categories from the brain."""
        mem_records = self._scroll_all(config.MEMORY_COLLECTION, max_points=5000)
        doc_records = self._scroll_all(config.DOCUMENT_COLLECTION, max_points=5000)

        categories: set[str] = set()
        tags: set[str] = set()

        for r in mem_records:
            p = r.payload or {}
            if p.get("category"):
                categories.add(p["category"])
            for t in p.get("tags", []):
                tags.add(t)

        for r in doc_records:
            p = r.payload or {}
            if p.get("category"):
                categories.add(p["category"])
            for t in p.get("tags", []):
                tags.add(t)

        return {
            "categories": sorted(list(categories)),
            "tags": sorted(list(tags)),
            "total_categories": len(categories),
            "total_tags": len(tags),
        }

    def inspect_health(self) -> dict[str, Any]:
        """System health, collection counts, and memory footprint."""
        mem_info = self.client.get_collection(config.MEMORY_COLLECTION)
        doc_info = self.client.get_collection(config.DOCUMENT_COLLECTION)

        proc = psutil.Process()
        proc_mem_mb = round(proc.memory_info().rss / (1024 * 1024), 1)

        sys_mem = psutil.virtual_memory()
        avail_gb = round(sys_mem.available / (1024**3), 2)
        total_gb = round(sys_mem.total / (1024**3), 2)

        return {
            "status": "healthy",
            "qdrant_url": config.QDRANT_URL,
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "vector_dimension": config.VECTOR_DIM,
            "total_memories": mem_info.points_count,
            "total_document_chunks": doc_info.points_count,
            "process_memory_mb": proc_mem_mb,
            "system_available_ram_gb": avail_gb,
            "system_total_ram_gb": total_gb,
        }

    def clear_brain(self, target: str = "memories") -> dict[str, Any]:
        """Clear memories or documents safely."""
        target_clean = target.strip().lower()
        if target_clean in ("memories", "all"):
            self.client.delete_collection(config.MEMORY_COLLECTION)
        if target_clean in ("documents", "all"):
            self.client.delete_collection(config.DOCUMENT_COLLECTION)

        self._ensure_collections()
        return {"cleared": target_clean, "status": "reset_complete"}
