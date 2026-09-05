"""Configuration settings for the Qdrant Collective Brain MCP Server."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Qdrant connection - defaults to localhost:6333 when running on host,
    # overridden to http://qdrant:6333 in docker-compose.yml
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
    QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "30.0"))

    # Internal collections managed exclusively by MCP
    MEMORY_COLLECTION: str = os.getenv("MEMORY_COLLECTION", "collective_memories")
    DOCUMENT_COLLECTION: str = os.getenv("DOCUMENT_COLLECTION", "collective_documents")

    # FastEmbed Models (ONNX CPU - ultra lightweight & fast)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "384"))

    # Server settings
    PORT: int = int(os.getenv("PORT", "5006"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Document Chunking defaults
    CHUNK_SIZE_DEFAULT: int = int(os.getenv("CHUNK_SIZE_DEFAULT", "800"))
    CHUNK_OVERLAP_DEFAULT: int = int(os.getenv("CHUNK_OVERLAP_DEFAULT", "150"))
    MAX_CANDIDATES_FOR_RERANK: int = int(os.getenv("MAX_CANDIDATES_FOR_RERANK", "8"))

    # Retrieval quality / hang-avoidance.
    # Cosine scores for unrelated bge-small queries land around 0.55-0.63; real hits are typically 0.70+.
    # Below VECTOR_SCORE_THRESHOLD we skip the cross-encoder entirely and return "not found".
    VECTOR_SCORE_THRESHOLD: float = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.64"))
    MIN_RELEVANCE_CONFIDENCE: float = float(os.getenv("MIN_RELEVANCE_CONFIDENCE", "0.05"))
    RERANK_MAX_CHARS: int = int(os.getenv("RERANK_MAX_CHARS", "1200"))
    RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", "4"))

    # ONNX Runtime: 1 thread avoids oversubscription + cgroup memory stalls in the 1GB container.
    ONNX_THREADS: int = int(os.getenv("ONNX_THREADS", "1"))
    TOOL_TIMEOUT: float = float(os.getenv("TOOL_TIMEOUT", "20.0"))

    # Hybrid retrieval / write-time near-dup (does not rewrite existing vectors).
    NEAR_DUP_COSINE_THRESHOLD: float = float(os.getenv("NEAR_DUP_COSINE_THRESHOLD", "0.92"))
    RECENCY_TIE_BOOST: float = float(os.getenv("RECENCY_TIE_BOOST", "0.03"))
    MAX_MEMORY_CHARS: int = int(os.getenv("MAX_MEMORY_CHARS", "20000"))
    MAX_DOCUMENT_CHARS: int = int(os.getenv("MAX_DOCUMENT_CHARS", "200000"))
    MAX_BATCH_ITEMS: int = int(os.getenv("MAX_BATCH_ITEMS", "50"))


config = Config()
