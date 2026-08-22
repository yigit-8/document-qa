"""
Shared configuration for ingestion, retrieval and serving.

Chunking, embedding, retrieval and generation settings live here so the
ingest CLI, the API and the retrieval evaluation cannot drift apart. Every
value can be overridden with an environment variable; the defaults are the
values the project ships with.
"""

import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")

# Storage
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(_ROOT, "data", "chroma_db"))
DB_PATH = os.getenv("DB_PATH", os.path.join(_ROOT, "data", "documents.db"))

# Chunking and embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))
EXCERPT_CHARS = int(os.getenv("EXCERPT_CHARS", "300"))

# Generation
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
