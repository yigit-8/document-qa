"""
Document Q&A API.

POST /ingest        - upload a PDF or text file to the vector store
POST /ask           - ask a question, get an answer with sources and excerpts
POST /ask/stream    - same but streams the answer token by token
GET  /documents     - list all indexed documents
DELETE /documents   - clear the entire vector store
GET  /health        - readiness probe

Requires ANTHROPIC_API_KEY environment variable.
"""

import os
import shutil
import sqlite3
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel

from src.config import (
    ANTHROPIC_MODEL,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DEFAULT_K,
    EMBED_MODEL,
    EXCERPT_CHARS,
    MAX_TOKENS,
)

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Extensions we know how to load. Anything else falls back to the content type.
KNOWN_SUFFIXES = (".pdf", ".txt")

vectorstore = None
qa_chain = None
llm = None


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT,
            chunks     INTEGER,
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_document(filename: str, chunks: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO documents (filename, chunks) VALUES (?, ?)",
            (filename, chunks),
        )
        conn.commit()
    finally:
        conn.close()


def safe_display_name(filename: str | None) -> str:
    """Reduce a client-supplied filename to something safe to store and echo.

    Only ever used as a label (SQLite row, JSON response) -- never to build a
    filesystem path.
    """
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    return name or "upload"


def suffix_for(display_name: str, content_type: str | None) -> str:
    """Pick the temp-file extension from a whitelist, not from raw input."""
    ext = os.path.splitext(display_name)[1].lower()
    if ext in KNOWN_SUFFIXES:
        return ext
    return ".pdf" if content_type == "application/pdf" else ".txt"


def build_chain():
    global vectorstore, qa_chain, llm
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    llm = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=api_key, max_tokens=MAX_TOKENS)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K}),
        return_source_documents=True,
    )
    print("QA chain ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    build_chain()
    yield


app = FastAPI(
    title="Document Q&A API",
    description="Upload documents and ask questions about them using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str
    k: int = DEFAULT_K


@app.get("/")
def root():
    return {"message": "Document Q&A API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized.")
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    # Plain `def`, not `async def`: FastAPI runs sync endpoints in a
    # threadpool, so the blocking disk write, the PDF parse and the embedding
    # computation all stay off the event loop. Matches /ask and /documents.
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized.")

    # The temp path is generated, never derived from the upload: a filename
    # like "../../evil.txt" would otherwise escape /tmp, and two uploads of
    # the same name would race on one path. The original name survives only
    # as a display/log label.
    display_name = safe_display_name(file.filename)
    suffix = suffix_for(display_name, file.content_type)

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            if suffix == ".pdf":
                docs = PyPDFLoader(tmp_path).load()
            else:
                docs = TextLoader(tmp_path, encoding="utf-8").load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

        chunks = SPLITTER.split_documents(docs)
        vectorstore.add_documents(chunks)
        log_document(display_name, len(chunks))
        return {"filename": display_name, "chunks_indexed": len(chunks)}
    finally:
        # The temp file is ours alone, so it is always safe to remove.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask")
def ask(request: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Reuse the chain built once at startup for the default k. A different k
    # genuinely needs its own retriever, so that branch rewires a chain around
    # the *already loaded* embedding model, Chroma handle and LLM client --
    # nothing expensive is rebuilt. Overriding qa_chain.retriever.search_kwargs
    # in place would be cheaper still, but this endpoint is sync and therefore
    # runs in a threadpool, so concurrent requests asking for different k
    # values would race on that one shared dict.
    if request.k == DEFAULT_K:
        chain = qa_chain
    else:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": request.k}),
            return_source_documents=True,
        )

    result = chain.invoke({"query": request.question})

    source_docs = result["source_documents"]
    sources = list({doc.metadata.get("source", "unknown") for doc in source_docs})
    excerpts = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
            "text": doc.page_content[:EXCERPT_CHARS],
        }
        for doc in source_docs
    ]

    return {
        "answer": result["result"],
        "sources": sources,
        "excerpts": excerpts,
    }


@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
    docs = retriever.invoke(request.question)
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {request.question}\n\nAnswer:"
    )

    async def generate():
        async for chunk in llm.astream(prompt):
            yield chunk.content

        yield f"\n\n[sources: {', '.join(sources)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/documents")
def list_documents():
    if not os.path.exists(DB_PATH):
        return {"documents": []}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename, chunks, timestamp FROM documents ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return {
        "documents": [{"filename": r[0], "chunks": r[1], "ingested_at": r[2]} for r in rows],
        "total": len(rows),
    }


@app.delete("/documents")
def clear_documents():
    vectorstore.delete_collection()
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM documents")
        conn.commit()
        conn.close()
    return {"message": "All documents cleared."}
