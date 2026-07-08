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
import sqlite3
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

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents.db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

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


def build_chain():
    global vectorstore, qa_chain, llm
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key, max_tokens=1024)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
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
    k: int = 4


@app.get("/")
def root():
    return {"message": "Document Q&A API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized.")
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    suffix = ".pdf" if file.content_type == "application/pdf" else ".txt"
    tmp_path = f"/tmp/{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        else:
            docs = TextLoader(tmp_path, encoding="utf-8").load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    chunks = SPLITTER.split_documents(docs)
    vectorstore.add_documents(chunks)
    log_document(file.filename, len(chunks))
    return {"filename": file.filename, "chunks_indexed": len(chunks)}


@app.post("/ask")
def ask(request: QuestionRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    result = chain.invoke({"query": request.question})

    source_docs = result["source_documents"]
    sources = list({doc.metadata.get("source", "unknown") for doc in source_docs})
    excerpts = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
            "text": doc.page_content[:300],
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
