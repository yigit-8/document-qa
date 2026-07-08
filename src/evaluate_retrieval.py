"""
Retrieval evaluation for the RAG pipeline.

Measures recall@k: for each question in the golden set, whether the chunk
that actually contains the answer was among the top-k chunks the retriever
returned. This is the metric that matters before generation even happens —
if the retriever misses the right chunk, no LLM can produce a grounded
answer. Runs against a small multi-topic corpus with real embeddings
(no ANTHROPIC_API_KEY needed, since generation isn't evaluated here).

Usage:
    python -m src.evaluate_retrieval
"""

import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
CORPUS_PATH = os.path.join(FIXTURES_DIR, "eval_corpus.txt")
QUESTIONS_PATH = os.path.join(FIXTURES_DIR, "eval_questions.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_eval_vectorstore() -> Chroma:
    """In-memory only (no persist_directory) — the eval store is throwaway,
    and skipping disk I/O sidesteps Chroma's file-locking issues on Windows.
    """
    docs = TextLoader(CORPUS_PATH, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(chunks, embeddings)


def evaluate_recall_at_k(vectorstore: Chroma, k: int = 4) -> dict:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = []
    for item in golden:
        retrieved = retriever.invoke(item["question"])
        hit = any(
            item["expected_keyword"].lower() in doc.page_content.lower() for doc in retrieved
        )
        results.append({"question": item["question"], "hit": hit})

    recall = sum(r["hit"] for r in results) / len(results)
    return {"k": k, "recall_at_k": recall, "results": results}


def main():
    vectorstore = build_eval_vectorstore()
    report = evaluate_recall_at_k(vectorstore, k=4)

    print(f"Retrieval evaluation (recall@{report['k']}): {report['recall_at_k']:.2%}\n")
    for r in report["results"]:
        status = "HIT " if r["hit"] else "MISS"
        print(f"  [{status}] {r['question']}")


if __name__ == "__main__":
    main()
