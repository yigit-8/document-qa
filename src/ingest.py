"""
Document ingestion pipeline.

Reads PDF or plain text files, splits them into chunks,
embeds them, and stores them in ChromaDB.

Usage:
    python src/ingest.py data/docs/my_document.pdf
    python src/ingest.py data/docs/  (ingests all files in a directory)
"""

import argparse
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def load_file(path: str):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def ingest_file(path: str) -> int:
    docs = load_file(path)
    chunks = SPLITTER.split_documents(docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {os.path.basename(path)}")
    return len(chunks)


def ingest_directory(directory: str) -> int:
    total = 0
    for filename in os.listdir(directory):
        if filename.endswith((".pdf", ".txt")):
            total += ingest_file(os.path.join(directory, filename))
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or directory to ingest")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        total = ingest_directory(args.path)
    else:
        total = ingest_file(args.path)

    print(f"Total chunks indexed: {total}")
