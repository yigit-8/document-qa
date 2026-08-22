"""
Document ingestion pipeline.

Reads PDF or plain text files, splits them into chunks,
embeds them, and stores them in ChromaDB.

Usage:
    python -m src.ingest data/docs/my_document.pdf
    python -m src.ingest data/docs/  (ingests all files in a directory)
"""

import argparse
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def get_vectorstore() -> Chroma:
    """Load the embedding model and open Chroma.

    Expensive (the sentence-transformer is read from disk), so callers doing
    more than one file should build it once and pass it down.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def load_file(path: str):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    return TextLoader(path, encoding="utf-8").load()


def ingest_file(path: str, vectorstore: Chroma | None = None) -> int:
    if vectorstore is None:
        vectorstore = get_vectorstore()

    docs = load_file(path)
    chunks = SPLITTER.split_documents(docs)
    vectorstore.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {os.path.basename(path)}")
    return len(chunks)


def ingest_directory(directory: str, vectorstore: Chroma | None = None) -> int:
    # Build the vector store once for the whole batch. Doing it per file
    # reloaded the embedding model and reopened Chroma for every document.
    if vectorstore is None:
        vectorstore = get_vectorstore()

    total = 0
    for filename in os.listdir(directory):
        if filename.endswith((".pdf", ".txt")):
            total += ingest_file(os.path.join(directory, filename), vectorstore)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or directory to ingest")
    args = parser.parse_args()

    store = get_vectorstore()
    if os.path.isdir(args.path):
        total = ingest_directory(args.path, store)
    else:
        total = ingest_file(args.path, store)

    print(f"Total chunks indexed: {total}")
