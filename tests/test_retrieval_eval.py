"""
Enforces a minimum retrieval quality bar in CI: if a change to chunking,
the embedding model, or retriever settings makes the retriever worse at
finding the right chunk, this test fails instead of silently shipping a
regression.
"""

import pytest

from src.evaluate_retrieval import build_eval_vectorstore, evaluate_recall_at_k

MIN_RECALL_AT_K = 0.8


@pytest.fixture(scope="module")
def eval_vectorstore():
    return build_eval_vectorstore()


def test_retrieval_recall_meets_minimum_bar(eval_vectorstore):
    report = evaluate_recall_at_k(eval_vectorstore, k=4)
    misses = [r["question"] for r in report["results"] if not r["hit"]]
    assert report["recall_at_k"] >= MIN_RECALL_AT_K, f"Missed: {misses}"
