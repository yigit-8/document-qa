import io
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from fastapi.testclient import TestClient

import src.serve as serve_module
from src.serve import app

mock_doc = MagicMock()
mock_doc.metadata = {"source": "test.pdf", "page": 0}
mock_doc.page_content = "Machine learning is a subset of artificial intelligence."

mock_chain = MagicMock()
mock_chain.invoke.return_value = {
    "result": "The document discusses machine learning.",
    "source_documents": [mock_doc],
}

mock_llm = MagicMock()
mock_llm.astream = AsyncMock(return_value=iter([MagicMock(content="Test answer")]))

mock_vectorstore = MagicMock()
mock_vectorstore.as_retriever.return_value = MagicMock()
mock_vectorstore.add_documents.return_value = None
mock_vectorstore.as_retriever.return_value.invoke.return_value = [mock_doc]


@pytest.fixture(scope="module")
def client():
    with (
        patch("src.serve.build_chain"),
        patch("src.serve.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("src.serve.Chroma", return_value=mock_vectorstore),
    ):
        with TestClient(app) as c:
            serve_module.qa_chain = mock_chain
            serve_module.vectorstore = mock_vectorstore
            serve_module.llm = mock_llm
            yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_ask_returns_answer_with_excerpts(client):
    with patch("src.serve.RetrievalQA") as mock_qa:
        mock_qa.from_chain_type.return_value = mock_chain
        response = client.post("/ask", json={"question": "What is this document about?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "excerpts" in data


def test_ask_empty_question_returns_400(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_custom_k(client):
    with patch("src.serve.RetrievalQA") as mock_qa:
        mock_qa.from_chain_type.return_value = mock_chain
        response = client.post("/ask", json={"question": "Test question", "k": 6})
    assert response.status_code == 200


def test_documents_returns_list(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert "documents" in response.json()
    assert "total" in response.json()


def test_ingest_text_file(client):
    with (
        patch("src.serve.TextLoader") as mock_loader,
        patch("src.serve.SPLITTER") as mock_splitter,
    ):
        mock_loader.return_value.load.return_value = [mock_doc]
        mock_splitter.split_documents.return_value = [mock_doc, mock_doc]

        content = b"Machine learning is a subset of artificial intelligence."
        response = client.post(
            "/ingest",
            files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
        )
    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 2
