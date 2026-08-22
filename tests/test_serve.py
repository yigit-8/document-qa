import io
import os
import tempfile
from unittest.mock import MagicMock, patch

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


class _StreamChunk:
    def __init__(self, content: str):
        self.content = content


async def _fake_astream(prompt):
    for piece in ["Test ", "answer"]:
        yield _StreamChunk(piece)


mock_llm = MagicMock()
mock_llm.astream = _fake_astream

mock_vectorstore = MagicMock()
mock_vectorstore.as_retriever.return_value = MagicMock()
mock_vectorstore.add_documents.return_value = None
mock_vectorstore.as_retriever.return_value.invoke.return_value = [mock_doc]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient whose sqlite DB and Chroma directory live under tmp_path.

    Without this the app would create ``data/documents.db`` inside the repo the
    first time the lifespan handler runs ``init_db()``.
    """
    state_dir = tmp_path_factory.mktemp("document-qa-state")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(serve_module, "DB_PATH", str(state_dir / "documents.db"))
        mp.setattr(serve_module, "CHROMA_DIR", str(state_dir / "chroma_db"))
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


def test_ask_stream_returns_answer_and_sources(client):
    response = client.post("/ask/stream", json={"question": "What is this document about?"})
    assert response.status_code == 200
    body = response.text
    assert "Test answer" in body
    assert "sources:" in body


def test_ask_stream_empty_question_returns_400(client):
    response = client.post("/ask/stream", json={"question": ""})
    assert response.status_code == 400


def test_ingest_traversal_filename_stays_inside_temp_dir(client):
    """A filename like ../../evil.txt must never become part of a real path."""
    escape_targets = [
        os.path.join(tempfile.gettempdir(), "evil.txt"),
        os.path.abspath(os.path.join(tempfile.gettempdir(), "..", "..", "evil.txt")),
        os.path.abspath("evil.txt"),
        "/evil.txt",
    ]
    for target in escape_targets:
        assert not os.path.exists(target), f"precondition: {target} already exists"

    with (
        patch("src.serve.TextLoader") as mock_loader,
        patch("src.serve.SPLITTER") as mock_splitter,
    ):
        mock_loader.return_value.load.return_value = [mock_doc]
        mock_splitter.split_documents.return_value = [mock_doc]

        response = client.post(
            "/ingest",
            files={"file": ("../../evil.txt", io.BytesIO(b"payload"), "text/plain")},
        )

    assert response.status_code == 200
    # The traversal segments are stripped; only a display name survives.
    assert response.json()["filename"] == "evil.txt"

    for target in escape_targets:
        assert not os.path.exists(target), f"file escaped the temp dir: {target}"

    # The loader was handed a generated path, not the client's filename.
    loaded_path = mock_loader.call_args[0][0]
    assert "evil.txt" not in loaded_path
    assert os.path.dirname(loaded_path) == tempfile.gettempdir()

    # That generated temp file is removed once parsing is done.
    assert not os.path.exists(loaded_path)


def test_delete_documents_clears_store(client):
    response = client.delete("/documents")
    assert response.status_code == 200
    assert "message" in response.json()
    mock_vectorstore.delete_collection.assert_called()
    assert client.get("/documents").json()["total"] == 0
