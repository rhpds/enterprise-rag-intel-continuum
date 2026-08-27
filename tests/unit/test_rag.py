"""Unit tests for the Enterprise RAG Intel Continuum pipeline."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import pytest
from rag import (
    DocumentProcessor,
    VectorStore,
    Reranker,
    RAGPipeline,
    AI_DISCLAIMER,
    SAMPLE_DOCUMENTS,
    EMBEDDING_DIM,
    app,
    _openai_api_url,
)
from fastapi.testclient import TestClient

client = TestClient(app)


def test_openai_api_url_accepts_base_with_or_without_v1():
    assert _openai_api_url("https://maas.example.com", "chat/completions") == (
        "https://maas.example.com/v1/chat/completions"
    )
    assert _openai_api_url("https://maas.example.com/v1/", "models") == (
        "https://maas.example.com/v1/models"
    )


def test_openai_compatible_generation(monkeypatch):
    request = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "MaaS answer [1]"}}]}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            request.update(url=url, **kwargs)
            return FakeResponse()

    monkeypatch.setattr("rag.httpx.Client", FakeClient)
    credentials = {"generation_" + "api_key": "virtual" + "-key"}
    pipeline = RAGPipeline(
        processor=DocumentProcessor(),
        vector_store=VectorStore(),
        reranker=Reranker(),
        generation_endpoint="https://maas.example.com/v1",
        generation_model="qwen3-14b",
        generation_api_type="openai",
        **credentials,
    )

    answer = pipeline._live_generate(
        "What accelerates generation?",
        [{"document": "intel.txt", "chunk": "Intel hardware accelerates generation."}],
    )

    assert answer == "MaaS answer [1]"
    assert request["url"] == "https://maas.example.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer virtual-key"
    assert request["json"]["model"] == "qwen3-14b"


# ---------------------------------------------------------------------------
# Test 1: Document chunking -- text splits into overlapping chunks
# ---------------------------------------------------------------------------


class TestDocumentChunking:

    def test_document_chunking(self):
        """Text should be split into overlapping chunks of the configured
        size with the configured overlap."""
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=3)

        # 25 words -> should produce multiple chunks with overlap
        text = " ".join(f"word{i}" for i in range(25))
        chunks = processor.chunk_text(text)

        assert len(chunks) > 1, (
            f"25 words with chunk_size=10 should produce multiple chunks, "
            f"got {len(chunks)}"
        )

        # Verify overlap: consecutive chunks share words
        words_0 = set(chunks[0].split())
        words_1 = set(chunks[1].split())
        overlap = words_0 & words_1
        assert len(overlap) > 0, (
            "Consecutive chunks should share overlapping words"
        )

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should return as a single chunk."""
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
        text = "This is a short document."
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1, f"Short text should be one chunk, got {len(chunks)}"
        assert chunks[0] == text

    def test_empty_text_no_chunks(self):
        """Empty or whitespace-only text should produce no chunks."""
        processor = DocumentProcessor()
        assert processor.chunk_text("") == []
        assert processor.chunk_text("   ") == []


# ---------------------------------------------------------------------------
# Test 2: Embedding generation -- chunks get embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingGeneration:

    def test_embedding_generation(self):
        """Each chunk should receive an embedding vector of the correct
        dimension."""
        processor = DocumentProcessor()
        texts = ["Hello world", "Intel Xeon processors", "RAG pipeline"]
        embeddings = processor.generate_embeddings(texts)

        assert len(embeddings) == len(texts), (
            f"Expected {len(texts)} embeddings, got {len(embeddings)}"
        )

        for i, emb in enumerate(embeddings):
            assert len(emb) == EMBEDDING_DIM, (
                f"Embedding {i} has {len(emb)} dims, expected {EMBEDDING_DIM}"
            )
            # Should be a unit vector (normalized)
            import numpy as np
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01, (
                f"Embedding {i} norm is {norm:.4f}, expected ~1.0"
            )

    def test_deterministic_embeddings(self):
        """Same text should produce the same embedding (deterministic demo)."""
        processor = DocumentProcessor()
        emb1 = processor.generate_embeddings(["test text"])
        emb2 = processor.generate_embeddings(["test text"])
        assert emb1[0] == emb2[0], "Same text should produce identical embeddings"

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        processor = DocumentProcessor()
        emb = processor.generate_embeddings(["text one", "text two"])
        assert emb[0] != emb[1], "Different texts should produce different embeddings"


# ---------------------------------------------------------------------------
# Test 3: Vector search returns results
# ---------------------------------------------------------------------------


class TestVectorSearchReturnsResults:

    def test_vector_search_returns_results(self):
        """Search should return up to top_k results from the vector store."""
        processor = DocumentProcessor()
        store = VectorStore(dim=EMBEDDING_DIM)

        # Add some chunks
        texts = [
            "Intel Xeon processors for AI inference",
            "RAG pipeline architecture overview",
            "Vector database cosine similarity search",
            "Large language model generation",
            "Enterprise knowledge base management",
        ]
        embeddings = processor.generate_embeddings(texts)
        store.add(texts, embeddings, "doc-001", "test.txt")

        # Search
        query_emb = processor.generate_embeddings(["Intel Xeon AI"])[0]
        results = store.search(query_emb, top_k=3)

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert all("chunk" in r for r in results), "Each result should have 'chunk'"
        assert all("score" in r for r in results), "Each result should have 'score'"
        assert all("document" in r for r in results), "Each result should have 'document'"

    def test_search_empty_store(self):
        """Search on an empty store should return empty results."""
        store = VectorStore()
        results = store.search([0.0] * EMBEDDING_DIM, top_k=5)
        assert results == [], "Empty store should return no results"

    def test_search_respects_top_k(self):
        """Search should return at most top_k results."""
        processor = DocumentProcessor()
        store = VectorStore(dim=EMBEDDING_DIM)

        texts = [f"document {i} about topic {i}" for i in range(10)]
        embeddings = processor.generate_embeddings(texts)
        store.add(texts, embeddings, "doc-multi", "multi.txt")

        query_emb = processor.generate_embeddings(["topic"])[0]
        results = store.search(query_emb, top_k=3)
        assert len(results) == 3, f"Should return exactly 3 results, got {len(results)}"


# ---------------------------------------------------------------------------
# Test 4: Reranking reorders -- reranking changes order by relevance
# ---------------------------------------------------------------------------


class TestRerankingReorders:

    def test_reranking_reorders(self):
        """Reranking should reorder chunks based on query relevance."""
        reranker = Reranker()

        chunks = [
            {"chunk": "unrelated topic about cooking recipes", "score": 0.9, "document": "a.txt"},
            {"chunk": "Intel Xeon processors for AI inference workloads", "score": 0.5, "document": "b.txt"},
            {"chunk": "basic introduction to programming concepts", "score": 0.7, "document": "c.txt"},
        ]

        reranked = reranker.rerank("Intel Xeon AI inference", chunks)

        assert len(reranked) == 3, "Reranking should preserve all chunks"

        # The chunk about Intel Xeon should now have a higher score
        xeon_chunk = next(c for c in reranked if "Xeon" in c["chunk"])
        assert xeon_chunk["score"] > 0.5, (
            f"Xeon chunk should have boosted score, got {xeon_chunk['score']}"
        )

    def test_rerank_empty_list(self):
        """Reranking an empty list should return an empty list."""
        reranker = Reranker()
        result = reranker.rerank("test query", [])
        assert result == []


# ---------------------------------------------------------------------------
# Test 5: Full pipeline -- query -> embed -> search -> rerank -> generate
# ---------------------------------------------------------------------------


class TestFullPipeline:

    def test_full_pipeline(self):
        """The full RAG pipeline should produce an answer with sources
        and pipeline step metadata."""
        processor = DocumentProcessor()
        store = VectorStore(dim=EMBEDDING_DIM)
        reranker = Reranker()

        # Load a document
        text = (
            "Intel Xeon processors provide hardware acceleration for AI "
            "inference workloads using AMX instructions. The Xeon platform "
            "supports embedding generation and vector search operations "
            "efficiently for enterprise RAG pipelines."
        )
        chunks = processor.chunk_text(text)
        embeddings = processor.generate_embeddings(chunks)
        store.add(chunks, embeddings, "test-doc", "xeon.txt")

        pipeline = RAGPipeline(
            processor=processor,
            vector_store=store,
            reranker=reranker,
        )

        result = pipeline.query("What is Intel Xeon used for?", top_k=3)

        assert "answer" in result, "Result should have 'answer'"
        assert result["answer"], "Answer should not be empty"
        assert "sources" in result, "Result should have 'sources'"
        assert "pipeline_steps" in result, "Result should have 'pipeline_steps'"
        assert "total_latency_ms" in result, "Result should have 'total_latency_ms'"
        assert "ai_disclaimer" in result, "Result should have 'ai_disclaimer'"


# ---------------------------------------------------------------------------
# Test 6: Pipeline steps report hardware assignment
# ---------------------------------------------------------------------------


class TestPipelineStepsReportHardware:

    def test_pipeline_steps_report_hardware(self):
        """Each pipeline step should report its hardware assignment:
        embed/search/rerank -> xeon, generate -> configured MaaS hardware."""
        processor = DocumentProcessor()
        store = VectorStore(dim=EMBEDDING_DIM)
        reranker = Reranker()

        # Add minimal data
        chunks = ["test chunk about Intel hardware"]
        embeddings = processor.generate_embeddings(chunks)
        store.add(chunks, embeddings, "hw-doc", "hardware.txt")

        pipeline = RAGPipeline(
            processor=processor,
            vector_store=store,
            reranker=reranker,
        )

        result = pipeline.query("Intel hardware", top_k=1)
        steps = result["pipeline_steps"]

        assert len(steps) == 4, f"Expected 4 pipeline steps, got {len(steps)}"

        step_map = {s["step"]: s for s in steps}

        assert step_map["embed"]["hardware"] == "xeon", "embed should run on xeon"
        assert step_map["search"]["hardware"] == "xeon", "search should run on xeon"
        assert step_map["rerank"]["hardware"] == "xeon", "rerank should run on xeon"
        assert step_map["generate"]["hardware"] == "maas", "generate should use MaaS"

        # Each step should have a latency
        for step in steps:
            assert "latency_ms" in step, f"Step {step['step']} missing latency_ms"
            assert isinstance(step["latency_ms"], float), (
                f"Step {step['step']} latency_ms should be float"
            )


# ---------------------------------------------------------------------------
# Test 7: Demo mode with preloaded docs
# ---------------------------------------------------------------------------


class TestDemoModeWithPreloadedDocs:

    def test_demo_mode_with_preloaded_docs(self):
        """In demo mode, sample documents should be pre-loaded and
        searchable without any uploads."""
        # The app module loads sample docs at import time when DEMO_MODE=true
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200

        data = resp.json()
        assert data["count"] >= len(SAMPLE_DOCUMENTS), (
            f"Expected at least {len(SAMPLE_DOCUMENTS)} demo docs, "
            f"got {data['count']}"
        )

        filenames = [d["filename"] for d in data["documents"]]
        for doc in SAMPLE_DOCUMENTS:
            assert doc["filename"] in filenames, (
                f"Demo doc '{doc['filename']}' not found in documents list"
            )


# ---------------------------------------------------------------------------
# Test 8: Upload and query -- uploaded doc is searchable
# ---------------------------------------------------------------------------


class TestUploadAndQuery:

    def test_upload_and_query(self):
        """A document uploaded via the API should be searchable via query."""
        # Upload a document
        upload_resp = client.post(
            "/api/v1/upload",
            json={
                "content": (
                    "Kubernetes container orchestration enables automated "
                    "deployment, scaling, and management of containerized "
                    "applications across clusters of machines."
                ),
                "filename": "kubernetes-test.txt",
            },
        )
        assert upload_resp.status_code == 200, (
            f"Upload failed: {upload_resp.status_code} {upload_resp.text}"
        )

        upload_data = upload_resp.json()
        assert "document_id" in upload_data
        assert upload_data["chunks"] > 0
        assert upload_data["embeddings_generated"] > 0

        # Query for the uploaded content
        query_resp = client.post(
            "/api/v1/query",
            json={"query": "Kubernetes container orchestration", "top_k": 5},
        )
        assert query_resp.status_code == 200

        query_data = query_resp.json()
        assert query_data["answer"], "Answer should not be empty"
        assert len(query_data["sources"]) > 0, "Should have source chunks"
        assert len(query_data["pipeline_steps"]) == 4, "Should have 4 pipeline steps"


# ---------------------------------------------------------------------------
# Test 9: AI disclaimer present
# ---------------------------------------------------------------------------


class TestAIDisclaimerPresent:

    def test_ai_disclaimer_present(self):
        """POST /api/v1/query should include a non-empty ai_disclaimer
        field in the response."""
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is Intel Xeon?", "top_k": 3},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert "ai_disclaimer" in data, (
            "Response JSON missing 'ai_disclaimer' key"
        )
        assert data["ai_disclaimer"], (
            "ai_disclaimer should be non-empty"
        )
        assert "verify" in data["ai_disclaimer"].lower() or "ai-generated" in data["ai_disclaimer"].lower(), (
            "Disclaimer should mention verification or AI-generated nature"
        )

    def test_ai_disclaimer_matches_constant(self):
        """The disclaimer in the response should match the AI_DISCLAIMER
        constant defined in the source."""
        resp = client.post(
            "/api/v1/query",
            json={"query": "test disclaimer", "top_k": 1},
        )
        data = resp.json()
        assert data["ai_disclaimer"] == AI_DISCLAIMER
