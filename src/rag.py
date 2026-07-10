"""
Enterprise RAG across the Intel Inference Continuum -- FastAPI Application

Adapted from the Red Hat Intel Partner Demo RAG pipeline. Provides document
upload, chunking, embedding (nomic-embed-text), in-memory numpy vector search,
LLM-based reranking, and answer generation (qwen2.5:1.5b) with source
attribution and per-step hardware assignment reporting.

Intel Continuum mapping:
  - Embed query    -> Xeon (nomic-embed-text)
  - Vector search  -> Xeon (numpy cosine similarity)
  - Rerank         -> Xeon (LLM scoring)
  - Generate       -> Gaudi (qwen2.5:1.5b)
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "")
GENERATION_ENDPOINT = os.environ.get("GENERATION_ENDPOINT", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwen2.5:1.5b")
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() == "true"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

AI_DISCLAIMER = (
    "RAG answers are AI-generated from retrieved context "
    "-- verify against source documents."
)

# ---------------------------------------------------------------------------
# Sample Documents for Demo Mode
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    {
        "filename": "intel-xeon-scalable.txt",
        "content": (
            "Intel Xeon Scalable processors deliver workload-optimized "
            "performance for demanding enterprise AI and data center workloads. "
            "The latest generation features Intel Advanced Matrix Extensions "
            "(AMX) for accelerated AI inference and training directly on the "
            "CPU. Built-in accelerators for AI, analytics, networking, and "
            "storage reduce the need for discrete accelerators for many "
            "workloads. Intel Xeon supports up to 64 cores per socket with "
            "DDR5 memory and PCIe 5.0 connectivity. The platform provides "
            "enterprise-grade reliability, security, and manageability "
            "features including Intel Software Guard Extensions (SGX) for "
            "confidential computing and Intel Trust Domain Extensions (TDX) "
            "for VM-level isolation. Xeon processors are optimized for "
            "embedding generation, vector similarity search, and reranking "
            "in RAG pipelines, handling these compute-intensive but "
            "latency-tolerant tasks efficiently."
        ),
    },
    {
        "filename": "intel-gaudi-accelerator.txt",
        "content": (
            "Intel Gaudi accelerators are purpose-built for deep learning "
            "training and inference workloads. Gaudi 2 delivers up to 2x "
            "the inference throughput compared to the previous generation "
            "for large language model serving. The architecture features "
            "integrated networking with RDMA over Converged Ethernet (RoCE) "
            "for efficient multi-node scaling. Gaudi processors include a "
            "Matrix Math Engine optimized for transformer architectures "
            "and a cluster of Tensor Processing Cores for parallel execution. "
            "In RAG pipelines, Gaudi handles the generation step where "
            "low-latency token generation is critical for user experience. "
            "The combination of Xeon for preprocessing steps and Gaudi for "
            "generation creates an efficient inference continuum that "
            "optimizes both cost and performance across the full pipeline."
        ),
    },
    {
        "filename": "openshift-ai-platform.txt",
        "content": (
            "Red Hat OpenShift AI provides a platform for building, deploying, "
            "and managing AI and machine learning applications at enterprise "
            "scale. It integrates with the OpenShift container platform to "
            "deliver consistent environments from development to production. "
            "OpenShift AI supports model serving with vLLM, TGI, and other "
            "inference runtimes, enabling organizations to deploy large "
            "language models on Intel hardware. The platform includes "
            "built-in model registry, experiment tracking, and pipeline "
            "orchestration. For RAG workloads, OpenShift AI provides "
            "Kubernetes-native deployment of embedding services, vector "
            "databases, and generation endpoints with automatic scaling "
            "and health monitoring. Integration with Intel hardware "
            "accelerators is managed through device plugins and operator "
            "frameworks."
        ),
    },
    {
        "filename": "rag-pipeline-architecture.txt",
        "content": (
            "A Retrieval-Augmented Generation pipeline combines information "
            "retrieval with language model generation to produce grounded, "
            "source-attributed answers. The pipeline consists of four key "
            "steps: embedding, search, reranking, and generation. In the "
            "embedding step, the user query is converted into a dense vector "
            "representation using a model like nomic-embed-text. The search "
            "step performs cosine similarity matching against a vector store "
            "of pre-indexed document chunks to find the most relevant "
            "passages. Reranking uses a cross-encoder or LLM-based scorer "
            "to refine the initial retrieval results by evaluating query-chunk "
            "relevance more precisely. Finally, the generation step uses a "
            "large language model to synthesize an answer from the top-ranked "
            "chunks, providing source attribution so users can verify claims "
            "against the original documents. This architecture reduces "
            "hallucination and enables enterprise knowledge base applications."
        ),
    },
]


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------


class DocumentProcessor:
    """Accepts text or PDF content, chunks with overlap, and generates
    embeddings via the configured embedding endpoint (nomic-embed-text)."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_endpoint: str = "",
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks of approximately chunk_size
        tokens (words as proxy)."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def extract_text(self, filename: str, content: bytes) -> str:
        """Extract text from file content. Supports plain text and basic
        PDF extraction."""
        if filename.lower().endswith(".pdf"):
            return self._extract_pdf(content)
        return content.decode("utf-8", errors="ignore")

    def _extract_pdf(self, content: bytes) -> str:
        """Best-effort PDF text extraction."""
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = []
                for page in pdf.pages[:50]:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                return "\n\n".join(pages)
        except ImportError:
            pass
        return content.decode("utf-8", errors="ignore")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts. Uses the Ollama
        embedding API when available, falls back to deterministic
        demo embeddings."""
        if self.embedding_endpoint and not DEMO_MODE:
            return self._live_embeddings(texts)
        return self._demo_embeddings(texts)

    def _live_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama /api/embed endpoint."""
        endpoint = self.embedding_endpoint.rstrip("/")
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{endpoint}/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": texts,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("embeddings", [])
        except Exception:
            return self._demo_embeddings(texts)

    def _demo_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic demo embeddings from text content.
        Uses a hash-seeded random vector so the same text always
        produces the same embedding."""
        embeddings = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(EMBEDDING_DIM).astype(np.float64)
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


# ---------------------------------------------------------------------------
# Vector Store (in-memory, numpy-based)
# ---------------------------------------------------------------------------


class VectorStore:
    """In-memory numpy-based vector store with cosine similarity search.
    No pgvector dependency -- designed for laptop/demo mode."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.embeddings: list[np.ndarray] = []
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
    ) -> int:
        """Add chunks with their embeddings to the store. Returns count added."""
        added = 0
        for chunk, emb in zip(chunks, embeddings):
            vec = np.array(emb, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings.append(vec)
            self.chunks.append(chunk)
            self.metadata.append({
                "document_id": document_id,
                "filename": filename,
            })
            added += 1
        return added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Cosine similarity search. Returns top_k results sorted by score."""
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float64)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Stack all embeddings into a matrix for vectorized cosine similarity
        matrix = np.stack(self.embeddings)
        scores = matrix @ query_vec

        # Get top_k indices
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "document": self.metadata[idx]["filename"],
                "document_id": self.metadata[idx]["document_id"],
            })
        return results

    def get_documents(self) -> list[dict]:
        """Return a deduplicated list of indexed documents."""
        seen = {}
        for meta in self.metadata:
            doc_id = meta["document_id"]
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "chunks": 0,
                }
            seen[doc_id]["chunks"] += 1
        return list(seen.values())


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Rerank retrieved chunks by relevance. Uses LLM-based scoring when
    an endpoint is available, otherwise falls back to score-based sorting."""

    def __init__(
        self,
        generation_endpoint: str = "",
        generation_model: str = GENERATION_MODEL,
    ):
        self.generation_endpoint = generation_endpoint
        self.generation_model = generation_model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[dict]:
        """Rerank chunks by relevance to the query."""
        if not chunks:
            return []

        if self.generation_endpoint and not DEMO_MODE:
            return self._llm_rerank(query, chunks)
        return self._demo_rerank(query, chunks)

    def _llm_rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Use LLM to score relevance of each chunk to the query."""
        try:
            scored = []
            endpoint = self.generation_endpoint.rstrip("/")
            with httpx.Client(timeout=30.0) as client:
                for chunk_data in chunks:
                    prompt = (
                        f"Rate the relevance of the following text to the query "
                        f"on a scale of 0 to 10. Respond with ONLY a number.\n\n"
                        f"Query: {query}\n\nText: {chunk_data['chunk'][:300]}\n\n"
                        f"Relevance score (0-10):"
                    )
                    resp = client.post(
                        f"{endpoint}/api/chat",
                        json={
                            "model": self.generation_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": 5, "temperature": 0.0},
                        },
                    )
                    resp.raise_for_status()
                    content = resp.json().get("message", {}).get("content", "5")
                    # Extract numeric score
                    try:
                        score = float("".join(c for c in content if c.isdigit() or c == ".") or "5")
                        score = min(10.0, max(0.0, score)) / 10.0
                    except ValueError:
                        score = chunk_data.get("score", 0.5)
                    scored.append({**chunk_data, "score": score})
            return sorted(scored, key=lambda c: c["score"], reverse=True)
        except Exception:
            return self._demo_rerank(query, chunks)

    def _demo_rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Demo reranking: boost chunks containing query terms, then sort."""
        query_terms = set(query.lower().split())
        scored = []
        for chunk_data in chunks:
            chunk_lower = chunk_data["chunk"].lower()
            # Count query term matches as relevance signal
            matches = sum(1 for term in query_terms if term in chunk_lower)
            boost = matches / max(len(query_terms), 1) * 0.3
            new_score = min(1.0, chunk_data.get("score", 0.5) + boost)
            scored.append({**chunk_data, "score": round(new_score, 4)})
        return sorted(scored, key=lambda c: c["score"], reverse=True)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """Orchestrates the 4-step RAG pipeline: embed -> search -> rerank ->
    generate. Each step reports its hardware assignment and latency.

    Hardware mapping:
      embed    -> xeon  (nomic-embed-text)
      search   -> xeon  (numpy cosine similarity)
      rerank   -> xeon  (LLM scoring)
      generate -> gaudi (qwen2.5:1.5b)
    """

    def __init__(
        self,
        processor: DocumentProcessor,
        vector_store: VectorStore,
        reranker: Reranker,
        generation_endpoint: str = "",
        generation_model: str = GENERATION_MODEL,
    ):
        self.processor = processor
        self.vector_store = vector_store
        self.reranker = reranker
        self.generation_endpoint = generation_endpoint
        self.generation_model = generation_model

    def query(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> dict:
        """Execute the full RAG pipeline and return the result with
        per-step hardware and latency reporting."""
        pipeline_steps = []
        total_start = time.time()

        # Step 1: Embed query (Xeon)
        step_start = time.time()
        query_embeddings = self.processor.generate_embeddings([query])
        query_embedding = query_embeddings[0] if query_embeddings else [0.0] * EMBEDDING_DIM
        embed_latency = round((time.time() - step_start) * 1000, 1)
        pipeline_steps.append({
            "step": "embed",
            "hardware": "xeon",
            "latency_ms": embed_latency,
        })

        # Step 2: Search (Xeon)
        step_start = time.time()
        results = self.vector_store.search(query_embedding, top_k=top_k)
        search_latency = round((time.time() - step_start) * 1000, 1)
        pipeline_steps.append({
            "step": "search",
            "hardware": "xeon",
            "latency_ms": search_latency,
        })

        # Step 3: Rerank (Xeon)
        step_start = time.time()
        if rerank and results:
            results = self.reranker.rerank(query, results)
        rerank_latency = round((time.time() - step_start) * 1000, 1)
        pipeline_steps.append({
            "step": "rerank",
            "hardware": "xeon",
            "latency_ms": rerank_latency,
        })

        # Step 4: Generate (Gaudi)
        step_start = time.time()
        answer = self._generate_answer(query, results)
        generate_latency = round((time.time() - step_start) * 1000, 1)
        pipeline_steps.append({
            "step": "generate",
            "hardware": "gaudi",
            "latency_ms": generate_latency,
        })

        total_latency = round((time.time() - total_start) * 1000, 1)

        sources = [
            {
                "chunk": r["chunk"][:500],
                "score": round(r["score"], 4),
                "document": r["document"],
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "pipeline_steps": pipeline_steps,
            "total_latency_ms": total_latency,
            "ai_disclaimer": AI_DISCLAIMER,
        }

    def _generate_answer(self, query: str, chunks: list[dict]) -> str:
        """Generate an answer from retrieved chunks using the LLM."""
        if self.generation_endpoint and not DEMO_MODE:
            return self._live_generate(query, chunks)
        return self._demo_generate(query, chunks)

    def _live_generate(self, query: str, chunks: list[dict]) -> str:
        """Generate answer via Ollama chat API."""
        context_parts = []
        for i, chunk in enumerate(chunks[:4], 1):
            context_parts.append(f"[Source {i}: {chunk['document']}]\n{chunk['chunk'][:400]}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question ONLY "
            "using the document context provided below. If the answer is not "
            "in the documents, say so. Be concise -- use bullet points, keep "
            "responses under 300 words. Cite source numbers [1], [2] etc.\n\n"
            f"<document_context>\n{context}\n</document_context>"
        )

        try:
            endpoint = self.generation_endpoint.rstrip("/")
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{endpoint}/api/chat",
                    json={
                        "model": self.generation_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                        "stream": False,
                        "options": {"num_predict": 512, "temperature": 0.3},
                    },
                )
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "")
        except Exception:
            return self._demo_generate(query, chunks)

    def _demo_generate(self, query: str, chunks: list[dict]) -> str:
        """Generate a demo answer by synthesizing from retrieved chunks."""
        if not chunks:
            return (
                "No relevant documents found in the knowledge base. "
                "Please upload documents first or refine your query."
            )

        # Build a demo answer from the top chunks
        top_chunks = chunks[:3]
        source_refs = []
        for i, chunk in enumerate(top_chunks, 1):
            preview = chunk["chunk"][:150].rstrip()
            if not preview.endswith("."):
                # Find last sentence boundary
                last_dot = preview.rfind(".")
                if last_dot > 50:
                    preview = preview[:last_dot + 1]
                else:
                    preview += "..."
            source_refs.append(f"- {preview} [Source {i}: {chunk['document']}]")

        sources_text = "\n".join(source_refs)

        return (
            f"Based on the indexed documents, here is what I found regarding "
            f"your query:\n\n{sources_text}\n\n"
            f"The retrieved context spans {len(top_chunks)} source(s) with "
            f"relevance scores ranging from "
            f"{min(c['score'] for c in top_chunks):.2f} to "
            f"{max(c['score'] for c in top_chunks):.2f}."
        )


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TextUploadRequest(BaseModel):
    content: str = Field(..., description="Text content to index")
    filename: str = Field(..., description="Document name")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=50)
    rerank: bool = Field(True, description="Whether to rerank results")


class UploadResponse(BaseModel):
    document_id: str
    chunks: int
    embeddings_generated: int


class SourceChunkModel(BaseModel):
    chunk: str
    score: float
    document: str


class PipelineStepModel(BaseModel):
    step: str
    hardware: str
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunkModel]
    pipeline_steps: list[PipelineStepModel]
    total_latency_ms: float
    ai_disclaimer: str = AI_DISCLAIMER


class DocumentInfoModel(BaseModel):
    document_id: str
    filename: str
    chunks: int
    indexed_at: Optional[str] = None


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfoModel]
    count: int


class StatsResponse(BaseModel):
    total_queries: int
    total_documents: int
    total_chunks: int
    average_latency_ms: float
    hardware_utilization: dict
    demo_mode: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

_start_time = time.time()

_stats = {
    "total_queries": 0,
    "total_latency": 0.0,
}

# Initialize components
processor = DocumentProcessor(
    embedding_endpoint=EMBEDDING_ENDPOINT,
    embedding_model=EMBEDDING_MODEL,
)
vector_store = VectorStore(dim=EMBEDDING_DIM)
reranker = Reranker(
    generation_endpoint=GENERATION_ENDPOINT,
    generation_model=GENERATION_MODEL,
)
pipeline = RAGPipeline(
    processor=processor,
    vector_store=vector_store,
    reranker=reranker,
    generation_endpoint=GENERATION_ENDPOINT,
    generation_model=GENERATION_MODEL,
)

# Document index tracking
_document_index: dict[str, dict] = {}


def _load_sample_documents():
    """Pre-load sample documents for demo mode."""
    for doc in SAMPLE_DOCUMENTS:
        doc_id = str(uuid.uuid4())
        chunks = processor.chunk_text(doc["content"])
        embeddings = processor.generate_embeddings(chunks)
        vector_store.add(chunks, embeddings, doc_id, doc["filename"])
        _document_index[doc_id] = {
            "document_id": doc_id,
            "filename": doc["filename"],
            "chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }


# Load samples on startup if in demo mode
if DEMO_MODE:
    _load_sample_documents()


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Intel Continuum",
    version="1.0.0",
    description=(
        "Full RAG pipeline across the Intel inference continuum. "
        + AI_DISCLAIMER
    ),
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "enterprise-rag-intel-continuum",
        "version": "1.0.0",
        "demo_mode": DEMO_MODE,
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
    }


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_document_json(body: TextUploadRequest):
    """Upload a document for RAG indexing via JSON text body."""
    text = body.content
    filename = body.filename

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document content is empty.")

    doc_id = str(uuid.uuid4())
    chunks = processor.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks extracted from document.")

    embeddings = processor.generate_embeddings(chunks)
    vector_store.add(chunks, embeddings, doc_id, filename)

    _document_index[doc_id] = {
        "document_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }

    return UploadResponse(
        document_id=doc_id,
        chunks=len(chunks),
        embeddings_generated=len(embeddings),
    )


@app.post("/api/v1/upload/file", response_model=UploadResponse)
async def upload_document_file(file: UploadFile = File(...)):
    """Upload a document for RAG indexing via multipart file upload."""
    content = await file.read()
    filename = file.filename or "uploaded.txt"
    text = processor.extract_text(filename, content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document content is empty.")

    doc_id = str(uuid.uuid4())
    chunks = processor.chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks extracted from document.")

    embeddings = processor.generate_embeddings(chunks)
    vector_store.add(chunks, embeddings, doc_id, filename)

    _document_index[doc_id] = {
        "document_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }

    return UploadResponse(
        document_id=doc_id,
        chunks=len(chunks),
        embeddings_generated=len(embeddings),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Execute the full RAG pipeline."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = pipeline.query(
        query=req.query,
        top_k=req.top_k,
        rerank=req.rerank,
    )

    _stats["total_queries"] += 1
    _stats["total_latency"] += result["total_latency_ms"]

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunkModel(**s) for s in result["sources"]],
        pipeline_steps=[PipelineStepModel(**s) for s in result["pipeline_steps"]],
        total_latency_ms=result["total_latency_ms"],
        ai_disclaimer=result["ai_disclaimer"],
    )


@app.get("/api/v1/documents", response_model=DocumentsResponse)
async def list_documents():
    """List all indexed documents."""
    docs = list(_document_index.values())
    return DocumentsResponse(
        documents=[DocumentInfoModel(**d) for d in docs],
        count=len(docs),
    )


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Pipeline statistics."""
    total_q = _stats["total_queries"]
    avg_latency = (
        round(_stats["total_latency"] / total_q, 1) if total_q > 0 else 0.0
    )
    return StatsResponse(
        total_queries=total_q,
        total_documents=len(_document_index),
        total_chunks=vector_store.size,
        average_latency_ms=avg_latency,
        hardware_utilization={
            "xeon_steps": 3,
            "gaudi_steps": 1,
        },
        demo_mode=DEMO_MODE,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
