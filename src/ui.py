"""
Enterprise RAG Intel Continuum -- Gradio UI

Provides a dashboard for querying the RAG pipeline, uploading documents,
browsing the document library, and visualizing pipeline performance
across the Intel inference continuum.
"""

import gradio as gr
import httpx
import os

APP_URL = os.environ.get("APP_URL", "http://localhost:8080")


def _api_get(path: str):
    """GET request to the FastAPI backend."""
    try:
        resp = httpx.get(f"{APP_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise gr.Error(
            f"Cannot connect to backend at {APP_URL}. "
            "Ensure the FastAPI server is running."
        )
    except httpx.HTTPStatusError as exc:
        raise gr.Error(
            f"Backend returned {exc.response.status_code}: "
            f"{exc.response.text}"
        )
    except Exception as exc:
        raise gr.Error(f"Request failed: {exc}")


def _api_post(path: str, json_body: dict):
    """POST request to the FastAPI backend."""
    try:
        resp = httpx.post(f"{APP_URL}{path}", json=json_body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise gr.Error(
            f"Cannot connect to backend at {APP_URL}. "
            "Ensure the FastAPI server is running."
        )
    except httpx.HTTPStatusError as exc:
        raise gr.Error(
            f"Backend returned {exc.response.status_code}: "
            f"{exc.response.text}"
        )
    except Exception as exc:
        raise gr.Error(f"Request failed: {exc}")


# -- Tab 1: Ask a Question ---------------------------------------------------


def ask_question(query: str, top_k: int, rerank: bool):
    """Send a RAG query and display the full pipeline result."""
    if not query or not query.strip():
        raise gr.Error("Enter a question to ask.")

    payload = {
        "query": query.strip(),
        "top_k": int(top_k),
        "rerank": rerank,
    }

    result = _api_post("/api/v1/query", payload)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    steps = result.get("pipeline_steps", [])
    total_ms = result.get("total_latency_ms", 0.0)
    disclaimer = result.get("ai_disclaimer", "")

    # Build source chunks display
    source_rows = []
    for i, src in enumerate(sources, 1):
        chunk_preview = src.get("chunk", "")[:200]
        score = src.get("score", 0.0)
        doc = src.get("document", "unknown")
        source_rows.append(
            f"| {i} | {doc} | {score:.4f} | {chunk_preview}... |"
        )
    sources_table = "\n".join(source_rows) if source_rows else "| - | No sources found | - | - |"

    # Build pipeline steps display
    step_rows = []
    for s in steps:
        step_name = s.get("step", "")
        hw = s.get("hardware", "")
        lat = s.get("latency_ms", 0.0)
        hw_label = f"Intel {hw.title()}" if hw else "unknown"
        step_rows.append(f"| **{step_name}** | {hw_label} | {lat:.1f} ms |")
    steps_table = "\n".join(step_rows)

    output = f"""## Answer

{answer}

---

### Source Chunks

| # | Document | Relevance | Chunk Preview |
|---|----------|-----------|---------------|
{sources_table}

---

### Pipeline Steps

| Step | Hardware | Latency |
|------|----------|---------|
{steps_table}
| **Total** | | **{total_ms:.1f} ms** |

---

*{disclaimer}*
"""
    return output


# -- Tab 2: Upload Documents -------------------------------------------------


def upload_text_document(text_content: str, filename: str):
    """Upload a text document for indexing."""
    if not text_content or not text_content.strip():
        raise gr.Error("Enter or paste document text to index.")
    if not filename or not filename.strip():
        raise gr.Error("Provide a filename for the document.")

    payload = {
        "content": text_content.strip(),
        "filename": filename.strip(),
    }

    result = _api_post("/api/v1/upload", payload)

    doc_id = result.get("document_id", "unknown")
    chunks = result.get("chunks", 0)
    embeddings = result.get("embeddings_generated", 0)

    return f"""## Document Indexed Successfully

| Field | Value |
|-------|-------|
| **Document ID** | `{doc_id}` |
| **Filename** | {filename} |
| **Chunks Created** | {chunks} |
| **Embeddings Generated** | {embeddings} |

The document is now searchable via the RAG pipeline.
"""


# -- Tab 3: Document Library -------------------------------------------------


def load_documents():
    """Load the list of indexed documents."""
    result = _api_get("/api/v1/documents")

    docs = result.get("documents", [])
    count = result.get("count", 0)

    if not docs:
        return "No documents indexed yet. Upload documents in the Upload tab."

    doc_rows = []
    for d in docs:
        doc_id = d.get("document_id", "")[:12] + "..."
        filename = d.get("filename", "unknown")
        chunks = d.get("chunks", 0)
        indexed = d.get("indexed_at", "N/A")
        doc_rows.append(f"| {filename} | {doc_id} | {chunks} | {indexed} |")

    table = "\n".join(doc_rows)

    return f"""## Document Library

**Total Documents:** {count}

| Filename | Document ID | Chunks | Indexed At |
|----------|------------|--------|------------|
{table}
"""


# -- Tab 4: Pipeline Performance ----------------------------------------------


def load_performance():
    """Load pipeline performance statistics."""
    result = _api_get("/api/v1/stats")

    total_q = result.get("total_queries", 0)
    total_docs = result.get("total_documents", 0)
    total_chunks = result.get("total_chunks", 0)
    avg_lat = result.get("average_latency_ms", 0.0)
    hw = result.get("hardware_utilization", {})
    demo = result.get("demo_mode", True)
    uptime = result.get("uptime_seconds", 0.0)

    xeon_steps = hw.get("xeon_steps", 3)
    gaudi_steps = hw.get("gaudi_steps", 1)
    uptime_min = uptime / 60

    continuum_diagram = """```
    Intel Inference Continuum -- RAG Pipeline

    Query
      |
      v
    +---------------------------+
    | EMBED (nomic-embed-text)  |  Intel Xeon
    +---------------------------+
      |
      v
    +---------------------------+
    | SEARCH (numpy cosine)     |  Intel Xeon
    +---------------------------+
      |
      v
    +---------------------------+
    | RERANK (LLM scoring)      |  Intel Xeon
    +---------------------------+
      |
      v
    +---------------------------+
    | GENERATE (RHDP MaaS)      |  Intel-backed model service
    +---------------------------+
      |
      v
    Answer with Sources
    ```"""

    return f"""## Pipeline Performance

| Metric | Value |
|--------|-------|
| **Total Queries** | {total_q} |
| **Total Documents** | {total_docs} |
| **Total Chunks** | {total_chunks} |
| **Average Latency** | {avg_lat:.1f} ms |
| **Demo Mode** | {demo} |
| **Uptime** | {uptime_min:.1f} minutes |

### Hardware Utilization

| Hardware | Pipeline Steps | Role |
|----------|---------------|------|
| **Intel Xeon** | {xeon_steps} steps | Embed, Search, Rerank |
| **Intel-backed MaaS** | {gaudi_steps} step | Generate |

### Intel Inference Continuum

{continuum_diagram}

### Step-by-Step Breakdown

| Step | Hardware | Model/Engine | Purpose |
|------|----------|-------------|---------|
| **Embed** | Intel Xeon | nomic-embed-text | Convert query to dense vector |
| **Search** | Intel Xeon | numpy cosine similarity | Find relevant chunks |
| **Rerank** | Intel Xeon | LLM cross-encoder | Refine relevance ordering |
| **Generate** | Intel-backed MaaS | Allocated model | Synthesize answer from context |

Xeon handles {xeon_steps} of 4 pipeline steps (embedding, search, reranking)
while the RHDP MaaS allocation handles generation. The exact inference
hardware is determined by the model service backing the allocated model.
"""


# -- Build UI -----------------------------------------------------------------

with gr.Blocks(title="Enterprise RAG -- Intel Continuum") as demo:

    # Top banner with disclaimer
    gr.Markdown(
        """
# Enterprise RAG across the Intel Inference Continuum
### Xeon handles retrieval -- RHDP MaaS handles generation

> **Disclaimer:** RAG answers are AI-generated from retrieved context
> -- verify against source documents.
"""
    )

    # -- Tab 1: Ask a Question ------------------------------------------------
    with gr.Tab("Ask a Question"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="What are the benefits of Intel Xeon for AI inference?",
                    lines=3,
                )
                top_k_slider = gr.Slider(
                    label="Top K (number of chunks to retrieve)",
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                )
                rerank_toggle = gr.Checkbox(
                    label="Enable Reranking",
                    value=True,
                )
                ask_btn = gr.Button("Ask", variant="primary")
            with gr.Column():
                answer_output = gr.Markdown(label="RAG Result")

        ask_btn.click(
            fn=ask_question,
            inputs=[query_input, top_k_slider, rerank_toggle],
            outputs=[answer_output],
        )

    # -- Tab 2: Upload Documents ----------------------------------------------
    with gr.Tab("Upload Documents"):
        with gr.Row():
            with gr.Column():
                upload_text = gr.Textbox(
                    label="Document Text",
                    placeholder="Paste or type document content here...",
                    lines=10,
                )
                upload_filename = gr.Textbox(
                    label="Filename",
                    placeholder="e.g. my-document.txt",
                )
                upload_btn = gr.Button("Index Document", variant="primary")
            with gr.Column():
                upload_output = gr.Markdown(label="Upload Result")

        upload_btn.click(
            fn=upload_text_document,
            inputs=[upload_text, upload_filename],
            outputs=[upload_output],
        )

    # -- Tab 3: Document Library ----------------------------------------------
    with gr.Tab("Document Library"):
        docs_output = gr.Markdown(label="Indexed Documents")
        docs_btn = gr.Button("Refresh Library", variant="secondary")

        docs_btn.click(
            fn=load_documents,
            inputs=[],
            outputs=[docs_output],
        )

    # -- Tab 4: Pipeline Performance ------------------------------------------
    with gr.Tab("Pipeline Performance"):
        perf_output = gr.Markdown(label="Performance Stats")
        perf_btn = gr.Button("Refresh Stats", variant="secondary")

        perf_btn.click(
            fn=load_performance,
            inputs=[],
            outputs=[perf_output],
        )

    # Load docs and performance on page load
    def on_load():
        try:
            docs = load_documents()
        except Exception:
            docs = "*Backend not available. Start the server to view documents.*"
        try:
            perf = load_performance()
        except Exception:
            perf = "*Backend not available. Start the server to view performance.*"
        return docs, perf

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[docs_output, perf_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
