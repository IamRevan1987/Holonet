# 🌐 Holonet — Vectorless RAG Engine

**Tree-indexed retrieval without embeddings, powered by a 1B parameter model.**

Holonet is a Retrieval-Augmented Generation (RAG) system that replaces traditional vector databases with a **hierarchical tree index**. Documents are parsed into a nested node structure, summarized bottom-up by an LLM, and queried using structured reasoning — no embedding models, no vector stores, no cosine similarity.

**Lab Results**: The first ingestion tests happened fast. The largest of documents, a 13MB PDF with 96 pages of content, was processed in as little as **15 minutes!** I managed to ingest 32 PDFs **(a total of 93.7MB)** in less than 6 hours, which is a huge improvement over standard RAG API ingestion with both **1B and 4B models!**

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Drop PDFs and/or TXT files into the Library
cp your_documents/*.pdf Knowledgebase_Alpha/Library/

# 3. Run ingestion + queries
uv run python Program_Ingest/ingest.py
```

> **Prerequisites**: [Python 3.13+](https://python.org), [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.ai) with `gemma3:1b` pulled.

---

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │  PDF / TXT  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  pymupdf4llm │  ← PDF → Markdown (with pymupdf_layout)
                    │  or raw read │  ← TXT → Markdown
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Markdown   │  ← Cached in markdown_cache/
                    │  Splitter   │  ← Splits on # / ## / ### headers
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Tree Index │  ← Hierarchical node structure
                    │  Builder    │  ← Each node: title, content, children
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  LLM        │  ← gemma3:1b via Ollama
                    │  Summarizer │  ← Bottom-up: leaves first, then parents
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Tree JSON  │  ← Saved as holonet_resources.json
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Query Pipeline         │
              │  1. Structured search   │  ← LLM reasons over tree structure
              │  2. Node retrieval      │  ← Extract content from matched nodes
              │  3. Answer generation   │  ← Context-grounded response
              └─────────────────────────┘
```

### Why Vectorless?

| Aspect | Traditional RAG | Holonet |
|---|---|---|
| **Retrieval** | Cosine similarity on embeddings | LLM reasons over tree structure |
| **Index** | Vector store (FAISS, Chroma, etc.) | JSON tree with summaries |
| **Dependencies** | Embedding model + vector DB | Single LLM only |
| **Explainability** | Opaque similarity scores | Structured reasoning trace |
| **Hardware** | GPU for embeddings + inference | GPU for inference only |

---

## 📁 Project Structure

```
Holonet/
├── Program_Ingest/
│   ├── ingest.py              ← Main application (ingestion + query engine)
│   └── OriginCode.txt         ← Original prototype reference
├── Knowledgebase_Alpha/
│   ├── Library/               ← Drop your PDFs and TXT files here
│   ├── markdown_cache/        ← Intermediate .md files (auto-generated)
│   ├── holonet_resources.json ← Master tree index (auto-generated)
│   └── processed_files.json   ← Ingestion manifest (auto-generated)
├── pyproject.toml             ← Project metadata and dependencies
└── README.md                  ← This file
```

---

## 📖 How It Works

### Ingestion Pipeline

1. **Scan** — The app scans `Knowledgebase_Alpha/Library/` for `.pdf` and `.txt` files
2. **Convert** — PDFs are converted to Markdown via `pymupdf4llm` (enhanced by `pymupdf_layout` for better page structure detection); TXT files are read directly
3. **Fallback Chunking** — If a document produces fewer than 2 markdown headers, the system automatically inserts `## Page N` markers to ensure the tree has useful depth
4. **Tree Build** — Markdown is split on `#`, `##`, `###` headers into a hierarchical `TreeNode` structure
5. **Summarize** — Each node is summarized bottom-up by the LLM (leaves first, then parents aggregate child summaries)
6. **Merge & Save** — All document trees are merged under a master root and saved to `holonet_resources.json`
7. **Manifest** — Each processed file is tracked by name and modification time in `processed_files.json` to prevent re-processing

### Query Pipeline

1. **Tree Search** — The LLM receives the full tree structure (titles + summaries, no content) and reasons about which nodes are relevant to the query
2. **Content Retrieval** — Content is extracted from the selected nodes, respecting a token budget (4,500 tokens max)
3. **Answer Generation** — The LLM generates a grounded answer using the retrieved context

---

## 🔧 Configuration

Key constants in `ingest.py` that can be tuned:

| Constant | Default | Purpose |
|---|---|---|
| `MAX_CONTEXT_TOKENS` | 4500 | Token budget for retrieved context per query |
| `MAX_RETRIEVAL_NODES` | 4 | Max nodes to pull content from per query |
| `MAX_SUMMARY_TOKENS` | 3000 | Max tokens fed to the summarizer per node |
| `FALLBACK_CHUNK_TOKENS` | 1500 | Token window size for fallback page chunking |
| `MIN_HEADER_COUNT` | 2 | Threshold below which fallback chunking activates |

### Model Configuration

The default model is `gemma3:1b` running through Ollama with these settings:

```python
model = ChatOllama(
    model="gemma3:1b",
    temperature=0,       # Deterministic outputs
    num_ctx=6000,        # Context window
    seed=24,             # Reproducibility
    num_predict=1024,    # Max output tokens
    num_thread=12,       # CPU threads
    num_gpu=1,           # GPU layers
    timeout=120,         # Per-request timeout (seconds)
)
```

---

## ⏱️ Performance Benchmarks

Benchmarks from a test run with 3 PDFs on a system with `gemma3:1b` via Ollama:

### Ingestion Speed

| PDF | File Size | Markdown Output | Tree Nodes | Ingestion Time |
|---|---|---|---|---|
| Assets, Threats & Vulnerabilities | 2.2 MB | 150,468 chars | 22 nodes | **12.0 min** |
| Detection and Response | 4.1 MB | 134,882 chars | 22 nodes | **13.0 min** |
| Python Essentials 1 | 2.6 MB | 65,252 chars | 5 nodes | **~3 min** |

### Performance Characteristics

- **Bottleneck**: Node summarization by the LLM (~30-40s per node with `gemma3:1b`)
- **PDF conversion** is near-instant (seconds, not minutes)
- **Node count is the primary time driver**, not file size
- PDFs with proper `#` headers produce fewer, more targeted nodes (faster)
- PDFs without headers trigger fallback chunking (~1 node per page, ~20 nodes for a 20-page doc)

### Estimated Ingestion Times by Document Size

| Category | Typical Node Count | Time per Document |
|---|---|---|
| **Small** (<1 MB, <20 pages) | 5–15 nodes | **3–8 min** |
| **Medium** (1–5 MB, 20–80 pages) | 15–25 nodes | **10–15 min** |
| **Large** (5–13 MB, 80–200+ pages) | 25–60 nodes | **20–60 min** |

### Query Latency

Each query involves two LLM calls (tree search + answer generation):
- **Per-query time**: ~30 seconds - 2 minutes with `gemma3:1b`
- Faster models (e.g., `gemma3:4b` with more GPU) will reduce this proportionally

---

## 🛡️ Edge Cases & Robustness

| Scenario | Behavior |
|---|---|
| PDF with no markdown headers | Fallback chunker inserts `## Page N` markers per page |
| Empty or corrupt PDF | Caught, logged with error message, skipped |
| Empty or unreadable TXT | Caught, logged, skipped |
| Already-processed file | Manifest check by name + mtime, skipped silently |
| Modified file (same name, new content) | Re-ingested automatically |
| Empty Library folder | Graceful message, clean exit |
| Process crash mid-batch | Manifest saves per-file; re-run continues where it left off |
| Large content exceeding token limits | `trim_to_token_limit()` enforced before all LLM calls |
| Deep tree rendering | Display capped at depth 4 to prevent terminal overflow |
| Memory usage > 10 GB | Safety threshold raises `MemoryError` |

---

## 🔌 Integration

The core functions can be imported directly into another application:

```python
from Program_Ingest.ingest import load_tree, retrieve_and_answer, KB_FILE

# Load the pre-built tree
tree = load_tree(KB_FILE)

# Query it
answer, node_ids, reasoning = retrieve_and_answer(tree, "What is NIST 800-63?")
print(answer)
```

---

## 📋 Supported File Types

| Extension | Converter | Notes |
|---|---|---|
| `.pdf` | `pymupdf4llm` + `pymupdf_layout` | Best results with well-formatted PDFs that have heading structure |
| `.txt` | Direct read (UTF-8) | Automatically wrapped with fallback chunking |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` | LLM orchestration framework |
| `langchain-ollama` | Ollama model integration |
| `langchain-text-splitters` | `MarkdownHeaderTextSplitter` for tree building |
| `pymupdf4llm` | PDF → Markdown conversion |
| `pymupdf-layout` | Enhanced page layout analysis for pymupdf4llm |
| `tiktoken` | Token counting for context budget management |
| `rich` | Terminal UI (panels, trees, syntax highlighting) |
| `psutil` | Memory monitoring safety threshold |
| `pydantic` | Structured LLM output for tree search results |

---

## 📌 Version

**v0.1.0** — Initial release with PDF + TXT ingestion, tree-indexed RAG, and `gemma3:1b` inference.

---

## 📜 License

MIT License
