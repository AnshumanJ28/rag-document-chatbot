# RAG Document Intelligence Chatbot

A production-style Retrieval-Augmented Generation (RAG) pipeline built in Google Colab. Upload any PDF and ask questions — the system retrieves the most relevant chunks using MMR + cross-encoder re-ranking, then generates grounded answers using Groq's Llama 3.1. All queries are tracked with MLflow.

> Built as part of an ML Engineering portfolio. Focuses on retrieval quality, hallucination prevention, and experiment tracking — not just "plug in an LLM and call it a day."

---

## Demo

![Gradio UI](assets/demo_screenshot.png)

> *Left: conversational interface. Right: live source citations with rerank scores per query.*

---

## Results

All 5 test queries logged via MLflow:

| Question | Latency | Top Rerank Score | Answer Length |
|---|---|---|---|
| What is the capital of Japan? | 0.59s | -11.06 | 73 chars |
| How does chunking strategy affect retrieval? | 0.64s | 7.21 | 354 chars |
| What is MMR and why is it used in retrieval? | 0.69s | -1.10 | 254 chars |
| Explain the difference between FAISS index types? | 0.77s | -1.68 | 301 chars |
| What evaluation metrics are used for RAG systems? | 0.81s | 7.85 | 183 chars |

**Key observation:** The Japan question (not in the document) returns a top rerank score of -11 and correctly triggers the hallucination guard. On-topic questions score 7+ and receive accurate, cited answers. The re-ranker's score distribution is the signal — not just retrieval count.

Full run logs: [`results/mlflow_runs.csv`](results/mlflow_runs.csv)

---

## Architecture

```
PDF Input
    │
    ▼
PDF Ingestion (PyMuPDF)
    │  noise removal, page extraction
    ▼
Chunking (RecursiveCharacterTextSplitter)
    │  chunk_size=512, overlap=64
    ▼
Embedding (sentence-transformers/all-MiniLM-L6-v2)
    │  384-dim dense vectors
    ▼
FAISS Vector Store
    │  saved to disk, reloadable
    ▼
MMR Retrieval  ──────────────────────────────────┐
    │  fetch_k=10, lambda=0.7                    │
    ▼                                            │
Cross-Encoder Re-ranking                         │
    │  ms-marco-MiniLM-L-6-v2, keep top 4       │
    ▼                                            │
LLM Generation (Groq / Llama 3.1-8b)            │
    │  grounded prompt, no outside knowledge     │
    ▼                                            │
Answer + Source Citations ◄───────────────────────┘
    │
    ▼
MLflow Logging
    │  latency, rerank scores, answer length, config params
    ▼
Gradio UI
    │  chat interface + live source panel
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS (faiss-cpu) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.1-8b-instant via Groq API |
| Orchestration | LangChain |
| Experiment tracking | MLflow |
| UI | Gradio |
| Runtime | Google Colab (T4 GPU, free tier) |

---

## Repo Structure

```
rag-document-chatbot/
├── notebook/
│   └── rag_chatbot.ipynb       # full pipeline, run top to bottom
├── results/
│   └── mlflow_runs.csv         # logged metrics from all test queries
├── sample_data/
│   └── sample_rag_test.pdf     # sample PDF used for testing
├── assets/
│   └── demo_screenshot.png     # Gradio UI screenshot
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/AnshumanJ28/rag-document-chatbot
cd rag-document-chatbot
```

### 2. Open in Colab
Click the badge or upload `notebook/rag_chatbot.ipynb` to [colab.research.google.com](https://colab.research.google.com).

### 3. Add secrets
In Colab, click the key icon (left sidebar) and add:

| Secret name | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free |
| `GITHUB_TOKEN` | GitHub → Settings → Developer settings → Tokens |
| `NGROK_AUTHTOKEN` | [ngrok.com](https://ngrok.com) — free (optional) |

### 4. Run all cells top to bottom
- Cell 6 generates a sample PDF automatically so you can test immediately
- Cell 7 lets you upload your own PDF instead
- The Gradio UI cell gives you a public shareable link

---

## Pipeline Design Decisions

**Why MMR before re-ranking?**
Pure similarity search returns redundant chunks — all saying the same thing from nearby pages. MMR enforces diversity at fetch time (top 10), then cross-encoder re-ranking re-scores those 10 diverse candidates for precision, keeping the best 4. Two-stage retrieval: diversity first, precision second.

**Why cross-encoder re-ranking?**
Bi-encoders (used for embedding + FAISS search) encode query and document independently — fast but less accurate for ranking. Cross-encoders take the query-document pair as joint input, giving finer-grained relevance scores. Too slow for first-stage retrieval over thousands of chunks, ideal for re-ranking a small candidate set.

**Why the hallucination guard matters?**
The prompt explicitly instructs the LLM to answer only from the provided context. The Japan question (not in the PDF) returns a rerank score of -11 across all chunks — signalling nothing relevant was retrieved — and the model correctly refuses to answer. The rerank score is a proxy for "should I trust this answer."

**Why MLflow?**
Latency and rerank scores vary per query and per config. MLflow makes it easy to compare chunk sizes, top-k values, or LLM models across runs without manually tracking results — the same pattern used in production ML systems.

---

## Configuration

All key hyperparameters live in `RAGConfig` (Cell 2) — swap without touching the pipeline:

```python
@dataclass
class RAGConfig:
    chunk_size: int        = 512    # try 256 or 1024
    chunk_overlap: int     = 64     # try 128 for denser overlap
    top_k_retrieval: int   = 10     # candidates fetched by MMR
    top_k_rerank: int      = 4      # final chunks passed to LLM
    llm_model: str         = "llama-3.1-8b-instant"   # or mixtral-8x7b-32768
    temperature: float     = 0.2
```

---

## Limitations & Future Work

- **No memory across turns** — each query is independent; adding conversational memory (LangChain `ConversationBufferMemory`) would enable follow-up questions
- **Single PDF at a time** — the vector store can hold multiple PDFs but the UI currently re-builds per session; adding persistent storage (e.g. Pinecone) would fix this
- **Free-tier latency** — Groq free tier is fast but rate-limited; swap to a local Ollama model for unlimited offline inference
- **No RAGAS evaluation** — adding reference-based evaluation (faithfulness, answer relevancy) would make the MLflow dashboard more meaningful

---

## Author

**Anshuman Pandey**
B.Tech CSE (AI/ML) — VIT Bhopal
[GitHub](https://github.com/AnshumanJ28) · [LinkedIn](https://linkedin.com/in/anshuman-pandey)
