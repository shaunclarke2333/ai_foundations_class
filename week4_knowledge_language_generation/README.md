# The Knowledge-Grounded Assistant (RAG Chatbot)

**Course:** CSC6313 AI Foundations  
**Week:** 04 Retrieval-Augmented Generation  
**Author:** Shaun Clarke

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system the industry-standard technique for grounding AI responses in real, private data rather than relying solely on a model's training knowledge. The chatbot loads a local document collection, converts it into vector embeddings stored in ChromaDB, and uses semantic search to retrieve relevant context before sending an augmented prompt to a **Llama 3.1 8B** LLM via the Hugging Face Inference API.

The result is a chatbot that answers questions using *only* the documents you provide dramatically reducing hallucinations compared to a bare LLM call.

**Added Challenge completed:** The system handles `.txt`, `.pdf`, `.md`, and `.csv` files not just plain text.

---

## How RAG Works

```
User Question
      ↓
[Embed query → 384-dim vector via all-MiniLM-L6-v2]
      ↓
[Cosine similarity search ChromaDB → top 3 relevant chunks]
      ↓
[Format chunks with source headers → build context string]
      ↓
[Inject context + guardrails + question into system prompt]
      ↓
[Send to Llama 3.1-8B via Hugging Face InferenceClient]
      ↓
Answer grounded strictly in your documents
```

Without the retrieval step, the LLM would answer from its training data potentially hallucinating facts not in your documents. The retrieved context "grounds" the model to your specific knowledge base.

---

## Test Documents Used

The following files were used to build and test the knowledge base. They were chosen to cover all four supported file types and a range of content styles.

### `test1.txt` HTTP Status Codes
Descriptions of HTTP 1xx informational status codes: `100 Continue`, `101 Switching Protocols`, `102 Processing`, and `103 Early Hints`. Used to test plain text ingestion and factual recall.

**Example query:** *"What does HTTP status code 103 do?"*

---

### `test2.txt` California Pregnancy Leave Quick Reference
A detailed comparison table of three leave types **Pregnancy Disability Leave (PDL)**, **California Family Rights Act (CFRA)**, and **Family & Medical Leave Act (FMLA)**. Covers eligibility, duration, pay, job protection, and notification requirements under California and federal law.

**Example query:** *"How much PDL leave am I entitled to?"*  
**Example query:** *"Will I lose my job if I take CFRA leave?"*

---

### `Pregnancy-Disability-Leave-Fact-Sheet_ENG.pdf` California CRD Fact Sheet (2 pages)
Official fact sheet from the California Civil Rights Department covering Pregnancy Disability Leave requirements, employee obligations, salary and benefits during PDL, return rights, CFRA non-pregnancy leave, accommodations while working, and how to file a complaint. Parsed page-by-page with page numbers preserved in metadata.

**Example query:** *"Can my employer require me to use sick leave during PDL?"*  
**Example query:** *"What happens if my employer fires me while I'm on pregnancy leave?"*

---

### `pdf_test.pdf` Python First-Class Functions (2 pages)
A short technical document explaining that Python functions are first-class objects and can be passed as parameters to other functions, enabling flexible and reusable code design. Used to test PDF ingestion and multi-page chunking.

**Example query:** *"Can you pass a function as a parameter in Python?"*

---

### `computers.csv` Computer Inventory (4 rows)
A small inventory table with three columns: `computer_name`, `HDD`, and `RAM`. Each row is converted to a `column: value` formatted string before chunking. Used to test CSV ingestion and row-level metadata.

```
computer_name: mb-eng-test3 | HDD: yes | RAM: Yes
computer_name: computer2    | HDD: Yes | RAM: No
computer_name: computer3    | HDD: NA  | RAM: Yes
computer_name: computer4    | HDD: No  | RAM: Yes
```

**Example query:** *"Which computers have RAM but no HDD?"*

---

### `notes.md` Sample Data Instructions (Markdown)
A markdown file containing setup notes and example document content for ChromaDB, RAG, and embeddings. Used to test `.md` file ingestion parsed with the same `open()` reader as `.txt` files since markdown is plain text.

**Example query:** *"What is ChromaDB used for?"*

---

## Architecture

The system is built across three classes plus a `main()` function.

### `DocumentLoader`
Handles reading, parsing, and chunking all supported file types from the `documents_project4/` directory.

All parsed content is stored as a list of dictionaries using the shared `collect_file_content()` helper, which enforces a consistent shape across all file types:

```python
{
    "content": "...",   # the text
    "source":  "...",   # file path
    "page":    ...,     # PDF page number (None for txt/md/csv)
    "row":     ...,     # CSV row index   (None for txt/md/pdf)
}
```

**Per file type:**
- `.txt` / `.md` read with `open()`, returned as one document per file
- `.pdf` parsed page-by-page with `pymupdf4llm.to_markdown(page_chunks=True)`, page number captured from each page's metadata dictionary
- `.csv` read with `pandas`, each row converted to a multi-line `column: value` string; `NaN` values replaced with `"Unknown"` via `fillna()` before formatting

**Chunking (`chunk_text`):**

```
chunk_size    = 200 characters
chunk_overlap = 50 characters
step_size     = chunk_size - chunk_overlap = 150
```

The window advances by 150 chars per iteration but captures 200 chars — so the last 50 characters of each chunk reappear at the start of the next. This prevents key facts from being lost when a sentence straddles a chunk boundary. Documents shorter than `chunk_size` are returned as a single chunk.

---

### `VectorStore`
Manages ChromaDB and the sentence embedding model.

- Uses `chromadb.PersistentClient` embeddings written to `./chroma_db/` persist between runs
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- `add_documents()` encodes chunks into numpy vectors, assigns IDs (`chunk_0`, `chunk_1`, ...), strips `None` metadata fields (ChromaDB requires strings), converts numpy array to list of lists, calls `collection.add()`
- `query_db()` embeds the user's question into the same 384-dim space, queries for top-N chunks by cosine similarity, reformats ChromaDB's nested-list output into clean dictionaries

> The query is embedded explicitly with the same `SentenceTransformer` model not ChromaDB's internal embedding to guarantee the query vector lives in exactly the same embedding space as the stored chunks.

---

### `RAGChatbot`
The orchestrator that ties document loading, vector search, and LLM generation into a single pipeline.

| Method | What it does |
|---|---|
| `load_knowledge_base(directory)` | Loads and chunks all files, populates `VectorStore` |
| `query_vector_db(query)` | Retrieves top 3 chunks; formats each with `[Source: ..., Page/Row: ...]` header; joins with `---` separators |
| `generate_response(query, context)` | Builds guardrailed system prompt, calls `InferenceClient.chat_completion()`, handles API errors |
| `chatbot(query)` | Single entry point calls `query_vector_db` then `generate_response` |

---

## Prompt Engineering & Guardrails

```
You are a helpful assistant. Answer the question using ONLY the provided context below.

IMPORTANT RULES:
- If the answer is not in the provided context, say "I don't know based on the provided information"
- Do not make up information or use knowledge outside the provided context.
- Be concise and direct, do not ramble on.
- Cite the source when possible

CONTEXT:
[retrieved chunks with source/page/row headers]

QUESTION: [user's question]

ANSWER:
```

**LLM generation parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `max_tokens` | 250 | Concise responses within free tier limits |
| `temperature` | 0.7 | Balanced — not too rigid, not too random |
| `top_p` | 0.9 | Nucleus sampling — most probable token pool only |

---

## Supported File Types

| Extension | Parser | Metadata Captured |
|---|---|---|
| `.txt` | `open()` | Source filename |
| `.md` | `open()` | Source filename |
| `.pdf` | `pymupdf4llm` | Source filename, page number |
| `.csv` | `pandas` | Source filename, row index |

Files with unsupported extensions and subdirectories are skipped silently.

---

## Prerequisites

- Python 3.10+
- Free [Hugging Face](https://huggingface.co) account and API key
- Llama 3.1 license accepted at:  
  `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`

```bash
pip install chromadb sentence-transformers pandas pymupdf4llm huggingface-hub numpy
```

---

## How to Run

### 1. Create the documents directory and add your files

```bash
mkdir documents_project4
```

Place any `.txt`, `.pdf`, `.md`, or `.csv` files inside. The chatbot will only answer from these documents.

### 2. Start the chatbot

```bash
python rag_chatbot.py
```

### 3. Enter your API key and start chatting

```
============================================================
T1000 CHATBOT - Knowledge Grounded Assistant
============================================================

Welcome! This chatbot answers questions using your documents.
It also promises not to travel back in time to find Sarah Connor.

Enter your Hugging Face API key, or else: [your key]

============================================================
CHAT STARTED
============================================================
Ask questions about your documents!
Type 'quit', 'exit', or 'q' to end the conversation.

You: What does HTTP status code 103 do?
T1000: Based on the provided context (Source: test1.txt), HTTP 103 Early Hints
       is primarily intended for use with the Link header, letting the user agent
       start preloading resources while the server prepares a response.

You: Which computers have RAM but no HDD?
T1000: Based on the inventory (Source: computers.csv), computer2 has RAM but no HDD.
       computer4 also has RAM listed as Yes with HDD as No.
```

> **Security:** Your API key is entered at runtime and never written to any file. Do not hardcode it.

---

## Error Handling

| Error | Cause | Chatbot Response |
|---|---|---|
| `403` | Llama license not accepted | Directs to license acceptance URL |
| `503` | Model still loading on HF servers | Prompts to wait 30 seconds and retry |
| `429` | Free tier rate limit hit | Prompts to wait 60 seconds and retry |
| Empty input | User hit Enter without typing | Loop continues, prompts again |
| Missing directory | `documents_project4/` not found | Prints error and exits cleanly |
| Empty API key | User skipped key input | Prints required message and exits |

---

## Design Notes

**Why overlapping chunks?** Hard chunk boundaries can split a sentence mid-thought. The 50-character overlap means every key fact appears complete in at least one retrievable chunk even if it straddles a boundary — important for dense documents like the PDL fact sheet and leave comparison table.

**Why source headers in context chunks?** Formatting each chunk as `[Source: Pregnancy-Disability-Leave-Fact-Sheet_ENG.pdf, Page: 1]` before injecting it into the prompt gives the LLM clear attribution. It can then cite the source in its answer, making responses verifiable.

**Why `---` separators between chunks?** Without separators, adjacent chunks blur together and the model loses track of where one source ends and another begins especially problematic when mixing a PDF page, a CSV row, and a `.txt` excerpt in the same context string.

**Why strip `None` values before ChromaDB storage?** ChromaDB metadata requires string values. A `.txt` file has no page number passing `None` directly would raise a runtime error. Only fields that exist are added to the metadata dictionary.

**Why `NaN → "Unknown"` for CSV?** The `computers.csv` file has `NA` in one cell. Pandas reads this as `NaN` (a float). Without `fillna('Unknown')`, the row would be formatted as `HDD: nan` confusing context for the LLM. The replacement keeps every row human-readable.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `chromadb` | Local persistent vector database |
| `sentence-transformers` | Text → 384-dim vectors (`all-MiniLM-L6-v2`) |
| `huggingface-hub` | `InferenceClient` for Llama 3.1 via HF free API |
| `pymupdf4llm` | PDF parsing with per-page content and metadata |
| `pandas` | CSV ingestion with row iteration and NaN handling |
| `numpy` | Vector array format before ChromaDB list conversion |

---

## File Structure

```
week04/
├── rag_chatbot.py                              # Full implementation
├── documents_project4/                         # Knowledge base documents
│   ├── test1.txt                               # HTTP status codes
│   ├── test2.txt                               # CA pregnancy leave reference
│   ├── Pregnancy-Disability-Leave-Fact-Sheet_ENG.pdf   # CRD PDL fact sheet
│   ├── pdf_test.pdf                            # Python first-class functions
│   ├── computers.csv                           # Computer inventory
│   └── notes.md                               # RAG/ChromaDB setup notes
├── chroma_db/                                  # Auto-created — ChromaDB storage
└── README.md                                   # This file
```

> Add `chroma_db/` to `.gitignore` — it contains generated embeddings, not source code.  
> Never commit your Hugging Face API key.