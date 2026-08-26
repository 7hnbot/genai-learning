# GenAI Learning

Hands-on learning journey into Generative AI and AI Engineering.

## Current Progress

### 01 — LLM Applications
- OpenRouter API
- Gemini
- Multi-turn conversations
- System prompts
- Conversation history
- Streaming responses

### 02 — Embeddings
- Sentence Transformers
- 384-dimensional embeddings
- Semantic similarity
- Cosine similarity

### 03 — Semantic Search
- Converted documents and queries into embeddings
- Calculated cosine similarity
- Ranked documents by semantic relevance
- Implemented Top-K retrieval

### 04 — Vector Database
- Used ChromaDB for vector storage and retrieval
- Stored documents with embeddings
- Added document metadata
- Performed similarity-based Top-K retrieval
- Explored vector distance and relevance ranking

### 05 — RAG
- Connected Chroma vector retrieval with an LLM
- Retrieved relevant context before generation
- Implemented context-grounded responses
- Added similarity thresholding
- Added source metadata and attribution
- Added fallback behavior for insufficient information

### 06 — Document Ingestion
- Built a document ingestion pipeline
- Implemented sentence-based chunking
- Added chunk overlap to preserve context
- Stored chunks in ChromaDB
- Added metadata for source tracking
- Added PDF text extraction with PyPDF
- Preserved page-level metadata
- Improved chunking with sentence-aware splitting
- Added sentence overlap between chunks
- Tested retrieval quality on a 1,000+ chunk PDF dataset

## Tech Stack

- Python
- OpenRouter
- Gemini
- Sentence Transformers
- Git/GitHub

### PDF Dataset

The PDF used for testing is not included in this repository.
Place your own PDF inside `06_document_ingestion/` before running
the ingestion pipeline.

## Roadmap

LLMs
→ Embeddings → Semantic Search → Vector Database → RAG → Tool Calling → AI Agents → MCP → Deployment
