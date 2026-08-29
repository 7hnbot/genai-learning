# GenAI Learning

Hands-on learning journey into Generative AI and AI Engineering.

## Current Project

Built a PDF-based RAG chatbot that can retrieve relevant
information from a large document and generate grounded
answers using Gemini through OpenRouter.

The current implementation uses a Computer Networks PDF as the knowledge source.

## Setup

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies
4. Create a `.env` file
5. Add your OpenRouter API key (or others):

- OPENROUTER_API_KEY=your_api_key_here
- OPENROUTER_MODEL=your_preferred_model

6. Place your PDF inside `06_document_ingestion/`
7. Run the ingestion script once
8. Run the RAG script to ask questions

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
- Added source metadata and attribution
- Added fallback behavior for insufficient information
- Connected the RAG pipeline to Gemini through OpenRouter
- Added cross-encoder reranking to improve retrieval relevance
- Added reranker-based relevance filtering for insufficient-information detection
- Evaluated retrieval using Recall@K, Precision@K, and MRR

### 06 — Document Ingestion
- Built a document ingestion pipeline
- Added PDF text extraction using PyPDF
- Implemented PDF text cleaning
- Implemented sentence-aware chunking
- Added sentence overlap to preserve context
- Stored document chunks and embeddings in ChromaDB
- Preserved source, page, and chunk metadata
- Tested different chunk sizes for retrieval quality
- Tested ingestion and retrieval on a 3,000+ chunk PDF dataset

## Retrieval Evaluation

The retrieval pipeline was evaluated using Recall@K, Precision@K, and MRR.

The evaluation compares:

- Vector search
- Cross-encoder reranked search

Current evaluation uses 10 test questions from the Computer Networks PDF.

### Overall Results

| Metric | Vector Search | Reranked Search |
|---|---:|---:|
| Recall@1 | 0.4379 | 0.5409 |
| Recall@3 | 0.6970 | 0.8409 |
| Recall@5 | 0.7939 | 0.8591 |
| Precision@1 | 0.8182 | 0.9091 |
| Precision@3 | 0.6364 | 0.7576 |
| Precision@5 | 0.4727 | 0.5091 |
| MRR | 0.9091 | 0.9545 |

Reranking improved Recall@1, Recall@3, Recall@5, Precision@1, Precision@3, Precision@5, and MRR on the current evaluation dataset. Results are based on a manually created evaluation set and can be further improved by adding more diverse test questions.

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

LLMs → Embeddings → Semantic Search → Vector Database → RAG → Tool Calling → AI Agents → MCP → Deployment
