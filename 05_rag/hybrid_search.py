import os
import chromadb
import re
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
RERANK_THRESHOLD = 0.0

def tokenize(text):
    words = re.findall(r"\b\w+\b", text.lower())

    return [
        word
        for word in words
        if word not in stop_words
    ]

def keyword_search(query, top_k=5):

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:
        results.append(
            (
                scores[index],
                ids[index],
                documents[index],
                metadatas[index]
            )
        )

    return results

def vector_search(query, top_k=5):

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    search_results = []

    for document, metadata, distance, doc_id in zip( documents, metadatas, distances, ids ):
        search_results.append(( distance, doc_id, document, metadata ))

    return search_results

def hybrid_search(query, top_k=5, k=60):

    keyword_results = keyword_search(query, top_k=top_k)
    vector_results = vector_search(query, top_k=top_k)

    rrf_scores = {}

    # Add BM25 rankings
    for rank, (score, document_id, document, metadata) in enumerate(
        keyword_results,
        start=1
    ):
        if document_id not in rrf_scores:
            rrf_scores[document_id] = {
                "score": 0,
                "document": document,
                "metadata": metadata
            }

        rrf_scores[document_id]["score"] += 1 / (k + rank)

    # Add vector search rankings
    for rank, (distance, document_id, document, metadata) in enumerate(
        vector_results,
        start=1
    ):
        if document_id not in rrf_scores:
            rrf_scores[document_id] = {
                "score": 0,
                "document": document,
                "metadata": metadata
            }

        rrf_scores[document_id]["score"] += 1 / (k + rank)

    hybrid_results = []

    for document_id, data in rrf_scores.items():
        hybrid_results.append(
            (
                data["score"],
                document_id,
                data["document"],
                data["metadata"]
            )
        )

    hybrid_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return hybrid_results[:top_k]

def rerank_results(query, hybrid_results, top_k=3):

    # Create query-document pairs for the cross-encoder
    pairs = [
        [query, document]
        for score, document_id, document, metadata
        in hybrid_results
    ]

    # Get relevance scores
    rerank_scores = reranker.predict(pairs)

    # Combine scores with original hybrid results
    reranked_results = []

    for rerank_score, (
        rrf_score,
        document_id,
        document,
        metadata
    ) in zip(rerank_scores, hybrid_results):

        reranked_results.append(
            (
                rerank_score,
                rrf_score,
                document_id,
                document,
                metadata
            )
        )

    # Sort by cross-encoder relevance score
    reranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return reranked_results[:top_k]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = chroma_client.get_collection(
    name="computer_networks_pdf"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Collection loaded successfully!")

data = collection.get(
    include=["documents", "metadatas"]
)

documents = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

print(f"Total documents loaded: {len(documents)}")

stop_words = set(stopwords.words("english"))

# Tokenize documents for BM25 keyword search
tokenized_documents = [
    tokenize(document)
    for document in documents
]

# Create the BM25 index
bm25 = BM25Okapi(tokenized_documents)

print("BM25 index created successfully!")

query = input("\nAsk a question: ")

keyword_results = keyword_search(query)

print("\nBM25 Keyword Search Results:")

for score, document_id, document, metadata in keyword_results:
    print(
        f"Score: {score:.4f} | "
        f"ID: {document_id} | "
        f"Page: {metadata['page']}"
    )

vector_results = vector_search(query)

print("\nVector Search Results:")

for distance, document_id, document, metadata in vector_results:
    print(
        f"Distance: {distance:.4f} | "
        f"ID: {document_id} | "
        f"Page: {metadata['page']}"
    )

hybrid_results = hybrid_search(query)

reranked_results = rerank_results(
    query,
    hybrid_results,
    top_k=3
)
best_rerank_score = reranked_results[0][0]

print("\nHybrid Search Results:")

for score, document_id, document, metadata in hybrid_results:
    print(
        f"RRF Score: {score:.6f} | "
        f"Page: {metadata['page']}"
    )

print("\nHybrid + Reranked Results:")

for rerank_score, rrf_score, document_id, document, metadata in reranked_results:
    print(
        f"Rerank Score: {rerank_score:.4f} | "
        f"RRF Score: {rrf_score:.6f} | "
        f"Page: {metadata['page']}"
    )

if best_rerank_score < RERANK_THRESHOLD:
    print("\nAI: ",end="")
    print("I don't have enough information.")

else:
    context_parts = []
    final_metadata = []

    for rerank_score, rrf_score, document_id, document, metadata in reranked_results:

        context_parts.append(
            f"[Source: {metadata['source']} | "
            f"Page: {metadata['page']}]\n"
            f"{document}"
        )

        final_metadata.append(metadata)

    context = "\n\n".join(context_parts)

    prompt = f"""
    Answer the question using only the provided context.

    Answer in plain text without Markdown formatting.

    Context:
    {context}

    Question:
    {query}

    If the context does not contain enough information to answer the question,
    say exactly:

    I don't have enough information.
    """

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL"),
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    print("\nAI:")
    print(answer)

    print("\nSources:")
    seen_sources = set()

    for metadata in final_metadata:
        source = (
            metadata["source"],
            metadata["page"]
        )
        if source not in seen_sources:
            print(
                f"- {metadata['source']} "
                f"(Page {metadata['page']})"
            )
            seen_sources.add(source)