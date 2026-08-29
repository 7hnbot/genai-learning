import os
import chromadb
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
RERANK_THRESHOLD = 0.0

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

query = input("Ask a question: ")

results = collection.query(
    query_texts=[query],
    n_results=10
)

distances = results["distances"][0]
retrieved_documents = results["documents"][0]
retrieved_metadata = results["metadatas"][0]

# Create question-document pairs for reranking
pairs = [
    [query, document]
    for document in retrieved_documents
]

# Get relevance scores from the cross-encoder
rerank_scores = reranker.predict(pairs)

# Combine reranking scores with documents and metadata
reranked_results = list(
    zip(
        rerank_scores,
        retrieved_documents,
        retrieved_metadata,
        distances
    )
)

# Sort by reranker score (highest first)
reranked_results.sort(
    key=lambda x: x[0],
    reverse=True
)

best_rerank_score = reranked_results[0][0]

if best_rerank_score < RERANK_THRESHOLD:
    print("I don't have enough information to answer this question.")
    exit()

# Keep the best 3 results
reranked_results = reranked_results[:3]
print("\nReranked results:")
for score, document, metadata, distance in reranked_results:
    print(
        f"Score: {score:.4f} | "
        f"Distance: {distance:.4f} | "
        f"Page: {metadata['page']}"
    )

# Build context using only the top reranked results
context_parts = []
filtered_metadata = []

for score, document, metadata, distance in reranked_results:

    context_parts.append(
        f"[Source: {metadata['source']} | Page: {metadata['page']}]\n"
        f"{document}"
    )

    filtered_metadata.append(metadata)

context = "\n\n".join(context_parts)

prompt = f"""
Answer the question using only the provided context.
Answer in plain text without Markdown formatting.
Context:
{context}

Question:
{query}

If the context does not contain enough information,
say that you don't have enough information.
"""

response = client.chat.completions.create(
    model = os.getenv("OPENROUTER_MODEL"),
    max_tokens=2000,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAI:", response.choices[0].message.content)

print("\nSources:")

for metadata in filtered_metadata:
    print(
        f"- {metadata['source']} "
        f"(Page {metadata['page']})"
    )