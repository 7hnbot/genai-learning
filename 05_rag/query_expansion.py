import os
import chromadb
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
def expand_query(query):

    prompt = f"""
    Generate exactly 3 alternative questions for the question below.

    Rules:
    - Reply only in English(US)
    - Keep the same meaning.
    - Use different wording or terminology.
    - Each query must be useful for searching a technical document.
    - Return exactly 3 queries.
    - Put one query on each line.
    - Do not include explanations, numbering, or bullet points.

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL"),
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    expanded_text = response.choices[0].message.content
    if not expanded_text:
        return []
    
    expanded_queries = [
        line.strip()
        for line in expanded_text.split("\n")
        if line.strip()
    ]

    return expanded_queries

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

expanded_queries = expand_query(query)

all_queries = [query] + expanded_queries

print("\nAll search queries:")

for search_query in all_queries:
    print("-", search_query)

all_results = []

for search_query in all_queries:

    results = collection.query(
        query_texts=[search_query],
        n_results=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        all_results.append(
            (document, metadata, distance)
        )

unique_results = []
best_results = {}

for document, metadata, distance in all_results:

    if (
        document not in best_results
        or distance < best_results[document][2]
    ):
        best_results[document] = ( document, metadata, distance )

unique_results = list(best_results.values())
unique_results.sort(
    key=lambda x: x[2]
)

print(
    f"\nTotal candidates before deduplication: "
    f"{len(all_results)}"
)

print(
    f"Unique candidates after deduplication: "
    f"{len(unique_results)}"
)

print("\nUnique candidates:")

for document, metadata, distance in unique_results:
    print(
        f"Distance: {distance:.4f} | "
        f"Page: {metadata['page']}"
    )

pairs = [
    [query, document]
    for document, metadata, distance in unique_results
]

rerank_scores = reranker.predict(pairs)
reranked_results = []

for score, (document, metadata, distance) in zip(
    rerank_scores,
    unique_results
):
    reranked_results.append(
        (score, document, metadata, distance)
    )

reranked_results.sort(
    key=lambda x: x[0],
    reverse=True
)

reranked_results = reranked_results[:3]
print("\nReranked results:")

for score, document, metadata, distance in reranked_results:
    print(
        f"Score: {score:.4f} | "
        f"Distance: {distance:.4f} | "
        f"Page: {metadata['page']}"
    )

context_parts = []
final_metadata = []

for score, document, metadata, distance in reranked_results:

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

for metadata in final_metadata:
    print(
        f"- {metadata['source']} "
        f"(Page {metadata['page']})"
    )