import os
import chromadb

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DISTANCE_THRESHOLD = 0.8

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

query = input("Ask a question: ")

results = collection.query(
    query_texts=[query],
    n_results=3
)

distances = results["distances"][0]
best_distance = distances[0]
if best_distance > DISTANCE_THRESHOLD:
    print("I don't have enough information to answer this question.")
    exit()

retrieved_documents = results["documents"][0]
retrieved_metadata = results["metadatas"][0]

context_parts = []
filtered_metadata = []

for document, metadata, distance in zip(
    retrieved_documents,
    retrieved_metadata,
    distances
):
    if distance <= DISTANCE_THRESHOLD:
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