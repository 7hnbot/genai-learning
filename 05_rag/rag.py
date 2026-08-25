import os
import chromadb

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = chroma_client.get_collection(
    name="computer_networks"
)

query = input("Ask a question: ")

results = collection.query(
    query_texts=[query],
    n_results=2
)

distances = results["distances"][0]
best_distance = distances[0]
if best_distance > 0.8:
    print("I don't have enough information to answer this question.")
    exit()

retrieved_documents = results["documents"][0]
retrieved_metadata = results["metadatas"][0]
context_parts = []

for document, metadata in zip(
    retrieved_documents,
    retrieved_metadata
):
    context_parts.append(
        f"[Source: {metadata['source']} | Topic: {metadata['topic']}]\n"
        f"{document}"
    )

context = "\n\n".join(context_parts)

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

If the context does not contain enough information,
say that you don't have enough information.
"""

response = client.chat.completions.create(
    model="google/gemini-3.7-flash",
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

for metadata in retrieved_metadata:
    print(
        f"- {metadata['source']} "
        f"({metadata['topic']})"
    )