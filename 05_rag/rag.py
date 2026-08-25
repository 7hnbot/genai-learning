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

retrieved_documents = results["documents"][0]

context = "\n\n".join(retrieved_documents)

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