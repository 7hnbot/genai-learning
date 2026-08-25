import chromadb

def chunk_text(text, sentences_per_chunk=2, overlap=1):
    sentences = [
        sentence.strip()
        for sentence in text.split("\n\n")
        if sentence.strip()
    ]

    chunks = []

    step = sentences_per_chunk - overlap

    for start in range(0, len(sentences), step):
        chunk = sentences[start:start + sentences_per_chunk]

        if not chunk:
            break

        chunks.append("\n\n".join(chunk))

        if start + sentences_per_chunk >= len(sentences):
            break

    return chunks

with open("06_document_ingestion/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_text(text)

client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

try:
    client.delete_collection("computer_networks_chunks")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="computer_networks_chunks"
)

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[
        {
            "source": "sample.txt",
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]
)

query = "How does TCP handle lost packets?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print("\nIDs:")
print(results["ids"][0])

print("\nDocuments:")
print(results["documents"][0])

print("\nDistances:")
print(results["distances"][0])

print("\nMetadata:")
print(results["metadatas"][0])

print("\nTop results:\n")

for document, distance, metadata in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"Distance: {distance:.4f}")
    print(f"Document: {document}")
    print(f"Metadata: {metadata}")
    print()