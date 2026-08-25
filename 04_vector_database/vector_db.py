import chromadb

client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = client.get_or_create_collection(
    name="computer_networks"
)

documents = [
    "TCP is a connection-oriented protocol that provides reliable delivery.",
    "UDP is a connectionless protocol with low overhead.",
    "DNS translates domain names into IP addresses.",
    "HTTP is used to transfer resources over the web.",
    "TCP uses acknowledgements and retransmission to provide reliability."
]

collection.add(
    documents=documents,
    ids=[
        "doc1",
        "doc2",
        "doc3",
        "doc4",
        "doc5"
    ],
    metadatas=[
        {"source": "computer_networks.pdf", "topic": "TCP"},
        {"source": "computer_networks.pdf", "topic": "UDP"},
        {"source": "computer_networks.pdf", "topic": "DNS"},
        {"source": "computer_networks.pdf", "topic": "HTTP"},
        {"source": "computer_networks.pdf", "topic": "TCP"}
    ]
)

print("Documents added!")

query = "How does TCP provide reliable communication?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

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