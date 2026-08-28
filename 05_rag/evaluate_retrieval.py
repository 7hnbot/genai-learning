import chromadb
from evaluation_questions import evaluation_questions

def recall_at_k(retrieved_pages, expected_pages, k):
    retrieved = set(retrieved_pages[:k])
    expected = set(expected_pages)

    if not expected:
        return 0.0

    return len(retrieved & expected) / len(expected)

def precision_at_k(retrieved_pages, expected_pages, k):
    retrieved = retrieved_pages[:k]
    expected = set(expected_pages)

    if k == 0:
        return 0.0

    relevant_count = sum(
        chunk in expected
        for chunk in retrieved
    )

    return relevant_count / k

def reciprocal_rank(retrieved_pages, expected_pages):
    expected = set(expected_pages)

    for rank, chunk in enumerate(retrieved_pages, start=1):
        if chunk in expected:
            return 1 / rank

    return 0.0

def evaluate_question(retrieved_pages, expected_pages):
    return {
        "recall@1": recall_at_k(retrieved_pages, expected_pages, 1),
        "recall@3": recall_at_k(retrieved_pages, expected_pages, 3),
        "recall@5": recall_at_k(retrieved_pages, expected_pages, 5),
        "precision@1": precision_at_k(retrieved_pages, expected_pages, 1),
        "precision@3": precision_at_k(retrieved_pages, expected_pages, 3),
        "precision@5": precision_at_k(retrieved_pages, expected_pages, 5),
        "rr": reciprocal_rank(retrieved_pages, expected_pages)
    }

client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = client.get_collection(
    name="computer_networks_pdf"
)

for item in evaluation_questions:

    results = collection.query(
        query_texts=[item["question"]],
        n_results=3
    )

    retrieved_pages = [
        metadata["page"]
        for metadata in results["metadatas"][0]
    ]

    metrics = evaluate_question(
        retrieved_pages,
        item["expected_pages"]
    )

    print("\nQuestion:")
    print(item["question"])

    print("Expected pages:")
    print(item["expected_pages"])

    print("Retrieved pages:")
    print(retrieved_pages)

    print("Metrics:")
    print(metrics)