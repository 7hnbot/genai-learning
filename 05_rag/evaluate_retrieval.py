import chromadb
from sentence_transformers import CrossEncoder
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

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = client.get_collection(
    name="computer_networks_pdf"
)

all_vector_metrics = []
all_reranked_metrics = []

for item in evaluation_questions:

    results = collection.query(
        query_texts=[item["question"]],
        n_results=10
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    # =========================
    # VECTOR SEARCH RESULTS
    # =========================

    vector_pages = [
        metadata["page"]
        for metadata in retrieved_metadata[:5]
    ]

    vector_metrics = evaluate_question(
        vector_pages,
        item["expected_pages"]
    )

    # =========================
    # RERANKING
    # =========================

    pairs = [
        [item["question"], document]
        for document in retrieved_documents
    ]

    rerank_scores = reranker.predict(pairs)

    reranked_results = list(
        zip(
            rerank_scores,
            retrieved_metadata
        )
    )

    reranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    reranked_results = reranked_results[:5]

    reranked_pages = [
        metadata["page"]
        for score, metadata in reranked_results
    ]

    reranked_metrics = evaluate_question(
        reranked_pages,
        item["expected_pages"]
    )

    all_vector_metrics.append(vector_metrics)
    all_reranked_metrics.append(reranked_metrics)

    # =========================
    # OUTPUT
    # =========================

    print("\n" + "=" * 50)

    print("\nQuestion:")
    print(item["question"])

    print("\nExpected pages:")
    print(item["expected_pages"])


    print("\n--- Vector Search ---")

    print("Retrieved pages:")
    print(vector_pages)

    print("Metrics:")
    print(vector_metrics)


    print("\n--- Reranked Search ---")

    print("Retrieved pages:")
    print(reranked_pages)

    print("Metrics:")
    print(reranked_metrics)

print("\n" + "=" * 50)
print("\n===== OVERALL RESULTS =====")


def calculate_average(metrics_list):
    averages = {}

    for key in metrics_list[0]:
        averages[key] = sum(
            metrics[key]
            for metrics in metrics_list
        ) / len(metrics_list)

    return averages


vector_averages = calculate_average(all_vector_metrics)
reranked_averages = calculate_average(all_reranked_metrics)


print("\n--- Vector Search Overall ---")

for key, value in vector_averages.items():

    if key == "rr":
        print(f"MRR: {value:.4f}")
    else:
        print(f"{key}: {value:.4f}")


print("\n--- Reranked Search Overall ---")

for key, value in reranked_averages.items():

    if key == "rr":
        print(f"MRR: {value:.4f}")
    else:
        print(f"{key}: {value:.4f}")