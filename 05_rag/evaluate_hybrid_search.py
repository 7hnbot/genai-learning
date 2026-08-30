import chromadb
from rank_bm25 import BM25Okapi
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
        page in expected
        for page in retrieved
    )

    return relevant_count / k


def reciprocal_rank(retrieved_pages, expected_pages):
    expected = set(expected_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
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

def keyword_search(query, top_k=10):

    tokenized_query = query.lower().split()

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
                document_ids[index],
                documents[index],
                metadatas[index]
            )
        )

    return results

def vector_search(query, top_k=10):

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    result_ids = results["ids"][0]
    result_documents = results["documents"][0]
    result_metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    search_results = []

    for distance, document_id, document, metadata in zip(
        distances,
        result_ids,
        result_documents,
        result_metadatas
    ):
        search_results.append(
            (
                distance,
                document_id,
                document,
                metadata
            )
        )

    return search_results

def hybrid_search(query, top_k=10, k=60):

    keyword_results = keyword_search(
        query,
        top_k=top_k
    )

    vector_results = vector_search(
        query,
        top_k=top_k
    )

    rrf_scores = {}

    # BM25 contribution
    for rank, (
        score,
        document_id,
        document,
        metadata
    ) in enumerate(keyword_results, start=1):

        if document_id not in rrf_scores:
            rrf_scores[document_id] = {
                "score": 0,
                "document": document,
                "metadata": metadata
            }

        rrf_scores[document_id]["score"] += (
            1 / (k + rank)
        )

    # Vector contribution
    for rank, (
        distance,
        document_id,
        document,
        metadata
    ) in enumerate(vector_results, start=1):

        if document_id not in rrf_scores:
            rrf_scores[document_id] = {
                "score": 0,
                "document": document,
                "metadata": metadata
            }

        rrf_scores[document_id]["score"] += (
            1 / (k + rank)
        )

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

def rerank_results(query, hybrid_results, top_k=10):

    pairs = [
        [query, document]
        for score, document_id, document, metadata
        in hybrid_results
    ]

    rerank_scores = reranker.predict(pairs)

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

    reranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return reranked_results[:top_k]

client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

collection = client.get_collection(
    name="computer_networks_pdf"
)

data = collection.get(
    include=["documents", "metadatas"]
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

documents = data["documents"]
metadatas = data["metadatas"]
document_ids = data["ids"]

print("Collection loaded successfully!")
print(f"Total documents loaded: {len(documents)}")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)
print("BM25 index created successfully!")

all_metrics = []

for item in evaluation_questions:

    question = item["question"]
    expected_pages = item["expected_pages"]

    print("\n" + "=" * 50)
    print("\nQuestion:")
    print(question)

    print("\nExpected pages:")
    print(expected_pages)

    # Hybrid retrieval
    hybrid_results = hybrid_search(
        question,
        top_k=10
    )

    # Reranking
    reranked_results = rerank_results(
        question,
        hybrid_results,
        top_k=10
    )

    # Extract retrieved pages
    retrieved_pages = []
    for result in reranked_results:
        page = result[4]["page"]
        if page not in retrieved_pages:
            retrieved_pages.append(page)

    retrieved_pages = retrieved_pages[:5]

    print("\nRetrieved pages after hybrid + reranking:")
    print(retrieved_pages)

    # Calculate metrics
    metrics = evaluate_question(
        retrieved_pages,
        expected_pages
    )

    print("\nMetrics:")
    print(metrics)

    all_metrics.append(metrics)

print("\n" + "=" * 50)
print("===== HYBRID + RERANKING OVERALL RESULTS =====")

metric_names = all_metrics[0].keys()

overall_metrics = {}

for metric in metric_names:

    average = sum(
        result[metric]
        for result in all_metrics
    ) / len(all_metrics)

    overall_metrics[metric] = average

for metric, value in overall_metrics.items():
    print(f"{metric}: {value:.4f}")