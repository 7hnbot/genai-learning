import os
import chromadb
from sentence_transformers import CrossEncoder
from evaluation_questions import evaluation_questions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
def expand_query(query):

    prompt = f"""
    Generate exactly 3 alternative search queries for the question below.

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
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    if not response.choices:
        print("Query expansion returned no choices.")
        return []
    expanded_text = response.choices[0].message.content
    if not expanded_text:
        return []
    
    expanded_queries = [
        line.strip()
        for line in expanded_text.split("\n")
        if line.strip()
    ]

    return expanded_queries

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

all_metrics = []

for item in evaluation_questions:

    question = item["question"]
    expected_pages = item["expected_pages"]

    # Generate alternative queries
    expanded_queries = expand_query(question)

    # Original question + expanded queries
    all_queries = [question] + expanded_queries

    # Store candidates from all searches
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

    # Deduplicate documents and keep best distance
    best_results = {}

    for document, metadata, distance in all_results:

        if (
            document not in best_results
            or distance < best_results[document][2]
        ):
            best_results[document] = (
                document,
                metadata,
                distance
            )

    unique_results = list(best_results.values())

    # Create pairs for reranking
    pairs = [
        [question, document]
        for document, metadata, distance in unique_results
    ]

    # Cross-encoder scores
    rerank_scores = reranker.predict(pairs)

    reranked_results = []

    for score, (document, metadata, distance) in zip(
        rerank_scores,
        unique_results
    ):
        reranked_results.append(
            (score, document, metadata, distance)
        )

    # Sort by relevance score
    reranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Keep top 5 for evaluation
    reranked_results = reranked_results[:5]

    reranked_pages = [
        metadata["page"]
        for score, document, metadata, distance
        in reranked_results
    ]

    metrics = evaluate_question(
        reranked_pages,
        expected_pages
    )

    all_metrics.append(metrics)

    # Print individual result
    print("\n" + "=" * 50)

    print("\nQuestion:")
    print(question)

    print("\nExpanded queries:")
    for expanded_query in expanded_queries:
        print("-", expanded_query)

    print("\nExpected pages:")
    print(expected_pages)

    print("\nRetrieved pages after expansion + reranking:")
    print(reranked_pages)

    print("\nMetrics:")
    print(metrics)

print("\n" + "=" * 50)
print("===== QUERY EXPANSION + RERANKING OVERALL =====")

metric_names = [
    "recall@1",
    "recall@3",
    "recall@5",
    "precision@1",
    "precision@3",
    "precision@5",
    "rr"
]

overall_metrics = {}

for metric in metric_names:

    overall_metrics[metric] = (
        sum(result[metric] for result in all_metrics)
        / len(all_metrics)
    )

for metric, value in overall_metrics.items():

    if metric == "rr":
        print(f"MRR: {value:.4f}")
    else:
        print(f"{metric}: {value:.4f}")