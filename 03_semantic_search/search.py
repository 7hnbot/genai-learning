from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "TCP is a connection-oriented protocol that provides reliable delivery.",
    "UDP is a connectionless protocol with low overhead.",
    "DNS translates domain names into IP addresses.",
    "HTTP is used to transfer resources over the web.",
    "TCP uses acknowledgements and retransmission to provide reliability."
]

query = "How does TCP provide reliable communication?"

document_embeddings = model.encode(documents)
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, document_embeddings)[0]

top_k = 2

top_results = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)[:top_k]

print("\nTop results:\n")

for document, score in top_results:
    print(f"{score.item():.4f} - {document}")