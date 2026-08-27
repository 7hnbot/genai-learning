import chromadb
import re
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize

def clean_text(text):
    # Remove soft hyphens inserted by PDFs
    text = re.sub(r"\u00ad\s*", "", text)

    # Fix words broken across lines with a hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace newlines with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def chunk_text(text, chunk_size=1000, overlap_sentences=1):
    sentences = sent_tokenize(text)

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:

        if (
            current_length + len(sentence) > chunk_size
            and current_sentences
        ):
            chunks.append(" ".join(current_sentences))

            current_sentences = current_sentences[
                -overlap_sentences:
            ]

            current_length = sum(
                len(s) for s in current_sentences
            )

        current_sentences.append(sentence)
        current_length += len(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

pdf_path = "06_document_ingestion/computer_networks.pdf"

reader = PdfReader(pdf_path)

all_chunks = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    text = clean_text(text)
    chunks = chunk_text(text)

    for chunk_number, chunk in enumerate(chunks):

        all_chunks.append({
            "text": chunk,
            "page": page_number,
            "chunk_id": chunk_number
        })

chroma_client = chromadb.PersistentClient(
    path="./04_vector_database/chroma_db"
)

try:
    chroma_client.delete_collection("computer_networks_pdf")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name="computer_networks_pdf"
)

collection.add(
    documents=[chunk["text"] for chunk in all_chunks],
    ids=[
        f"page_{chunk['page']}_chunk_{chunk['chunk_id']}"
        for chunk in all_chunks
    ],
    metadatas=[
        {
            "source": "computer_networks.pdf",
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in all_chunks
    ]
)

"""query = "How does TCP provide reliable communication?"

results = collection.query(
    query_texts=[query],
    n_results=3
)

print("\nTop results:\n")

for document, distance, metadata in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"Distance: {distance:.4f}")
    print(f"Source: {metadata['source']}")
    print(f"Page: {metadata['page']}")
    print(f"Chunk ID: {metadata['chunk_id']}")
    print(f"Document: {document}")
    print()"""