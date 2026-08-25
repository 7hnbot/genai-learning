from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love dogs",
    "I really like puppies",
    "I hate rainy weather"
]

embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print(sentence)
    print("Vector size:", embedding.shape)
    print("First 5 values:", embedding[:5])
    print()
    
similarity_1 = util.cos_sim(embeddings[0], embeddings[1])
similarity_2 = util.cos_sim(embeddings[0], embeddings[2])

print("Dog vs Puppies:", similarity_1.item())
print("Dog vs Rainy Weather:", similarity_2.item())