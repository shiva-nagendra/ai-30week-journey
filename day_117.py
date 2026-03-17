#weeek 19 day 5
# Sentence embeddings and semantic similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences
sentences = [
    "I love learning AI",
    "Artificial intelligence is fascinating",
    "The weather is very hot today"
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Compute similarity
similarity = cosine_similarity([embeddings[0]], embeddings)

print("Similarity scores with first sentence:\n")