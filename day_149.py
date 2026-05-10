#week 25 day 1 
#Loading real docs

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load file
with open("medical_notes.txt", "r") as file:
    text = file.read()

#Chunking

documents = text.split()
chunk_size = 50
chunks = []

for i in range(0, len(documents), chunk_size):
    chunk = " ".join(documents[i:i+chunk_size])

#Remove empty lines
documents = [doc.strip() for doc in documents if doc.strip()]

#Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(documents)

#Query
query = input("\nEnter your query:")
query_emb = model.encode([query])

scores = cosine_similarity(query_emb, doc_emb)[0]
top_indices = np.argsort(scores)[::-1][:3]

print("\nTop results:")
for idx in top_indices:
    print(documents[idx])
    print(f"Score: {scores[idx]:.3f}")

