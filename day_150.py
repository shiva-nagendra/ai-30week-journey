#week 25 day 2
#smart chunking with overlap

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load file
with open("medical_notes.txt", "r") as file:
    text = file.read()

words = text.split()

#Chunk settings
chunk_size = 12
overlap = 4

chunks = []

for i in range(0, len(words), chunk_size-overlap):

    chunk = words[i:i+chunk_size]

    chunk_text = " ".join(chunk)

    chunks.append(chunk_text)

    print("\nChunks:")

for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx+1}: \n{chunk}\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_emb = model.encode(chunks)

#Query
query = input("Enter your query: ")
query_emb = model.encode([query])

scores = cosine_similarity(query_emb, doc_emb)[0]
top_indices = np.argsort(scores)[::-1][:3]

