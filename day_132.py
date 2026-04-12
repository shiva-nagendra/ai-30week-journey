#week 22 day 2
#FAISS implementation

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Data
sentences = [
    "AI is transforming healthcare",
    "Machine learning analyzes data",
    "Deep learning improves vision",
    "Space rockets explore planets",
    "Healthy food improves life",
    "Exercise keeps the body fit",
    "Solar energy is renewable",
    "Wind power generates electricity"
    "Macbook is a great laptop",
    "Mallareddy college is on top",
    "Space exploration expands human knowledge",
    "Rockets are used to send satellites into orbit",
    "Astronomy studies celestial objects and space",
    "Satellites help in global communication",
    "Mars missions are a focus of space agencies",
]

embeddings = model.encode(sentences)

#convert to float32(FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

#create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

#add vectors to index
index.add(embeddings)

#Querry
query = input("\nEnter your querry:")
query_emb = model.encode([query]).astype("float32")

#search
k = 3
distances, indices = index.search(query_emb, k)

print("\nTop matches")

for i, idx in enumerate(indices[0]):
    print(f"{sentences[idx]} (distance: {distances[0][i]:.3f})")

