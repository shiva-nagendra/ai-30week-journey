#week 21 day 3
#Semantic search engine

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Knowledge base (documents)
sentences = [
    "Artificial intelligence is transforming healthcare.",
    "Machine learning helps analyze large datasets",
    "Space rockets travel beyond earth's atmosphere.",
    "Deep learning improves computer vision system",
    "Cooking pasta requires boiling water"
]

#convert dox to embeddings
doc_emb = model.encode(sentences)

#User querry
querry = input("\nEnter your querry: ")

#querry embeddings
que_emb = model.encode([querry])

#compute similarity
score = cosine_similarity(que_emb, doc_emb)[0]

#find best match
top_index = np.argsort(score)[::-1][:3]
print("\nTop Matches:\n")

for idx in top_index:
    print(f"{sentences[idx]} (scores:{score[idx]:.3f})")



