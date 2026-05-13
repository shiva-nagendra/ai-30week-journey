#week 25 day 5
#Hybrid Search (Keyword + Semantic)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#docs
documents = [

    "GPT-4 Turbo supports large context windows",

    "Machine learning improves healthcare systems",

    "Python is popular for AI engineering",

    "GPT-4 pricing depends on token usage",

    "Hospitals use AI for medical imaging"

]

#embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

#query
query = input("Ask question: ")

query_embedding = model.encode([query])

#semantic search

semantic_scores = cosine_similarity(
    query_embedding,
    doc_embeddings
)[0]

#keyword scores
query_words = query.lower().split()

keyword_scores = []

for doc in documents:

    doc_lower = doc.lower()
    score = 0

    for word in query_words:
        if word in doc_lower:
            score += 1

    keyword_scores.append(score)

keyword_scores = np.array(keyword_scores)

#normalize keyword scores
if keyword_scores.max() > 0:

    keyword_scores = keyword_scores / keyword_scores.max()

#hybrid score
final_scores = (
    semantic_scores * 0.7
    +
    keyword_scores * 0.3
)

#top scores
top_indices = np.argsort(final_scores)[::-1][:3]

