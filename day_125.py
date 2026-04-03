#week 21 day 1
#Sentance embeddings

from sentence_transformers import SentenceTransformer
import numpy as np

#load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#sentances
sentances = [
    "I Love AI",
    "Artificial intelligence is amazing",
    "The weather is very hot"
]

#convert to embeddings
embeddings = model.encode(sentances)

print("\nEmbeddings shape:", embeddings.shape)

print("\nFirst sentance embedding (short view):")
print(embeddings[0][:10]) #print first 10 numbers
