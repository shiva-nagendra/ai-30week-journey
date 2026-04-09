#week 21 project
#Semantic search engine (BRUTE Force)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#dataset
sentences = [
    "Artificial intelligence is transforming healthcare",
    "Machine learning helps analyze large datasets",
    "Deep learning improves computer vision systems",
    "Natural language processing enables chatbots",
    "Neural networks mimic the human brain",
    "Data science extracts insights from data",
    "Python is widely used for AI development",
    "Big data drives modern decision making",
    "Cloud computing enables scalable applications",
    "Cybersecurity protects systems from attacks",

    "Space exploration expands human knowledge",
    "Rockets are used to send satellites into orbit",
    "Astronomy studies celestial objects and space",
    "Satellites help in global communication",
    "Mars missions are a focus of space agencies",
    "Telescopes observe distant galaxies",
    "Gravity affects objects with mass",
    "The universe is constantly expanding",
    "Black holes have strong gravitational pull",
    "The moon affects ocean tides",

    "Healthy eating improves overall wellbeing",
    "Exercise helps maintain physical fitness",
    "Running is good for heart health",
    "Yoga improves flexibility and focus",
    "Drinking water is essential for health",
    "Sleep is important for recovery",
    "Mental health is as important as physical health",
    "Balanced diet supports body functions",
    "Meditation reduces stress",
    "Walking daily improves stamina",

    "Electric cars reduce environmental pollution",
    "Solar energy is a renewable resource",
    "Wind power generates clean electricity",
    "Recycling helps reduce waste",
    "Climate change affects global temperatures",
    "Sustainable living protects nature",
    "Plastic pollution harms marine life",
    "Green energy is the future",
    "Forests play a vital role in ecosystems",
    "Water conservation is important",

    "Cooking requires proper ingredients",
    "Baking involves precise measurements",
    "Spices enhance food flavor",
    "Restaurants serve different cuisines",
    "Street food is popular in many countries",
    "Healthy cooking uses fresh ingredients",
    "Food provides energy to the body",
    "Grilling adds smoky flavor to food",
    "Fruits are rich in vitamins",
    "Vegetables are essential for nutrition"
]

#sentence emb
sen_emb = model.encode(sentences)

#Querry loop
while True:
    querry = input("\nEnter your query: (or exit)")
    if querry == "exit":
        break

    querry_emb = model.encode([querry])

    