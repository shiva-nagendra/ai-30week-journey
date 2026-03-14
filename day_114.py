#week 19 day 2
#text summerizer - transformers

from transformers import pipeline

summerizer = pipeline("Summerization")

text = """
Artificial intellingence is transforming many industries including healthcare,
finance, transportation, and education. Machine learning models are helping
companies analyze massive datasets and make better decisions. AI systems
can detect patterns in data that humans might miss, enabling businesses
to optimize operations and improve efficiency.
"""