#week 19 #Day 1 
#Transformers #Hugging face sentiment analysis

from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
print(sentiment("I love learning AI"))
print(sentiment("I hate learning AI"))



