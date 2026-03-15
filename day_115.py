#week 19 day 3
#Image captioning with transformers

from transformers import pipeline
from PIL import Image

#load image captioning pipeline
captioner = pipeline("image-to-text")

#Load an image
image = Image.open("dog.jpg")

#generate caption
