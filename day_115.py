#week 19 day 3
#Image captioning with transformers

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO


#load image captioning pipeline
captioner = pipeline("image-to-text")

url = "https://hips.hearstapps.com/hmg-prod/images/close-up-portrait-of-cat-sitting-on-floor-royalty-free-image-1718201739.jpg?crop=0.668xw:1.00xh;0.180xw,0"
image = Image.open(BytesIO(requests.get(url).content))

#generate caption
result = captioner(image)

print("Caption:",result[0]["generated_text"])