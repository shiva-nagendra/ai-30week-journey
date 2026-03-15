#week 19 day 3
#Image captioning with transformers

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO


#load image captioning pipeline
captioner = pipeline("image-to-text")

url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(BytesIO(requests.get(url).content))

#generate caption
result = captioner(image)

print("Caption:",result[0]["generated_text"])