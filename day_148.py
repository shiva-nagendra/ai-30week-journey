#week 24 day 6
#Environment variables and secure configs

from dotenv import load_dotenv
import os

#load env file
load_dotenv()

#read variables
hf_token = os.getenv("HF_TOKEN")
app_mode = os.getenv("APP_MODE")

print("Hugging_face token: ")
print(hf_token)

print("\nApp_mode: ")
print(app_mode)