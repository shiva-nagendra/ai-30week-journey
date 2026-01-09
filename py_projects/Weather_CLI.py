#weather CLI

import requests

API_KEY = "ddb7db9161634ea7bed94503260901"

city = input("Enter your city name: \n")
url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=yes"

response = requests.get(url)
data = response.json()

temperature = data["current"]["temp_c"]
description = data["location"]


print(f"temp:{temperature}c")
print("description:", description)


