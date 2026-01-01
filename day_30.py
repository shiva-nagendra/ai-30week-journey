#API PRACTICE

import requests
import json

headers = {
    "Accept":"applicatiion/vnd.github+json",
    "User-Agent" : "ai-30week-journey"
}
response = requests.get(
    "https://api.github.com/users/shiva-nagendra/repos", 
     headers=headers,
     params={"per_page":5}
)
if response.status_code != 200:
    print("API invalid:", response.status_code)
    exit()

repos = response.json()
clean_data=[]
for f in repos:
    clean_data.append({
       "name": f["name"],
       "owner": f["owner"]
   })
    
with open("jsonuser.data.json","w",newline="") as j:
    json.dump(clean_data, j, indent=2)








