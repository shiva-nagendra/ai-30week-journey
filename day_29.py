#API → process → file

import os
import requests

headers = {
    "Accept" : "application/vnd.github+json",
    "User-Agent" : "ai-30week-journey",
    
}

response = requests.get(
    "https://api.github.com/users/shiva-nagendra/repos", #repos → list of dictionaries (raw data)
    headers=headers,
    params={"per_page": 5}
)

if response.status_code != 200:
    print("invalid API:", response.status_code)
    exit()

repos = response.json()

clean_data = []  #This step is feature selection in ML terms.
for repo in repos:
    clean_data.append({
        "name":repo["name"],
        "stars":repo["stargazers_count"],
        "language":repo["language"]
    })

import csv   #Save as CSV 

with open("repos_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["name", "stars", "language"]
    )
    writer.writeheader()
    writer.writerows(clean_data)

print("day 29 pipeline conplete")




