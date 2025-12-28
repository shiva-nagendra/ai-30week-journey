import requests
import json

headers = {
    "Accept":"applicatiion/vnd.github+json",
    "User-Agent" : "ai-30week-journey"
}
response = requests.get(
    "https://api.github.com/users/shiva-nagendra/repos", #repos → list of dictionaries (raw data)
     headers=headers,
     params={"per_day":5}
)
if response.status_code != 200:
    print("API invalid:", response.status_code)
    exit()

repos = response.json()

clean_data = []
for repo in repos:
    clean_data.append({
        "name": repo["name"],
        "stars": repo["stargazers_count"],
        "language": repo["language"]
    })

with open("repos_summary.json", "w") as f:
    json.dump(clean_data, f, indent=2)