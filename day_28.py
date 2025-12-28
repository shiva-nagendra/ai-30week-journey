#API practice

import requests
params = {
    "per_page" : 5
}

response = requests.get("https://api.github.com/users/shiva-nagendra/repos",
                        params=params)

print(response.url)




#practice

import requests
params = {
    "per_page" : 6
}
respond = requests.get("https://api.github.com/users/shiva-nagendra",
                        params=params)
print(respond.url)


headers = {
    "Accept":"application/vnd.github+json",
    "User-Agent" : "ai-30week-journey"
}

response = requests.get("https://api.github.com/users/shiva-nagendra",
                        headers=headers)

print(response.json()["bio"])


headers = {
    "Accept":"application/vnd.github+json",
    "User-Agent" : "ai-30week-journey"
}

response = requests.get("https://api.github.com/users/shiva-nagndra/repos",
                        headers=headers)
if response.status_code != 200:
    print(f"response is invalid, check the URL", "error: ",response.status_code)


