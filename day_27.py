#API, #read. n write

with open("notes.txt", "r") as file:
    content = file.read()
    print(content)

with open("notes.txt", "w") as file:
    content = file.write("learned how files work")

import csv

with open("data.csv", "w", newline="") as file:
    writer= csv.writer(file)
    writer.writerow(["step","loss"])
    writer.writerow([1, 3.6])
    writer.writerow([2, 2.3])


import requests

response = requests.get("https://api.github.com")

print(response.status_code)
print(response.json())

data = response.json()
print(data['commit_search_url'])


#practice
with open("day_27_files_and_apis.py","w", newline="") as f:
    write= f.write("#testing and success")

with open("day_27_files_and_apis.py", "r") as f:
    read = f.read()
    print(read)

import requests

apicall = requests.get("https://github.com/shiva-nagendra")

apicall = requests.get("https://api.github.com/users/shiva-nagendra")
print(apicall.status_code)
print(apicall.json())
joson=response.json()
print(joson['followers_url'])

print(apicall.headers["Content-Type"])


