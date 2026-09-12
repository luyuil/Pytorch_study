import requests

response = requests.get("https://luyuil.github.io/luyuil_blog/")
print(response.content)