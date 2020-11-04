import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Gender':'Male', 'Height':187, 'Weight':90})
print(r.json())