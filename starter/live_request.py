'''
A script that POSTS to the API using the requests module and 
returns both the result of model inference and the status code
'''

import requests

request_data = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "India"           
}

resp = requests.post("https://deploy-app-g6a7.onrender.com/inference", json=request_data)

print(resp.status_code)
print(resp.json())