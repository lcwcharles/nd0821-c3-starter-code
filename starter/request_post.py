import requests
import uvicorn
import json
from fastapi import FastAPI
# import sys
# sys.path.append('..')
from starter.ml import data, model
from pydantic import BaseModel
from typing import Union
import pandas as pd
import os
import joblib
import json

# headers = {
#     'Content-Type':'application/json'
# }

# class Attributes(BaseModel):
#     age: int = 42
#     workclass: str = "Private"
#     fnlgt: int = 159449
#     education: str = "Bachelors"
#     education_num: int = 13
#     marital_status: str = "Married-civ-spouse"
#     occupation: str = "Exec-managerial"
#     relationship: str = "Husband"
#     race: str = "White"
#     sex: str = "Male"
#     capital_gain: int = 0
#     capital_loss: int = 0
#     hours_per_week: int = 40 
#     native_country: str = "United-States"
# print('=======')
attributes = {
    "age": 42,
    "workclass": "Private",
    "fnlgt": 159449,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# response = requests.post('http://127.0.0.1:8000/inference', data=json.dumps(attributes))
response = requests.post('http://127.0.0.1:8000/inference', json=attributes)
print("POST sent")
print(str(type(attributes)))
print(response.status_code)
# print(response.headers)
# print(response.text)
print(response.json())

# app = FastAPI()

# # Defining a POST request performing model inference
# @app.post("/fastapi/")
# async def perform_inference(input_data: Attributes):
#     return input_data

# if __name__ == "__main__":
#     uvicorn.run(app='request_post:app', host='127.0.0.1', port=8100, reload=True, debug=True)