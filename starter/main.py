# Put the code for your API here.
from fastapi import FastAPI

from pydantic import BaseModel
import pandas as pd
import os
import joblib
import numpy as np
import uvicorn

import sys
sys.path.insert(1, './starter')
sys.path.append('./starter/starter')
from ml import data, model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Creating input data model structure using Pydantic
class Attributes(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "age": 39,
    #             "workclass": "State-gov",
    #             "fnlgt": 77516,
    #             "education": "Bachelors",
    #             "education_num": 13,
    #             "marital_status": "Never-married",
    #             "occupation": "Adm-clerical",
    #             "relationship": "Not-in-family",
    #             "race": "White",
    #             "sex": "Male",
    #             "capital_gain": 2174,
    #             "capital_loss": 0,
    #             "hours_per_week": 40,
    #             "native_country": "United-States"
    #         }
    #     }

# Instantiating the app
app = FastAPI()

# Defining a GET on the specified endpoint
@app.get("/")
async def welcome():
    return {"Welcome to machine learning!"}

# Defining a POST request performing model inference
@app.post("/inference")
async def perform_inference(input_data: Attributes):
    df_model = joblib.load(open(os.path.join(os.path.abspath(os.getcwd()), 
            "starter", "model", "model.pkl"), 'rb'))
    encoder = joblib.load(open(os.path.join(os.path.abspath(os.getcwd()), 
            "starter", "model", "model_encoder.pkl"), 'rb'))
    lb = joblib.load(open(os.path.join(os.path.abspath(os.getcwd()), 
            "starter", "model", "model_lb.pkl"), 'rb'))


    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
        ]
    # get the data and convert to array
    df_data = np.array([[input_data.age,input_data.workclass,input_data.fnlgt,input_data.education,
                        input_data.education_num, input_data.marital_status, input_data.occupation,
                        input_data.relationship, input_data.race, input_data.sex, 
                        input_data.capital_gain, input_data.capital_loss, 
                        input_data.hours_per_week, input_data.native_country]])
    # df_data = pd.DataFrame.from_dict([input_data])
    # df_data = pd.DataFrame(input_data, index=[0])
    # return df_data
    # return str(type(df_data))
    df_data = pd.DataFrame(df_data, columns=[
        "age", "workclass", "fnlgt", "education", "education_num", "marital_status", 
        "occupation", "relationship", "race", "sex", "capital_gain",
        "capital_loss", "hours_per_week", "native_country"])
    X, _, _, _ = data.process_data(df_data, 
                                   categorical_features=cat_features, 
                                   training=False, encoder=encoder, lb=lb)
    # process the data for the model prediction
    # return df_data
    preds = model.inference(df_model, X)
    prediction = lb.inverse_transform(preds)[0]
    return prediction

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)