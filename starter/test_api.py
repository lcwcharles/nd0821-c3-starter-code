import json
from fastapi.testclient import TestClient
import sys
sys.path.insert(1, '.')
sys.path.append('./starter')
from main import app

client = TestClient(app)

# Testing GET
def test_welcome():
    resp = client.get("http://127.0.0.1:8000/")
    assert resp.status_code == 200
    assert resp.json() == ["Welcome to machine learning!"]

# Testing POST for prediction <= 50k
def test_inference_less_than_50k():
    attributes = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"           
    }
    resp = client.post("http://127.0.0.1:8000/inference", json=attributes)
    assert resp.status_code == 200
    assert resp.json() == "<=50K"


# Testing POST for prediction > 50k
def test_inference_greater_than_50k():
    attributes = {
        # "age": 52,
        # "workclass": "Self-emp-not-inc",
        # "fnlgt": 209642,
        # "education": "HS-grad",
        # "education_num": 9,
        # "marital_status": "Married-civ-spouse",
        # "occupation": "Exec-managerial",
        # "relationship": "Husband",
        # "race": "White",
        # "sex": "Male",
        # "capital_gain": 0,
        # "capital_loss": 0,
        # "hours_per_week": 45,
        # "native_country": "United-States"
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
    resp = client.post("http://127.0.0.1:8000/inference", json=attributes)
    assert resp.status_code == 200
    assert resp.json() == ">50K"
