import json
import sys
print(sys.path)
sys.path.append('.')
from starter.main import app
# print(main.path)
from fastapi.testclient import TestClient


client = TestClient(app)

# Testing GET
def test_welcome():
    r = client.get("http://127.0.0.1:8000/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!"}

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
    resp_str = resp.json()
    print(resp_str)
    assert resp.status_code == 200
    assert resp_str == "<=50K"


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
    resp_str = resp.json()
    assert resp.status_code == 200
    assert resp_str == ">50K"
