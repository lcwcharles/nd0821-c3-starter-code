'''
Test library
date: 11.28.2022
author: Charles
'''


import pytest
import joblib
import os
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
# import joblib
import sys
sys.path.append('../..')
from starter.starter import train_model
from starter.starter.ml import model, data
# from starter.starter.ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression



logging.basicConfig(
    filename=f"../logs/deploy_machine_learning_test_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# @pytest.fixture()

def path():
    data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'starter')), 
            'data', 'census_clean.csv')
    return data_path

def import_data(path):
    '''
    returns dataframe for the csv found at path
    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df

def cencus_data(path):
    df_data = import_data(path)
    train, test = train_test_split(df_data, test_size=0.20)
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
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = data.process_data(
        test, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb
    )
    return X_train, y_train, X_test, y_test

# @pytest.fixture()
def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        import_data(path())
        logging.info(
            "Testing import_data: SUCCESS - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
    except FileNotFoundError as err:
        logging.error("Testing import_data ERROR: The file wasn't found - %s",
                      time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err

# @pytest.fixture()
def test_prosess_data():
    '''
    test encoder helper
    '''
    try:
        X_train, y_train, X_test, y_test = cencus_data(path())
        trained_model = model.train_model(X_train, y_train)
        # print(X_train.shape, y_train.shape)
        assert X_train.shape[0] ==  y_train.shape[0]
        logging.info(
            "Testing prosess_data: SUCCESS - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error("Testing prosess_data ERROR - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err

# @pytest.fixture()
def test_train_model():
    '''
    test train and save model function
    '''
    try:
        X_train, y_train, X_test, y_test = cencus_data(path())
        trained_model = model.train_model(X_train, y_train)
        # assert hasattr(trained_model 'fit')
        assert isinstance(trained_model, RandomForestClassifier)
        logging.info(
            "Testing train_model: SUCCESS - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error(
            "Testing train_model ERROR - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err

# @pytest.fixture()
def test_save_model():
    '''
    test train and save model function
    '''
    try:
        X_train, y_train, X_test, y_test = cencus_data(path())
        trained_model = model.train_model(X_train, y_train)

        model_dir = 'model'
        model_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'starter')), 
                 model_dir, 'model.pkl')
        train_model.save_model(trained_model, model_path)
        if os.path.exists(model_path):
            # assert Path(image_folder).exists()
            logging.info(
                "Testing save_model: SUCCESS - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
    except FileNotFoundError as err:
        logging.error(
            "Testing save_model ERROR: the file doesn't exist - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err

# if __name__ == "__main__":
#     df_cencus = test_import(train_model.import_data)
#     X, y = test_prosess_data(df_cencus)
#     trained_model = test_train_model(X, y)
#     model_dir = 'model'
#     model_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 
#                  model_dir, 'model.pkl')
#     test_save_model(trained_model, model_path)
#     logging.info(
#         "SUCCESS: all tests finished running - %s",
#         time.strftime('%b_%d_%Y_%H_%M_%S'))
#     print("Finished running testing script")
