'''
Test library
date: 11.28.2022
author: Charles
'''

import os
import logging
import time
import joblib
import sys
sys.path.append('../..')
from starter.starter import train_model
from starter.starter.ml import model, data
# from starter.starter.ml.data import process_data
from sklearn.linear_model import LogisticRegression


logging.basicConfig(
    filename=f"../logs/deploy_machine_learning_test_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("../data/census_clean.csv")
        logging.info(
            "Testing import_data: SUCCESS - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
    except FileNotFoundError as err:
        logging.error("Testing import_data ERROR: The file wasn't found - %s",
                      time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    else:
        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: ERROR - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
            raise err
    return df

def test_prosess_data(df_data):
    '''
    test encoder helper
    '''
    try:
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        X, y, encoder, lb = data.process_data(
            df_data, categorical_features=cat_features, label="salary", training=True
            )
        # print(X.shape, y.shape)
        assert X.shape == (32561 ,108)
        assert y.shape == (32561,)
        logging.info(
            "Testing prosess_data: SUCCESS - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error("Testing prosess_data ERROR - %s", time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    return X, y

def test_train_model(input_arr, output_arr):
    '''
    test train and save model function
    '''
    try:
        trained_model = model.train_model(input_arr, output_arr)
        # assert hasattr(trained_model 'fit')
        assert isinstance(trained_model, LogisticRegression)
        logging.info(
            "Testing train_model: SUCCESS - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error(
            "Testing train_model ERROR - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    return trained_model

def test_save_model(df_model, path):
    '''
    test train and save model function
    '''
    try:
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

if __name__ == "__main__":
    df_cencus = test_import(train_model.import_data)
    X, y = test_prosess_data(df_cencus)
    trained_model = test_train_model(X, y)
    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 
                 model_dir, 'model.pkl')
    test_save_model(trained_model, model_path)
    logging.info(
        "SUCCESS: all tests finished running - %s",
        time.strftime('%b_%d_%Y_%H_%M_%S'))
    print("Finished running testing script")
