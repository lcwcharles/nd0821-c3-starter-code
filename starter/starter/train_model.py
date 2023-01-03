# Script to train machine learning model.
# import imp
import os
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import sys
sys.path.append('../..')
from starter.starter.ml import data, model
# from starter.starter.ml.model import train_model, compute_model_metrics, inference
# from starter.starter.ml.data import process_data

# Add the necessary imports for the starter code.
logging.basicConfig(
    filename=f"../logs/train_model_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Add code to load in the data.
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

def train_model(input_arr, output_arr):
    '''
    train the data and return it
    '''
    trained_model = model.train_model(input_arr, output_arr)
    return trained_model

def save_model(df_model, path):
    '''
    save the model to specific path
    '''
    with open(path, 'wb') as file:
        joblib.dump(df_model, file)

if __name__ == '__main__':
    ## os.path.split(__file__)[0]
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 
            'data', 'census_clean.csv')
    # print(path)
    df = import_data(path)
    # df.columns = df.columns.str.strip()
    # data_df = df.drop_duplicates()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(df, test_size=0.20)

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
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = data.process_data(
        test, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb
    )
    logging.info('Training model - %s', time.strftime('%b_%d_%Y_%H_%M_%S'))
    trained_model = train_model(X_train, y_train)
    
    # Inference and performance metrics on the test dataset
    predictions = model.inference(trained_model, X_test)
    precision, recall, f_beta = model.compute_model_metrics(y_test, predictions)
    print('precision: ', precision)
    logging.info('precision:  %s - %s', precision, time.strftime('%b_%d_%Y_%H_%M_%S'))
    print('recall: ', recall)
    logging.info('recall:  %s - %s', recall, time.strftime('%b_%d_%Y_%H_%M_%S'))
    print('f_beta: ', f_beta)
    logging.info('f_beta:  %s - %s', f_beta, time.strftime('%b_%d_%Y_%H_%M_%S'))

    # Save model 
    model_dir = 'model'
    model_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 
                 model_dir, 'model.pkl')
    # print(model_path)
    save_model(trained_model, model_path)
    logging.info('Save the model to %s - %s', model_path, time.strftime('%b_%d_%Y_%H_%M_%S'))

    # Save encoder and binarizer
    binarizer_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')),
                     model_dir, 'model_lb.pkl')
    joblib.dump(lb, open(binarizer_path, 'wb'))
    encoder_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 
                    model_dir, 'model_encoder.pkl')
    joblib.dump(encoder, open(encoder_path, 'wb'))
