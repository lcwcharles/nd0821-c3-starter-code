# Script to train machine learning model.
# import imp
import os
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import sys
sys.path.insert(1, './starter')
sys.path.append('./starter/starter')
from ml import data, model
# from ml import data, model
# from starter.starter.ml.model import train_model, compute_model_metrics, inference
# from starter.starter.ml.data import process_data

# Add the necessary imports for the starter code.
logging.basicConfig(
    filename=f"starter/logs/train_model_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
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
    train the data and return the model
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
    path = os.path.join(os.path.abspath(os.getcwd()), 'starter',
            'data', 'census_clean.csv')
    # print(path)
    df = import_data(path)
    # df.columns = df.columns.str.strip()
    # data_df = df.drop_duplicates()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(df, test_size=0.20, random_state=20)

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
    model_path = os.path.join(os.path.abspath(os.getcwd()), 'starter',
                 model_dir, 'model.pkl')
    # print(model_path)
    save_model(trained_model, model_path)
    logging.info('Save the model to %s - %s', model_path, time.strftime('%b_%d_%Y_%H_%M_%S'))

    # Save encoder and binarizer
    encoder_path = os.path.join(os.path.abspath(os.getcwd()), 'starter',
                    model_dir, 'model_encoder.pkl')
    joblib.dump(encoder, open(encoder_path, 'wb'))
    logging.info('Save the encoder to %s - %s', encoder_path, time.strftime('%b_%d_%Y_%H_%M_%S'))

    binarizer_path = os.path.join(os.path.abspath(os.getcwd()), 'starter',
                     model_dir, 'model_lb.pkl')
    joblib.dump(lb, open(binarizer_path, 'wb'))
    logging.info('Save the binarizer to %s - %s', binarizer_path, time.strftime('%b_%d_%Y_%H_%M_%S'))

    slice_path = os.path.join(os.path.abspath(os.getcwd()), 'starter',
                     model_dir, 'slice_output.txt')
    with open(slice_path, 'w') as file:
        for feature in cat_features:
            # print(feature)
            file.write(f'Feature: {feature}\n')
            for f_value in test[feature].unique():
                # print(f_value)
                slice_df = test[test[feature] == f_value]
                X_slice, y_slice, _, _ = data.process_data(
                    slice_df,
                    categorical_features=cat_features,
                    label="salary", training=False,
                    encoder=encoder, lb=lb)
                predictions_slice = model.inference(trained_model, X_slice)
                precision_slice, recall_slice, f_beta_slice = model.compute_model_metrics(
                    y_slice, predictions_slice)
                # file.write('Value: '+ f_value + ', Precision: ' + str(precision_slice)
                        #  + ', Recall: ' + str(recall_slice) + ', F-beta score: ' + str(f_beta_slice))
                file.write(f'Value: {f_value}' + f', Precision: {precision_slice}'
                    + f', Recall: {recall_slice}' + f', F-beta score: {f_beta_slice}\n')
            file.write('--------------------\n')
        file.close()
        logging.info('Save the slice_output.txt to %s - %s', slice_path, time.strftime('%b_%d_%Y_%H_%M_%S'))