'''
slice library
date: 12.06.2022
author: Charles
'''

from statistics import mode
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import logging
from starter.starter.ml import model, data
# from starter.starter.train_model import import_data


path = '../data/census_clean.csv'
df = pd.read_csv(path)
df.columns = df.columns.str.strip()
data = df.drop_duplicates()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

slice_metrics = []
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

trained_model = joblib.load('../model/model.pkl')
for cat in cat_features:
    for cat_value in test[cat].unique():
        df_slice = test[test[cat] == cat_value]
        # encoder = pd.read_pickle(r"encoder.pkl")
        X_slice, y_slice, _, _ = data.process_data(
            df_slice,
            categorical_features=cat_features,
            label=None, encoder=encoder, lb=lb, training=False)

        y_preds = model.inference(trained_model, X_slice)
        y = df_slice.iloc[:, -1:]
        lb = LabelEncoder()
        y = lb.fit_transform(np.ravel(y))
        precision_sl, recall_sl, fbeta_sl = model.compute_model_metrics(y, y_preds)
        line = "[%s->%s] Precision: %s " \
               "Recall: %s FBeta: %s" % (cat, cat_value, precision_sl, recall_sl, fbeta_sl)
        logging.info(line)
        slice_metrics.append(line)


with open('slice_output.txt', 'w') as out:
    for slice_metric in slice_metrics:
        out.write(slice_metric + '\n')