from importlib_metadata import re
import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # # grid search
    # rfc = RandomForestClassifier(random_state=42)
    # param_grid = {
    #     'n_estimators': [200, 500],
    #     'max_features': ['auto', 'sqrt'],
    #     'max_depth': [4, 5, 100],
    #     'criterion': ['gini', 'entropy']
    # }
    # # define grid search and fit random forest classifier
    # cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # cv_rfc.fit(X_train, y_train)
    # # save best model
    # joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    # model = LogisticRegression(solver='lbfgs', max_iter=3000)
    ## when use LogisticRegression the score
    ## precision:  0.6289707750952986
    ## recall:  0.31190926275992437
    ## f_beta:  0.41701769165964614

    ## when use RandomForestClassifier the score
    ## precision:  0.7318181818181818
    ## recall:  0.6196279666452854
    ## f_beta:  0.6710663424800277

    model = RandomForestClassifier()

    # fit model
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # predictions for logistic regression
    preds = model.predict(X)
    return preds