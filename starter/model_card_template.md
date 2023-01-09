# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is based on Random Forest Classifier machine learning alogrithm from Sklearn to predict a persons salary. 

## Intended Use

The model is used to predict whether a person makes over 50K a year.

## Training Data

The dataset was extracted by Barry Becker from the 1994 Census database.More information on the data can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).
80 % of the dataset is used for training.

## Evaluation Data

20 % of the dataset is used for evaluation.

## Metrics

The model's performance on those metrics:
precision:  0.7307

recall:  0.6262

f_beta:  0.6744

## Ethical Considerations

The modeling attributes include race and gender, which may lead to discrimination.

## Caveats and Recommendations

The dataset was extracted from the 1994 Census database. Compared with the current population structure, the data may be outdated, and the prediction error may be large.