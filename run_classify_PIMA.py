from numpy import loadtxt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd

from data_preparation_classify_PIMA import data_preparation


def load_data():
    return loadtxt('data/pima-indians-diabetes.csv', delimiter=',')

def prediction(train):
    model = XGBClassifier()
    fit = model.fit(train['x_train'],train['y_train'])
    print(fit)

    # predictions for test data
    y_pred = model.predict(train['x_test'])
    predictions = [round(value) for value in y_pred]
    return fit, predictions

def evaluate(train_pred, train_test_data):
    accuracy = accuracy_score(train_test_data['y_test'], train_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':
    # data load
    dataset = load_data()

    # data preparation
    train_test_data = data_preparation(dataset)

    # modeling
    fit, train_pred = prediction(train_test_data)

    # evaluation
    evaluate(train_pred, train_test_data)