from sklearn.metrics import explained_variance_score
import pandas as pd
import statsmodels.api as sm


from data_preparation import data_preparation


def load_data():
    # return loadtxt('data/pima-indians-diabetes.csv', delimiter=',')
    test_data = pd.read_csv('./house_data/test.csv')
    train_data = pd.read_csv('./house_data/train.csv')
    return train_data, test_data

def prediction(train):
    model = sm.OLS(train['x_train']['LotArea'].astype(float), train['y_train'].astype(float))

    fit = model.fit()
    print(fit.summary())

    # predictions
    predicted_price = fit.predict(train['x_train']['LotArea'].astype(float))

    return predicted_price, train

def evaluate(test_pred, train_test_data):
    accuracy = explained_variance_score(train_test_data['y_train'], test_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':
    # data load
    dataset_train, dataset_test = load_data()

    # data preparation
    train_test_data = data_preparation(dataset_train, dataset_test)

    # modeling
    test_pred, train_test_data = prediction(train_test_data)

    # evaluation
    evaluate(test_pred, train_test_data)