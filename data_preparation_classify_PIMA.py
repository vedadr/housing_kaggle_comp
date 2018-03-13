from sklearn.model_selection import train_test_split


def generate_features(dataset):
    pass


def fix_missing_values(dataset):
    pass


def fix_outliers(dataset):
    pass


def data_preparation(dataset):
    generate_features(dataset)
    fix_missing_values(dataset)
    fix_outliers(dataset)

    x = dataset[:,:8]  # independent variables, pima
    y = dataset[:, 8]  # dependent variable, pima
    seed = 7
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}