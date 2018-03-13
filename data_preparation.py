from sklearn.model_selection import train_test_split


def generate_features(dataset):
    pass


def fix_missing_values(dataset):
    pass


def fix_outliers(dataset):
    pass


def data_preparation(train_ds, test_ds):
    generate_features(train_ds)
    generate_features(test_ds)
    fix_missing_values(train_ds)
    fix_missing_values(test_ds)
    fix_outliers(train_ds)
    fix_outliers(test_ds)

    x = train_ds.ix[:, :80]
    y = train_ds.ix[:, 80]
    # x = dataset[:,:8]  # independent variables, pima
    # y = dataset[:, 8]  # dependent variable, pima
    seed = 7
    test_size = 0.33
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    return {'x_train': x, 'x_test': test_ds.ix[:, :79], 'y_train': y, 'y_test': test_ds.ix[:, 79]}