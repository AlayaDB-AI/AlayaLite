from sklearn import preprocessing


def normalize(X):
    X = preprocessing.normalize(X, axis=1, norm="l2")
    return X
