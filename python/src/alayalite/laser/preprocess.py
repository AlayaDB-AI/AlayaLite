"""Normalization helpers for preprocessing vector data."""

from sklearn import preprocessing


def normalize(X):  # pylint: disable=invalid-name
    X = preprocessing.normalize(X, axis=1, norm="l2")
    return X
