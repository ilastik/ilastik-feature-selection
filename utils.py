__author__ = 'fabian'
import sklearn
import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import cross_validation


def load_iris():
    tmp = datasets.load_iris()
    data = np.array(tmp['data'])
    n_samples = data.shape[0]
    permIndexes=np.random.permutation(n_samples)
    data = data[permIndexes]
    target = np.array(tmp['target'])
    target = target[permIndexes]
    return data, target

def load_digits():
    tmp = datasets.load_digits()
    data = np.array(tmp['data'])
    n_samples = data.shape[0]
    permIndexes=np.random.permutation(n_samples)
    data = data[permIndexes]
    target = np.array(tmp['target'])
    target = target[permIndexes]
    return data, target

def kfold_train_and_predict(X, Y, classifier, k = 5, indices = None, features = None):
    if indices is None:
        indices = np.array(range(X.shape[0]))
    if features is None:
        features = np.array(range(X.shape[1]))
    kf = cross_validation.KFold(len(indices), n_folds=k)
    accurs = []
    for train, test in kf:
        train_ind = indices[train].astype("int")
        test_ind = indices[test].astype("int")

        classifier.fit(X[train_ind,:][:,features], Y[train_ind])
        accurs += [classifier.score(X[test_ind,:][:,features], Y[test_ind])]

    accurs = np.array(accurs)
    return np.mean(accurs), np.std(accurs)