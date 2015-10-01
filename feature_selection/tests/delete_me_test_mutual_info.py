__author__ = 'fabian'
import sys
sys.path.append("../../cython/")
import mutual_information_old
import utils
import numpy as np
import mutual_information

def normalize_data_for_MI(X):
    for i in xrange(X.shape[1]):
        std = X[:, i].std()
        if std != 0.:
            X[:, i] /= std
            X[:, i] -= X[:, i].min()
    return X

X, Y = utils.load_digits()
X = normalize_data_for_MI(X)
X = np.floor(X).astype("int")

print mutual_information.calculate_mutual_information(X[:,9], X[:,5])
print mutual_information_old.calculate_mutual_information_histogram_binning(X[:,9], X[:,5])