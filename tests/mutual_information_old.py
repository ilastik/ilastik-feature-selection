__author__ = 'fabian'


import numpy as np

def normalize_data_for_MI(X):
    for i in range(X.shape[1]):
        std = X[:, i].std()
        if std != 0.:
            X[:, i] /= std
            X[:, i] -= X[:, i].min()
    return np.floor(X).astype("int")

def calculate_mutual_information_histogram_binning(X1, X2, base = 2.):
    if len(X1)!=(len(X2)):
        raise ValueError("X1 and X2 must have the same length")

    X1 = np.array(X1)
    X2 = np.array(X2)

    X1 = np.floor(X1).astype("int")
    X2 = np.floor(X2).astype("int")

    # calculate marginal distributions of X1 and X2
    X1_cumstates = np.zeros(X1.max()+1)
    for i in np.arange(X1.max()+1):
        X1_cumstates[i] = np.sum(X1 == i)

    X2_cumstates = np.zeros(X2.max()+1)
    for i in np.arange(X2.max()+1):
        X2_cumstates[i] = np.sum(X2 == i)

    X1_cumstates /= np.sum(X1_cumstates).astype("float32")
    X2_cumstates /= np.sum(X2_cumstates).astype("float32")

    # calculate joint distribution
    histogram = np.zeros((np.max(X1)+1, np.max(X2)+1))

    for i in range(len(X1)):
        histogram[X1[i], X2[i]] += 1

    hist_normalized = histogram.astype("float32") / np.sum(histogram).astype("float32")

    mutual_information = 0.
    for i in range(np.max(X1)+1):
        for j in range(np.max(X2)+1):
            if hist_normalized[i, j] != 0:
                mutual_information += hist_normalized[i, j] * (np.log(hist_normalized[i, j] / X1_cumstates[i] / X2_cumstates[j]) / np.log(base))

    return mutual_information

def calculate_conditional_MI(X1, X2, Y, base = 2.):
    states = np.unique(Y)
    con_mi = 0.

    for state in states:
        indices = (Y == state)
        p_state = float(np.sum(indices)) / float(len(Y))
        mi = calculate_mutual_information_histogram_binning(X1[indices], X2[indices], base)
        con_mi += p_state * mi
    return con_mi