__author__ = 'fabian'

import ilastik_feature_selection
from sklearn import datasets
import numpy as np

def select_features_digits():
    # loads doigits dataset form sklearn and permutes instances (otherwise they are sorted in ascending order by their
    # target label
    X = datasets.load_digits()['data']
    Y = datasets.load_digits()['target']

    n_samples = X.shape[0]
    permIndexes = np.random.permutation(n_samples)
    X = X[permIndexes]
    Y = Y[permIndexes]


    # create feature selector instance. Default criterion is "ICAP"
    feat_selector = feature_selection.filter_feature_selection.FilterFeatureSelection(X, Y)

    # run feature selection. Desired number of features needs to be specified
    selected_features_ICAP = feat_selector.run(10)
    print("Selected features for the digits dataset (ICAP criterion) are: \n", str(selected_features_ICAP))

    # ------------------------------------------------------------------------------------------------------------------
    # some more things you can do:
    # it is possible to change the criterion:
    feat_selector.change_method("CIFE")
    selected_features_CIFE = feat_selector.run(10)
    print("Selected features for the digits dataset (CIFE criterion) are: \n", str(selected_features_CIFE))

    # available methods can be retrieved:
    available_methods = feat_selector.get_available_methods()
    print("available methods: ", str(available_methods))

    # notice how the feature selection becomes faster the more methods have already been run
    # this is because mutual information values will be re-used once they are calculated
    feat_selector_new = feature_selection.filter_feature_selection.FilterFeatureSelection(X, Y)
    available_methods_new = feat_selector_new.get_available_methods()
    print("\nmutual information values are retained once they are calculated. Speedup for consecutive runs with different methods")
    for method in available_methods:
        feat_selector_new.change_method(method)
        print(method, ": ", feat_selector_new.run(25))



if __name__ == "__main__":
    select_features_digits()