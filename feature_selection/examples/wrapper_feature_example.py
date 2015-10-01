__author__ = 'fabian'

from feature_selection import wrapper_feature_selection
from sklearn import ensemble
from sklearn import datasets
import numpy as np

def select_features_digits():
    # loads doigits dataset form sklearn and permutes instances (otherwise they are sorted in ascending order by their
    # target label
    X = datasets.load_iris()['data']
    Y = datasets.load_iris()['target']

    n_samples = X.shape[0]
    permIndexes = np.random.permutation(n_samples)
    X = X[permIndexes]
    Y = Y[permIndexes]

    # the evaluation function here is coupled to the sklearn random forest but in practice any classifier with a fit()
    # function and a score() function may be used
    rf = ensemble.RandomForestClassifier(n_estimators = 20)

    # the evaluation function here is provided and penalizes the set size (the higher the penalty the smaller the sets
    # should be). You may use your own evaluation function (interface: evaluate_set(X, Y, indices, feature_set))
    eval_function = wrapper_feature_selection.EvaluationFunction(rf, k_fold=5, complexity_penalty=0.05)

    feature_selector = wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_function.evaluate_feature_set_size_penalty, method="SFS")

    selected_features_SFS = feature_selector.run()
    selected_features_SFFS = feature_selector.run(do_advanced_search=True)

    feature_selector.change_method("SBE")
    selected_features_SBE = feature_selector.run()
    selected_features_SBFE = feature_selector.run(do_advanced_search=True)

    feature_selector.change_method("BFS")
    selected_features_BFS = feature_selector.run()
    selected_features_BFSc = feature_selector.run(do_advanced_search=True)

    # here are the different wrapper methods that are implemented in the FeatureSelection class

    print "SFS: ", selected_features_SFS
    print "SFFS: ", selected_features_SFFS
    print "SBE: ", selected_features_SBE
    print "SBFE: ", selected_features_SBFE
    print "BFS: ", selected_features_BFS
    print "BFSc: ", selected_features_BFSc


if __name__ == "__main__":
    select_features_digits()