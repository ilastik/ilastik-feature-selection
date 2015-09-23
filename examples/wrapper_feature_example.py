__author__ = 'fabian'

import sys
sys.path.append("../")
import wrapper_feature_selection
from utils import load_digits
from sklearn import ensemble

def select_features_digits():
    # loads doigits dataset form sklearn and permutes instances (otherwise they are sorted in ascending order by their
    # target label
    X, Y = load_digits()

    # the evaluation function here is coupled to the sklearn random forest but in practice any classifier with a fit()
    # function and a score() function may be used
    rf = ensemble.RandomForestClassifier(n_estimators = 20)

    # the evaluation function here is provided and penalizes the set size (the higher the penalty the smaller the sets
    # should be). You may use your own evaluation function (interface: evaluate_set(X, Y, indices, feature_set))
    eval_function = wrapper_feature_selection.EvaluationFunction(rf, k_fold=5, complexity_penalty=0.4)

    feature_selector = wrapper_feature_selection.FeatureSelection(eval_function.evaluate_feature_set_size_penalty)

    # here are the different wrapper methods that are implemented in the FeatureSelection class
    selected_features_SFS = feature_selector.sequential_feature_selection(X, Y, do_floating_search=False, direction="forward")
    selected_features_SFFS = feature_selector.sequential_feature_selection(X, Y, do_floating_search=True, direction="forward")
    selected_features_SBE = feature_selector.sequential_feature_selection(X, Y, do_floating_search=False, direction="backward")
    selected_features_SBFE = feature_selector.sequential_feature_selection(X, Y, do_floating_search=True, direction="backward")
    selected_features_BFS = feature_selector.best_first_search(X, Y, do_compound_operators=False)
    selected_features_BFSc = feature_selector.best_first_search(X, Y, do_compound_operators=True)

    print "SFS: ", selected_features_SFS
    print "SFFS: ", selected_features_SFFS
    print "SBE: ", selected_features_SBE
    print "SBFE: ", selected_features_SBFE
    print "BFS: ", selected_features_BFS
    print "BFSc: ", selected_features_BFSc


if __name__ == "__main__":
    select_features_digits()