__author__ = 'fabian'

import sys
sys.path.append("../")
import filter_feature_selection
from utils import load_digits

def select_features_digits():
    # loads doigits dataset form sklearn and permutes instances (otherwise they are sorted in ascending order by their
    # target label
    X, Y = load_digits()

    # create feature selector instance. Default criterion is "ICAP"
    feat_selector = filter_feature_selection.FilterFeatureSelection(X, Y)

    # run feature selection. Desired number of features needs to be specified
    selected_features_ICAP = feat_selector.run_selection(10)
    print "Selected features for the digits dataset (ICAP criterion) are: \n", str(selected_features_ICAP)

    # ------------------------------------------------------------------------------------------------------------------
    # some more things you can do:
    # it is possible to change the criterion:
    feat_selector.change_method("CIFE")
    selected_features_CIFE = feat_selector.run_selection(10)
    print "Selected features for the digits dataset (CIFE criterion) are: \n", str(selected_features_CIFE)

    # available methods can be retrieved:
    available_methods = feat_selector.get_available_methods()
    print "available methods: ", str(available_methods)

    # notice how the feature selection becomes faster the more methods have already been run
    # this is because mutual information values will be re-used once they are calculated
    feat_selector_new = filter_feature_selection.FilterFeatureSelection(X, Y)
    available_methods_new = feat_selector_new.get_available_methods()
    print "\nmutual information values are retained once they are calculated. Speedup for consecutive runs with different methods"
    for method in available_methods:
        feat_selector_new.change_method(method)
        print method, ": ", feat_selector_new.run_selection(25)



if __name__ == "__main__":
    select_features_digits()