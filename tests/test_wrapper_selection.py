__author__ = 'fabian'
import sys
sys.path.append('../')
import utils
import numpy as np
import wrapper_feature_selection
import sklearn

rf = sklearn.ensemble.RandomForestClassifier()

eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

X, Y = utils.load_digits()
# feat_selector.sequential_feature_selection(X, Y, direction = "backward")

print feat_selector.best_first_search(X, Y, do_compound_operators=False)