__author__ = 'fabian'

import utils
import numpy as np
import wrapper_feature_selection
import sklearn

rf = sklearn.ensemble.RandomForestClassifier()

eval_fct = wrapper_feature_selection.EvaluationFunctionRF(rf)
feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

X, Y = utils.load_iris()
feat_selector.sequential_feature_selection(X, Y, direction = "backward")