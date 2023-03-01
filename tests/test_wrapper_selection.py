"""
This file contains regression tests, this means that expected results have been
generated using the same methods as in the tests!

- 2018-12-18: re-generated expected results.
  https://github.com/ilastik/ilastik-feature-selection/issues/1
  Since the proposed workaround is to re-generate the expected results, this
  was done, having sklearn 0.18 as a dependency
"""

__author__ = 'fabian'
import numpy as np
import ilastik_feature_selection
import vigra
import os
from sklearn.metrics import accuracy_score
import pytest


class VigraRFwRandomState(vigra.learning.RandomForest):
    """
    Adaptor class that exposes an interface more similar to sklearn.ensemble.RandomForestClassifier
    which is expected in wrapper_selection.
    """
    def __init__(self, *args, random_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_state = random_state

    def fit(self, X, y):
        return self.learnRF(X, y, self._random_state if self._random_state else 0)

    def score(self, X, y, sample_weight=None) -> float:
        """
        evaluates X and returns mean accuracy wrt y
        """
        return accuracy_score(y, self.predictLabels(X), sample_weight=sample_weight)


@pytest.fixture(scope='module')
def digit_data():
    test_path = os.path.dirname(os.path.realpath(__file__))
    digits_X = np.load(test_path + "/digits_data.npy")
    digits_Y = np.load(test_path + "/digits_target.npy")
    return digits_X.astype("float32"), digits_Y.astype("uint32")[..., np.newaxis]


# Note: added random state in order to conform with the old tests. Maybe we
# should remove that, have all in the same random state.
@pytest.mark.parametrize('method, advanced_search, random_state, expected', [
    ('BFS', False, 14271, (set([9, 13, 20, 21, 27, 34, 36, 38, 42, 43, 50, 57, 58, 60]), 1.20525)),
    ('BFS', True, 14271, (set([18, 20, 21, 27, 30, 34, 42, 43, 44, 54, 61]), 1.20925)),
    ('SFS', False, 1275, (set([10, 13, 21, 26, 27, 28, 35, 38, 42, 44, 54]), 1.20525)),
    ('SFS', True, 1275, (set([10, 13, 21, 26, 27, 28, 35, 38, 42, 44, 54]), 1.20525)),
    ('SBE', False, 1275, (
        set([
            1, 9, 11, 12, 15, 18, 21, 23, 26, 27, 31, 32, 34, 37, 39, 42, 43,
            44, 45, 46, 48, 49, 54, 58, 60]), 1.12975)),
    ('SBE', True, 1275, (set([20, 21, 26, 27, 34, 37, 38, 42, 43, 44, 49, 61]), 1.215)),
])
def test_wrapper(digit_data, method, advanced_search, random_state, expected):
    X, Y = digit_data
    rf = VigraRFwRandomState(random_state=random_state, treeCount=10)
    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method=method)
    a = feat_selector.run(do_advanced_search=advanced_search)

    expected_feature_set, expected_eval_func_value = expected
    assert set(a[0]) == expected_feature_set
    np.testing.assert_almost_equal(a[1], expected_eval_func_value)


def test_initial_set(digit_data):
    X, Y = digit_data
    rf = VigraRFwRandomState(random_state=1275, treeCount=10)

    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

    a = feat_selector.run(do_advanced_search=False, initial_features=set(range(10)))
    expected_feature_set = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 22, 26, 27, 35, 38, 42, 43, 45, 53, 58])
    expected_eval_func_value = 1.14475
    assert set(a[0]) == expected_feature_set
    np.testing.assert_almost_equal(a[1], expected_eval_func_value)

    feat_selector.change_method("BFS")
    a = feat_selector.run(do_advanced_search=False, initial_features=set(range(10)))
    expected_feature_set = set([0, 3, 4, 5, 6, 9, 21, 26, 27, 30, 42, 43, 45, 50, 51, 52, 53])
    expected_eval_func_value = 1.19975
    assert set(a[0]) == expected_feature_set
    np.testing.assert_almost_equal(a[1], expected_eval_func_value)


def test_permitted_features(digit_data):
    X, Y = digit_data
    rf = VigraRFwRandomState(treeCount=10)

    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

    permitted_features = set(range(20))
    a = feat_selector.run(do_advanced_search=False, permitted_features=permitted_features)
    assert set(a[0]).issubset(permitted_features)

    feat_selector.change_method("BFS")
    a = feat_selector.run(do_advanced_search=False, permitted_features=permitted_features)
    assert set(a[0]).issubset(permitted_features)


def test_mandatory_features(digit_data):
    X, Y = digit_data
    rf = VigraRFwRandomState(treeCount=10)

    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

    mandatory_features = set(range(20))
    a = feat_selector.run(do_advanced_search=False, mandatory_features=mandatory_features)
    assert mandatory_features.issubset(set(a[0]))

    feat_selector.change_method("BFS")
    a = feat_selector.run(do_advanced_search=False, mandatory_features=mandatory_features)
    assert mandatory_features.issubset(set(a[0]))
