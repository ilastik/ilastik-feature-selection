"""
This file contains regression tests, this means that expected results have been
generated using the same methods as in the tests!
"""

__author__ = 'fabian'
import numpy as np
import ilastik_feature_selection
import sklearn.ensemble
import os
import pytest


@pytest.fixture(scope='module')
def digit_data():
    test_path = os.path.dirname(os.path.realpath(__file__))
    digits_X = np.load(test_path + "/digits_data.npy")
    digits_Y = np.load(test_path + "/digits_target.npy")
    return digits_X, digits_Y


# Note: added random state in order to conform with the old tests. Maybe we
# should remove that, have all in the same random state.
@pytest.mark.parametrize('method, advanced_search, random_state, expected', [
    ('BFS', False, 14271, (set([10, 13, 20, 29, 35, 37, 42, 44, 51, 53]), 1.2052499999999999)),
    ('BFS', True, 14271, (set([18, 19, 20, 21, 29, 34, 38, 42, 43, 44, 46, 51, 54, 61]), 1.2092500000000002)),
    ('SFS', False, 1275, (set([4, 18, 20, 21, 27, 28, 34, 42, 43, 58, 61, 62]), 1.2029999999999998)),
    ('SFS', True, 1275, (set([4, 18, 20, 21, 27, 28, 34, 42, 43, 58, 61, 62]), 1.2029999999999998)),
    ('SBE', False, 1275, (
        set([
            1, 3, 5, 7, 8, 10, 12, 15, 16, 20, 21, 26, 27, 29, 32, 33, 34, 36,
            41, 42, 43, 44, 49, 52, 53, 54, 55, 56, 58, 59, 63]), 1.12425)),
    ('SBE', True, 1275, (
        set([
            0, 1, 4, 5, 12, 15, 17, 20, 21, 22, 24, 26, 27, 29, 32, 33, 36, 39,
            41, 42, 43, 44, 46, 51, 52, 54, 58, 61]), 1.141)),
])
def test_wrapper(digit_data, method, advanced_search, random_state, expected):
    X, Y = digit_data
    rf = sklearn.ensemble.RandomForestClassifier(random_state=random_state, n_estimators=10)
    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method=method)
    a = feat_selector.run(do_advanced_search=advanced_search)

    expected_feature_set, expected_eval_func_value = expected
    assert set(a[0]) == expected_feature_set
    np.testing.assert_almost_equal(a[1], expected_eval_func_value)


def test_initial_set(digit_data):
    X, Y = digit_data
    rf = sklearn.ensemble.RandomForestClassifier(random_state=1275, n_estimators=10)

    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

    a = feat_selector.run(do_advanced_search=False, initial_features=set(range(10)))
    assert set(a[0]) == set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 21, 27, 30, 33, 42, 43, 53])
    np.assert_almost_equal(a[1], 1.1655)

    feat_selector.change_method("BFS")
    a = feat_selector.run(do_advanced_search=False, initial_features=set(range(10)))
    assert set(a[0]) == set([0, 1, 2, 3, 5, 9, 20, 21, 25, 27, 30, 40, 42, 43, 61])
    np.assert_almost_equal(a[1], 1.1862499999999998)


def test_permitted_features(digit_data):
    X, Y = digit_data
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)

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
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)

    eval_fct = ilastik_feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
    feat_selector = ilastik_feature_selection.wrapper_feature_selection.WrapperFeatureSelection(
        X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

    mandatory_features = set(range(20))
    a = feat_selector.run(do_advanced_search=False, mandatory_features=mandatory_features)
    assert mandatory_features.issubset(set(a[0]))

    feat_selector.change_method("BFS")
    a = feat_selector.run(do_advanced_search=False, mandatory_features=mandatory_features)
    assert mandatory_features.issubset(set(a[0]))
