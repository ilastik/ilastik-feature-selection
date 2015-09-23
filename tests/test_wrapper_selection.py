__author__ = 'fabian'
import sys
sys.path.append('../')
import utils
import numpy as np
import wrapper_feature_selection
import sklearn
import unittest
import logging

class TestWrapperMethod(unittest.TestCase):
    def test_wrapper_BFS_no_compound(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        # feat_selector.sequential_feature_selection(X, Y, direction = "backward")

        a = feat_selector.best_first_search(X, Y, do_compound_operators=False)
        self.assertEqual(a[1], 1.2572481429897864)
        self.assertEqual(set(a[0]), set([10, 12, 21, 26, 27, 30, 34, 42, 43, 58, 61]))

    def test_wrapper_BFS_compound(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        # feat_selector.sequential_feature_selection(X, Y, direction = "backward")

        a = feat_selector.best_first_search(X, Y, do_compound_operators=True)
        self.assertEqual(a[1], 1.2600274682760757)
        self.assertEqual(set(a[0]), set([10, 12, 18, 20, 21, 27, 34, 36, 42, 43, 61]))

    def test_SFS_floating(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=True)

        self.assertEqual(a[1], 1.2614055246053852)
        self.assertEqual(set(a[0]), set([10, 21, 26, 27, 30, 42, 43, 51, 61]))

    def test_SFS_no_floating(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=False)

        self.assertEqual(a[1], 1.2560198081089446)
        self.assertEqual(set(a[0]), set([10, 11, 21, 26, 27, 30, 34, 42, 43, 45, 54, 61]))

    def test_SBS_floating(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=True)

        self.assertEqual(a[1], 1.2441863974001857)
        self.assertEqual(set(a[0]), set([ 2, 18, 21, 26, 27, 35, 37, 42, 43, 53, 54, 55, 63]))

    def test_SBS_no_floating(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)

        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=False)

        self.assertEqual(a[1], 1.2344378675332717)
        self.assertEqual(set(a[0]), set([ 2, 18, 21, 26, 27, 35, 36, 39, 42, 43, 54]))

    def test_feature_search_space(self):
        rf = sklearn.ensemble.RandomForestClassifier()
        search_space = set([1,2,3,4,5,6,7,8,9])
        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=False, feature_search_space=search_space)
        self.assertTrue(set(a[0]).issubset(search_space))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=True, feature_search_space=search_space)
        self.assertTrue(set(a[0]).issubset(search_space))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=False, feature_search_space=search_space)
        self.assertTrue(set(a[0]).issubset(search_space))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=True, feature_search_space=search_space)
        self.assertTrue(set(a[0]).issubset(search_space))
        a = feat_selector.best_first_search(X, Y, feature_search_space=search_space, do_compound_operators=True)
        self.assertTrue(set(a[0]).issubset(search_space))
        a = feat_selector.best_first_search(X, Y, feature_search_space=search_space, do_compound_operators=False)
        self.assertTrue(set(a[0]).issubset(search_space))

    def test_constant_features(self):
        # feature 60 is not included in the sets of the tests above (excluding test_feature_search_space)

        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271)
        constant_features = set([60])
        eval_fct = wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = wrapper_feature_selection.FeatureSelection(eval_fct.evaluate_feature_set_size_penalty)

        X = np.load("digits_data.npy")
        Y = np.load("digits_target.npy")
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=False,
                                                       constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "forward", do_floating_search=True,
                                                       constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=False,
                                                       constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))
        a = feat_selector.sequential_feature_selection(X, Y, direction = "backward", do_floating_search=True,
                                                       constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))
        a = feat_selector.best_first_search(X, Y, do_compound_operators=True, constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))
        a = feat_selector.best_first_search(X, Y, do_compound_operators=True, constant_feature_ids=constant_features)
        self.assertTrue(constant_features.issubset(set(a[0])))


if __name__ == "__main__":
    unittest.main()