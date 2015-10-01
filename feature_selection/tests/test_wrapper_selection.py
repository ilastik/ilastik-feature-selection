__author__ = 'fabian'
import sys
sys.path.append('../')
import utils
import numpy as np
import feature_selection
import sklearn
import unittest
import os
import logging

class TestWrapperMethod(unittest.TestCase):
    def test_wrapper_BFS_no_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="BFS")

        a = feat_selector.run(do_advanced_search=False)
        self.assertEqual(a[1], 1.1995)
        self.assertEqual(set(a[0]), set([10, 13, 20, 29, 35, 37, 42, 44, 51, 53]))

    def test_wrapper_BFS_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 14271, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="BFS")

        a = feat_selector.run(do_advanced_search=True)
        self.assertEqual(a[1], 1.2064999999999999)
        self.assertEqual(set(a[0]), set([18, 19, 20, 21, 29, 34, 38, 42, 43, 44, 46, 51, 54, 61]))

    def test_wrapper_SFS_no_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 1275, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

        a = feat_selector.run(do_advanced_search=False)
        self.assertEqual(a[1], 1.2029999999999998)
        self.assertEqual(set(a[0]), set([ 4, 18, 20, 21, 27, 28, 34, 42, 43, 58, 61, 62]))

    def test_wrapper_SFS_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 1275, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

        a = feat_selector.run(do_advanced_search=True)
        self.assertEqual(a[1], 1.2029999999999998)
        self.assertEqual(set(a[0]), set([ 4, 18, 20, 21, 27, 28, 34, 42, 43, 58, 61, 62]))

    def test_wrapper_SBE_no_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 1275, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SBE")

        a = feat_selector.run(do_advanced_search=False)
        self.assertEqual(a[1], 1.12425)
        self.assertEqual(set(a[0]), set([ 1,  3,  5,  7,  8, 10, 12, 15, 16, 20, 21, 26, 27, 29, 32, 33, 34,
       36, 41, 42, 43, 44, 49, 52, 53, 54, 55, 56, 58, 59, 63]))

    def test_wrapper_SBE_adv(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 1275, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SBE")

        a = feat_selector.run(do_advanced_search=True)
        self.assertEqual(a[1], 1.141)
        self.assertEqual(set(a[0]), set([ 0,  1,  4,  5, 12, 15, 17, 20, 21, 22, 24, 26, 27, 29, 32, 33, 36,
       39, 41, 42, 43, 44, 46, 51, 52, 54, 58, 61]))

    def test_initial_set(self):
        rf = sklearn.ensemble.RandomForestClassifier(random_state = 1275, n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

        a = feat_selector.run(do_advanced_search=False, initial_features = set(range(10)))
        self.assertEqual(set(a[0]), set([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 18, 21, 27, 30, 33, 42, 43,
       53]))
        self.assertEqual(a[1], 1.1655)

        feat_selector.change_method("BFS")
        a = feat_selector.run(do_advanced_search=False, initial_features = set(range(10)))
        self.assertEqual(set(a[0]), set([ 0,  1,  2,  3,  5,  9, 20, 21, 25, 27, 30, 40, 42, 43, 61]))
        self.assertEqual(a[1], 1.1862499999999998)

    def test_permitted_features(self):
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

        permitted_features = set(range(20))
        a = feat_selector.run(do_advanced_search=False, permitted_features = permitted_features)
        self.assertTrue(set(a[0]).issubset(permitted_features))

        feat_selector.change_method("BFS")
        a = feat_selector.run(do_advanced_search=False, permitted_features = permitted_features)
        self.assertTrue(set(a[0]).issubset(permitted_features))

    def test_mandatory_features(self):
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
        X = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_data.npy")
        Y = np.load(os.path.dirname(os.path.realpath(__file__)) +"/digits_target.npy")

        eval_fct = feature_selection.wrapper_feature_selection.EvaluationFunction(rf, complexity_penalty=0.4)
        feat_selector = feature_selection.wrapper_feature_selection.WrapperFeatureSelection(X, Y, eval_fct.evaluate_feature_set_size_penalty, method="SFS")

        mandatory_features = set(range(20))
        a = feat_selector.run(do_advanced_search=False, mandatory_features = mandatory_features)
        self.assertTrue(set(mandatory_features).issubset(set(a[0])))

        feat_selector.change_method("BFS")
        a = feat_selector.run(do_advanced_search=False, mandatory_features = mandatory_features)
        self.assertTrue(set(mandatory_features).issubset(set(a[0])))


if __name__ == "__main__":
    unittest.main()