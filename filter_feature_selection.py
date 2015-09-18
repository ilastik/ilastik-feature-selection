__author__ = 'fabian'
import mutual_information
import numpy as np
import logging
import IPython

logger = logging.Logger('filter_feature_selection')
logger.setLevel(logging.DEBUG)

fhandler = logging.FileHandler('filter_log.txt', 'w')

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)

class FilterFeatureSelection(object):
    def __init__(self, X, Y, method):
        if X.shape[0] != len(Y):
            raise ValueError("X must have as many samples as there are labels in Y")

        self._n_features = X.shape[1]
        self._X = mutual_information.normalize_data_for_MI(X)
        self._Y = Y
        self._method_str = method
        self._methods = {
            "CIFE": self.__J_CIFE,
            "ICAP": self.__J_ICAP,
            "CMIM": self.__J_CMIM,
            "JMI": self.__J_JMI
        }
        self.change_method(method)
        self._method = self._methods[method]

        self._redundancy = np.zeros((self._n_features, self._n_features)) - 1.
        self._relevancy = np.zeros((self._n_features)) - 1
        self._class_cond_red = np.zeros((self._n_features, self._n_features)) - 1


    def change_method(self, method):
        if method not in self._methods.keys():
            raise ValueError("method must be one of the following: %s"%str(self._methods.keys()))
        self._method = self._methods[method]
        self._method_str = method

    def get_current_method(self):
        print self._method

    def get_available_methods(self):
        return self._methods.keys()

    def _get_relevancy(self, feat_id):
        if self._relevancy[feat_id] == -1:
            self._relevancy[feat_id] = mutual_information.calculate_mutual_information_histogram_binning(self._X[:, feat_id], self._Y)
        return self._relevancy[feat_id]

    def _get_redundancy(self, feat1, feat2):
        if self._redundancy[feat1, feat2] == -1:
            this_redundancy = mutual_information.calculate_mutual_information_histogram_binning(self._X[:, feat1], self._X[:, feat2])
            self._redundancy[feat1, feat2] = this_redundancy
            self._redundancy[feat2, feat1] = this_redundancy
        return self._redundancy[feat1, feat2]

    def _get_class_cond_red(self, feat1, feat2):
        if self._class_cond_red[feat1, feat2] == -1:
            this_class_cond_red = mutual_information.calculate_conditional_MI(self._X[:, feat1], self._X[:, feat2], self._Y)
            self._class_cond_red[feat1, feat2] = this_class_cond_red
            self._class_cond_red[feat2, feat1] = this_class_cond_red
        return self._class_cond_red[feat1, feat2]


    def __J_JMI(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += (this_redundancy - this_class_cond_red)
            j = relevancy - 1./float(len(features_in_set)) * tmp
        else:
            j = relevancy
        return j

    def __J_CIFE(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += (this_redundancy - this_class_cond_red)
            j = relevancy - tmp
        else:
            j = relevancy
        return j


    def __J_ICAP(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += np.max([0, (this_redundancy - this_class_cond_red)])
            j = relevancy - tmp
        else:
            j = relevancy
        return j


    def __J_CMIM(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmps = []
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmps += [this_redundancy - this_class_cond_red]
            j = relevancy - np.max(tmps)
        else:
            j = relevancy
        return j

    def evaluate_feature(self, features_in_set, feature_to_be_tested):
        return self._method(features_in_set, feature_to_be_tested)

    def run_selection(self, n_features_to_select):
        logger.info("Initialize filter feature selection:")
        logger.info("using filter method: %s"%self._method_str)
        def find_next_best_feature(current_feature_set):
            features_not_in_set = set(np.arange(self._n_features)).difference(set(current_feature_set))
            best_J = -999999.9
            best_feature = None
            for feature_candidate in features_not_in_set:
                j_feature = self.evaluate_feature(current_feature_set, feature_candidate)
                if j_feature > best_J:
                    best_J = j_feature
                    best_feature = feature_candidate
            if best_feature is not None:
                logger.info("Best feature found was %d with J_eval= %f. Feature set was %s"%(best_feature, best_J, str(current_feature_set)))
            return best_feature

        if n_features_to_select > self._n_features:
            raise ValueError("n_features_to_select must be smaller or equal to the number of features")

        selected_features = 0
        current_feature_set = []
        while selected_features < n_features_to_select:
            best_feature = find_next_best_feature(current_feature_set)
            if best_feature is not None:
                current_feature_set += [best_feature]
                selected_features += 1
            else:
                break
        logger.info("Filter feature selection done. Final set is: %s"%str(current_feature_set))
        return np.array(current_feature_set)