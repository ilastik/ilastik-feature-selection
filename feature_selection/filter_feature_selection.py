__author__ = 'fabian'

import mutual_information

import numpy as np
import logging
import IPython

logger = logging.getLogger(__name__)
# logger = logging.Logger('filter_feature_selection')
# logger.setLevel(logging.DEBUG)
#
# fhandler = logging.FileHandler('filter_log.txt', 'w')
#
# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# fhandler.setFormatter(formatter)
#
# logger.addHandler(fhandler)

class FilterFeatureSelection(object):
    def __init__(self, X, Y, method="ICAP"):
        """
        This class provides easy access to mutual information based filter feature selection.
        The default mutual information estimation algorithm used is the histogram binning method. If a more
        sophisticated approach is required, use the change_MI_estimator function to apply your own method.

        :param X: (n_samples, n_features) numpy array containing the training data
        :param Y: (n_samples) numpy array containing target labels
        :param method: filter criterion that will be applied to select the features. Available criteria are: (as string)
                       "CIFE" [Lin1996], "ICAP" [Jakulin2005], "CMIM" [Fleuret2004], "JMI"[Yang1999]
        """
        if X.shape[0] != len(Y):
            raise ValueError("X must have as many samples as there are labels in Y")

        self._n_features = X.shape[1]
        self._X = np.array(X)
        self._Y = np.array(Y)
        self._method_str = method
        self._methods = {
            "CIFE": self.__J_CIFE,
            "ICAP": self.__J_ICAP,
            "CMIM": self.__J_CMIM,
            "JMI": self.__J_JMI,
            "mRMR": self.__J_mRMR,
            "MIFS": self.__J_MIFS
        }
        self._mi_method_kwargs = {}
        self._filter_criterion_kwargs = {}
        self.change_method(method)
        self._method = self._methods[method]
        self._mutual_information_estimator = mutual_information.calculate_mutual_information

        self._redundancy = np.zeros((self._n_features, self._n_features)) - 1.
        self._relevancy = np.zeros((self._n_features)) - 1
        self._class_cond_red = np.zeros((self._n_features, self._n_features)) - 1
        self._class_cond_mi_method = self._calculate_class_conditional_MI

    def change_MI_estimator(self, mi_estimator, estimator_kwargs={}):
        """

        :param mi_estimator: mutual information estimator. Inferface: mutual_information = mi_estimator(X1, X2,
                             **estimator_kwargs) with X1 and X2 (size: n_samples) being numpy arrays containing the
                             values of one feature
        :param estimator_kwargs: kwargs of the chosen estimator (dictionary)
        """
        self._mutual_information_estimator = mi_estimator
        self._mi_method_kwargs = estimator_kwargs


    def change_method(self, method, **method_kwargs):
        """
        Changes the filter criterion which is used to select the features

        :param method: string indicating the desired criterion
        """
        if method not in self._methods.keys():
            raise ValueError("method must be one of the following: %s"%str(self._methods.keys()))
        self._method = self._methods[method]
        self._method_str = method
        self._filter_criterion_kwargs = method_kwargs

    def get_current_method(self):
        """
        Prints the currently selected criterion
        """
        print self._method

    def get_available_methods(self):
        """
        Returns the implemented criteria as strings
        :return: list of strings containing the implemented criteria
        """
        return self._methods.keys()

    def _calculate_class_conditional_MI(self, X1, X2, Y):
        states = np.unique(Y)
        con_mi = 0.

        for state in states:
            indices = (Y == state)
            p_state = float(np.sum(indices)) / float(len(Y))
            mi = self._mutual_information_estimator(X1[indices], X2[indices], **self._mi_method_kwargs)
            con_mi += p_state * mi
        return con_mi

    def _change_cmi_method(self, method):
        """
        Do not use this

        :param method: Seriously. Don't. Its for some testing purposes
        :return:
        """
        self._class_cond_mi_method = method

    def _get_relevancy(self, feat_id):
        if self._relevancy[feat_id] == -1:
            self._relevancy[feat_id] = self._mutual_information_estimator(self._X[:, feat_id], self._Y,
                                                                          **self._mi_method_kwargs)
        return self._relevancy[feat_id]

    def _get_redundancy(self, feat1, feat2):
        if self._redundancy[feat1, feat2] == -1:
            this_redundancy = self._mutual_information_estimator(self._X[:, feat1], self._X[:, feat2],
                                                                 **self._mi_method_kwargs)
            self._redundancy[feat1, feat2] = this_redundancy
            self._redundancy[feat2, feat1] = this_redundancy
        return self._redundancy[feat1, feat2]

    def _get_class_cond_red(self, feat1, feat2):
        if self._class_cond_red[feat1, feat2] == -1:
            this_class_cond_red = self._class_cond_mi_method(self._X[:, feat1], self._X[:, feat2], self._Y)
            self._class_cond_red[feat1, feat2] = this_class_cond_red
            self._class_cond_red[feat2, feat1] = this_class_cond_red
        return self._class_cond_red[feat1, feat2]

    def __J_MIFS(self, features_in_set, feature_to_be_tested, beta=1):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                tmp += this_redundancy
            j = relevancy - beta * tmp
        else:
            j = relevancy
        return j

    def __J_mRMR(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevancy(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                tmp += this_redundancy
            j = relevancy - 1./float(len(features_in_set)) * tmp
        else:
            j = relevancy
        return j

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

    def _evaluate_feature(self, features_in_set, feature_to_be_tested):
        return self._method(features_in_set, feature_to_be_tested, **self._filter_criterion_kwargs)

    def run(self, n_features_to_select):
        """
        Performs the actual feature selection using the specified filter criterion

        :param n_features_to_select: number of features to select
        :return: numpy array of selected features (as IDs)
        """
        logger.info("Initialize filter feature selection:")
        logger.info("using filter method: %s"%self._method_str)
        if self._mutual_information_estimator == mutual_information.calculate_mutual_information:
            def normalize_data_for_MI(X):
                for i in xrange(X.shape[1]):
                    std = X[:, i].std()
                    if std != 0.:
                        X[:, i] /= std
                        X[:, i] -= X[:, i].min()
                return np.floor(X).astype("int")
            old_data = np.array(self._X)
            self._X = normalize_data_for_MI(self._X)
        def find_next_best_feature(current_feature_set):
            features_not_in_set = set(np.arange(self._n_features)).difference(set(current_feature_set))
            best_J = -999999.9
            best_feature = None
            for feature_candidate in features_not_in_set:
                j_feature = self._evaluate_feature(current_feature_set, feature_candidate)
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
        if self._mutual_information_estimator == mutual_information.calculate_mutual_information:
            self._X = old_data

        return np.array(current_feature_set)


# Francois Fleuret. Fast Binary Feature Selection with Conditional Mutual Informa-
# tion. Journal of Machine Learning Research,

# Aleks Jakulin. Machine Learning Based on Attribute Interactions. PhD Thesis,
# University of Ljubljana, Slovenia, 2005.

# D Lin and X Tang. Conditional infomax learning: An integrated framework for
# feature extraction and fusion. In European Conference on Computer Vision, 1996.

# Howard Hua Yang and John Moody. Feature selection based on joint mutual infor-
# mation. Proceedings of International ICSC Symposium on Advances in Intelligent
# Data Analysis, pages 22 25, 1999.

# Roberto Battiti. Using mutual information for selecting features in supervised neural
# net learning. IEEE Transactions on Neural Networks, 5(4):537 550, 1994.

# H C Peng, F H Long, and C Ding. Feature selection based on mutual information:
# Criteria of max-dependency, max-relevance, and min-redundancy. Ieee Transactions
# on Pattern Analysis and Machine Intelligence, 27(8):1226 1238, 2005.
