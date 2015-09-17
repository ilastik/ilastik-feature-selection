__author__ = 'fabian'
import mutual_information

class FilterEvaluationFunction(object):
    def __init__(self, mi_method):
        self._mi_method = mi_method

    def J_CIFE(self, features_in_set, feature_to_be_tested, data, target):
        relevancy = mutual_information.calculate_mutual_information_histogram_binning(data[:,feature_to_be_tested], target)
        tmp = 0.
        for feature in features_in_set:
            redundancy = mutual_information.calculate_mutual_information_histogram_binning(data[:, feature], data[:, feature_to_be_tested])
            class_cond_red = mutual_information.calculate_conditional_MI(data[:, feature], data[:, feature_to_be_tested], target)
            tmp += (redundancy - class_cond_red)
        j_cife = relevancy - 1/float(len(features_in_set)) * tmp
        return j_cife

    def J_ICAP(self, features_in_set, feature_to_be_tested, data, target):
        raise NotImplementedError

    def J_CMIM(self, features_in_set, feature_to_be_tested, data, target):
        raise NotImplementedError


class FilterFeatureSelection(object):
    def __init__(self, method):
        if method not in ["CIFE", "ICAP", "CMIM"]:
            raise ValueError("method must be one of the following: CIFE, ICAP, CMIM")
        