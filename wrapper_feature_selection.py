__author__ = 'fabian'

import IPython
import numpy as np
import sklearn
import utils
from sklearn import cross_validation

class EvaluationFunctionRF(object):
    def __init__(self, classifier, k_fold = 5, complexity_penalty = 0.05):
        self._classifier = classifier
        self._k_fold = k_fold
        self._complexity_penalty = complexity_penalty

    @staticmethod
    def kfold_train_and_predict(X, Y, classifier, k = 5, indices = None, features = None):
        if indices is None:
            indices = np.array(range(X.shape[0]))
        if features is None:
            features = np.array(range(X.shape[1]))
        features = np.array(list(features))
        kf = cross_validation.KFold(len(indices), n_folds=k)
        accurs = []
        for train, test in kf:
            train_ind = indices[train].astype("int")
            test_ind = indices[test].astype("int")

            #IPython.embed()
            classifier.fit(X[train_ind,:][:,features], Y[train_ind])
            accurs += [classifier.score(X[test_ind,:][:,features], Y[test_ind])]

        accurs = np.array(accurs)
        return np.mean(accurs), np.std(accurs)

    def evaluate_feature_set_size_penalty(self, X, Y, indices, feature_set):
        accur, stdev = self.kfold_train_and_predict(X, Y, self._classifier, self._k_fold, indices, feature_set)
        score = accur + self._complexity_penalty * (1. - float(len(feature_set))/X.shape[1])
        return score


class FeatureSelection(object):
    def __init__(self, evaluation_function):
        self._evaluation_function = evaluation_function

    def apply_operation_to_feature_set(self, feature_set, feature_id, operation):
        """ Modifies a feature set by adding (operation = 1) or removing (operation = -1) the feature specified by
        feature_id form the feature_set

        :param feature_set:     set of integer values
        :param feature_id:      integer value
        :param operation:       determines the operation that will be performed on the set. 1 for adding, -1 for
                                removal of the id specified by feature_id
        :return:                modified feature_set object
        """
        assert isinstance(feature_set, set)
        assert operation in [-1, 1]
        feature_set = set(feature_set) # make sure not to override anything
        if operation == 1:
            if feature_id in feature_set:
                print("Warning: adding of feature %d: feature is already present in feature set %s"%(feature_id, str(feature_set)))
            else:
                feature_set.add(feature_id)
        else:
            if not feature_id in feature_set:
                print("Warning: removing feature %d: feature is not present in feature set %s"%(feature_id, str(feature_set)))
            else:
                feature_set.remove(feature_id)
        return feature_set

    def sequential_feature_selection(self, X, Y, indices = None, direction = "forward", do_floating_search = True, initial_feature_set = None,
                                     constant_feature_ids = None, feature_search_space = None, overshoot = 3, epsilon = 0.):
        """
        Description here... TODO

        Examples

        :param X:           (n_samples, n_features) numpy array containing the data
        :param Y:           1-d array containing the corresponding labels (length n_samples)
        :param indices:     integer array containing the indices that are used for feature selection.
                            Default value None: all indices will be used
        :param direction:   may be "forward" (sequential forward selection (SFS)) or "backward" (sequential backward
                            elimination (SBE))
        :param do_floating_search:      determines whether floating search methods will be applied. See [Pudil et al.
                                        1994] for more info
        :param initial_feature_set:     set of feature ids to start the search with.
                                        Default value None: SFS: empty feature set
                                                            SBE: full feature set (all features)
        :param constant_feature_ids:    set of feature ids that is always included and will not be modified by the
                                        selection process. Default value None: empty set
        :param feature_search_space:    set of feature ids that specify which feature ids will be searched for
                                        adding/removing features. Default value None: all features
        :param overshoot:   amount of iterations to continue running although no improvement over the evaluation
                            function could be achieved. Increasing this number may help overcome potential local minima.
                            Default value: 3
        :param epsilon:     threshold that determines by how much the evaluation function of a set must improve over the
                            currently best scoring set in order for the new set to be adopted. Default value 0.0
        :return:
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # this whole section is just to check whether all arguments are valid ------------------------------------------
        if n_samples != len(Y):
            raise AttributeError("Y must have the same length as X has rows (n_samples)")

        if indices == None:
            indices = np.arange(n_samples)

        if not ((indices.dtype == np.dtype('int64')) | (indices.dtype == np.dtype('int32'))):
            raise ValueError("indices must be either None or a numpy array of integer values")

        if direction not in ["forward", "backward"]:
            raise ValueError("direction must be either \"forward\" or \"backward\"")

        # here we set the default values for constant_feature_ids, feature_search_space and initial_feature_set
        # depending on the selected search direction -------------------------------------------------------------------
        if constant_feature_ids is None:
            constant_feature_ids = set([])
        if feature_search_space is None:
            feature_search_space = set(list(np.arange(n_features)))

        if direction == "forward":
            if initial_feature_set is None:
                initial_feature_set = set([])
            remaining_features = feature_search_space.difference(initial_feature_set)
            set_operation = 1
        else:
            if initial_feature_set is None:
                initial_feature_set = set(list(np.arange(n_features)))
            remaining_features = set([])
            set_operation = -1

        # check whether the entries of constant_feature_ids, feature_search_space and initial_feature_set are consistent
        # constant_feature_ids cannot be contained in the initial_feature_set
        if len(initial_feature_set.intersection(constant_feature_ids)) != 0:
            raise AttributeError("constant_feature_ids cannot be contained in initial_feature_ids")

        # init feature set must be a subset of the feature search space
        if feature_search_space.intersection(initial_feature_set) != initial_feature_set:
            raise AttributeError("initial_feature_set mus be a subset of feature_search_space")

        # constant features cannot be in the feature_search_space
        if len(feature_search_space.intersection(constant_feature_ids)) != 0:
            raise AttributeError("feature_search_space cannot contain features from constant_feature_ids")

        # score initialization, a higher score is better than a lower one
        if len(initial_feature_set) == 0:
            score_of_current_set = -9999999999.9

        else:
            score_of_current_set = self._evaluation_function(X, Y, indices, initial_feature_set)

        current_features = initial_feature_set
        overall_best_score = score_of_current_set

        overall_best = initial_feature_set
        floating_search_operation = - set_operation

        best_not_changed_in = 0

        #now start the feature selection process
        while (best_not_changed_in < overshoot):
            print("current best feature set")
            print(overall_best)
            score_of_best_feat_to_modify = -9999999999.9
            best_feat_to_modify = None

            # determine which features to look at in this iteration (all features not in current_features (=remaining
            # features) for SFS; all features in current_features for SBE)
            if direction == "forward":
                look_at = set(remaining_features)
            else:
                look_at = set(current_features)

            for i in look_at:
                # modify feature i (set_operation depends on direction=forward/backward) and append constant feature set
                new_feature_set = self.apply_operation_to_feature_set(current_features, i, set_operation)
                new_feature_set = new_feature_set.union(constant_feature_ids)

                if len(new_feature_set) == 0:
                    continue

                # evaluate this set
                score_with_new_set = self._evaluation_function(X, Y, indices, new_feature_set)

                if score_with_new_set > score_of_best_feat_to_modify:
                    best_feat_to_modify = i
                    score_of_best_feat_to_modify = score_with_new_set


            if best_feat_to_modify is not None:
                remaining_features = self.apply_operation_to_feature_set(remaining_features, best_feat_to_modify, floating_search_operation)
                current_features = self.apply_operation_to_feature_set(current_features, best_feat_to_modify, set_operation)
                just_modified_feature = best_feat_to_modify
                score_of_current_set = score_of_best_feat_to_modify

                print("curr set is now:")
                print(current_features)

                # the whole part here is for the floating search [Pudil et al 1994]. It is only accessed if adding/removing
                # a feature did improve the evaluation function in the previous step
                if score_of_current_set > overall_best_score:
                    # only actually do this if do_floating_search is TRUE
                    continue_to_float_search = do_floating_search

                    # if forward selection then curr set must not be empty
                    if (direction == "forward") & (len(current_features) < 2):
                        continue_to_float_search = False
                    # if backward selection then remaining features cannot be empty
                    if (direction == "backward") & (len(remaining_features) < 2):
                        continue_to_float_search = False

                    if continue_to_float_search:
                        continue_float_search = True

                        # now add/remove features to/from the set as long as it improves the evaluation function
                        while continue_float_search:
                            print("floating search: ")
                            best_feat_to_modify = None
                            best_feat_to_modify_score = -99999.0

                            if direction == "forward":
                                look_at = self.apply_operation_to_feature_set(current_features, just_modified_feature, -1)
                            else:
                                look_at = self.apply_operation_to_feature_set(remaining_features, just_modified_feature, -1)
                            for i in look_at:
                                new_feature_set = self.apply_operation_to_feature_set(current_features, i, floating_search_operation)
                                if len(new_feature_set) > 0:
                                    print new_feature_set
                                    score_with_new_feature_set = self._evaluation_function(X, Y, indices, new_feature_set)

                                    if score_with_new_feature_set > best_feat_to_modify_score:
                                        best_feat_to_modify = i
                                        best_feat_to_modify_score = score_with_new_feature_set
                            print("best floating search score: %f"%best_feat_to_modify_score)
                            if (best_feat_to_modify_score > score_of_current_set):
                                print("updated feature set thanks to float search: ")
                                remaining_features = self.apply_operation_to_feature_set(remaining_features, best_feat_to_modify, -floating_search_operation)
                                current_features = self.apply_operation_to_feature_set(current_features, best_feat_to_modify, floating_search_operation)
                                score_of_current_set = best_feat_to_modify_score
                                print(current_features)
                                if (direction == "forward") & (len(current_features) < 1):
                                    continue_float_search = False
                                if (direction == "backward") & (len(remaining_features) < 1):
                                    continue_float_search = False
                            else:
                                continue_float_search = False
            print("local best score is %f, overall best score is %f"%(score_of_current_set, overall_best_score))
            if score_of_current_set > (overall_best_score - epsilon):
                overall_best_score = score_of_current_set
                best_not_changed_in = 0
                overall_best = current_features
            else:
                best_not_changed_in += 1
                print("best set has not changed in %d iterations" % best_not_changed_in)

        return np.sort(list(overall_best)).astype("int"), overall_best_score




