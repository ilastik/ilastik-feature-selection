__author__ = 'fabian'

import IPython
import numpy as np
import sklearn
from sklearn import cross_validation
import logging

logger = logging.Logger('wrapper_feature_selection')
logger.setLevel(logging.DEBUG)

fhandler = logging.FileHandler('wrapper_log.txt', 'w')

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)


class EvaluationFunction(object):
    def __init__(self, classifier, k_fold=5, complexity_penalty=0.05):
        self._classifier = classifier
        self._k_fold = k_fold
        self._complexity_penalty = complexity_penalty

    @staticmethod
    def kfold_train_and_predict(X, Y, classifier, k=5, indices=None, features=None):
        """
        Performs a k-fold cross-validation on the data and returns the average accuracy on the test set as well as its
        standard deviation

        :param X: (n_samples by n_features) numpy array containing the data
        :param Y: (n_samples) numpy array containing the labels (as integer values)
        :param classifier:  classifier instance. Must have classifier.fit() and classifier.score() functions.
                            For Example:
                            classifier = sklearn.ensemble.RandomForestClassifier()
        :param k: number of cross-validations to perform on the data
        :param indices: sample indices to use for the cross-validation. Default (None) uses all samples
        :param features: features to use for the cross-validation. Default (None) uses all features
        :return: returns tuple of mean accuracy and standard deviation of the accuracy across the cross-validation runs
        """
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
        """
        Evaluation function used for the FeatureSelection class. It balances the accuracy achieved with a set with the
        size of the set

        :param X: (n_samples by n_features) numpy array containing the data
        :param Y: (n_samples) numpy array containing the labels (as integer values)
        :param indices: sample indices to use for the evalutation of the set
        :param feature_set: the feature ids of the features in the set (as numpy array)
        :return: score value (higher is better)
        """
        accur, stdev = self.kfold_train_and_predict(X, Y, self._classifier, self._k_fold, indices, feature_set)
        score = accur + self._complexity_penalty * (1. - float(len(feature_set))/X.shape[1])
        return score


class WrapperFeatureSelection(object):
    def __init__(self, X, Y, evaluation_function, method="SFS"):
        """
        This class performs wrapper feature selection. It requires an evaluation function for evaluating feature sets

        :param X:                   (n_samples by n_features) numpy array containing the data
        :param Y:                   (n_samples) numpy array containing target labels
        :param evaluation_function: must have interface evaluation_function(X, Y, indices, feature_set). Can
                                    theoretically be anything but it makes sense to use a k-fold cross-validation using
                                    the desired classifier and the feature_set on the samples indicated ny indices. The
                                    evaluation score may use the test set accuracy of the cross-validation runs or a
                                    related measure (f.ex.: accuracy penalized by feature set size). See
                                    EvaluationFunction.evaluate_feature_set_size_penalty as an example
        :param method:              Determines the search method that is applied:
                                    - "SFS":  sequential forward selection. Start either with an empty or a user defined
                                            feature set and sequentially add the feature that maximises the evaluation
                                            function to that set.
                                    - "SBE":  sequential backward elimination: Start either with a set contianing all
                                            features or a user defined feature set and sequentially eliminate the
                                            feature whose elimination yields the highest score using the evaluation
                                            function.
                                    - "BFS":  priority queue-like search algorithm. The initial (default:empty) feature
                                            set is expanded to all yet undiscovered children by adding/removing single
                                            features. All children and their score are saved. In each iteration, the
                                            child with the highest score is selected and expanded.
        """
        if X.shape[0] != len(Y):
            raise ValueError("X and Y must have have same amount of samples")
        self._X = X
        self._Y = Y
        self._evaluation_function = evaluation_function
        self.change_method(method)

    def change_method(self, method):
        """
        :param method:      Determines the search method that is applied:
                            - "SFS":  sequential forward selection. Start either with an empty or a user defined
                                    feature set and sequentially add the feature that maximises the evaluation
                                    function to that set.
                            - "SBE":  sequential backward elimination: Start either with a set contianing all
                                    features or a user defined feature set and sequentially eliminate the
                                    feature whose elimination yields the highest score using the evaluation
                                    function.
                            - "BFS":  priority queue-like search algorithm. The initial (default:empty) feature
                                    set is expanded to all yet undiscovered children by adding/removing single
                                    features. All children and their score are saved. In each iteration, the
                                    child with the highest score is selected and expanded.
        """
        if method not in ["SFS", "SBE", "BFS"]:
            raise ValueError("method can only be either SFS (sequential forward selection), SBE (sequential backward elimination) or BFS (best first search)")
        self._method = method

    def __apply_operation_to_feature_set(self, feature_set, feature_id, operation):
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
                logger.warning("Warning: adding of feature %d: feature is already present in feature set %s"%(feature_id, str(feature_set)))
            else:
                feature_set.add(feature_id)
        else:
            if not feature_id in feature_set:
                logger.warning("Warning: removing feature %d: feature is not present in feature set %s"%(feature_id, str(feature_set)))
            else:
                feature_set.remove(feature_id)
        return feature_set

    def run(self, **kwargs):
        """
        Runs the wrapper feature selection.
        :param kwargs:
            indices=None:               integer array containing the indices that are used for feature selection.
                                        Default value None: all indices will be used
            do_advanced_search=False:   SFS/SBE : enables/disables floating search (see Pudil1994). Floating search will
                                        allow a more dynamic search and eventually yield better feature sets at the cost
                                        of a higher runtime
                                        BFS : enables/disables compound operators (see Kohavi1997). If your dataset has
                                        few redundant features then enabling these will speed up the feature selection
            initial_features=None:      set of feature ids to start the search with.
                                        Default value None: SFS: empty feature set
                                                            SBE: full feature set (all features)
                                                            BFS: empty feature set
            mandatory_features=None:    set of feature ids that is always included and will not be modified by the
                                        selection process.
                                        Default value None: empty set
                                        (Explanation: Say you are working with MRT data (image processing, many
                                        different channels, each channel is acquired separately -> acquisition of each
                                        additional channel is expensive, task is to identify which channels are required
                                        to perform a certain task). There are mandatory channels by law. Therefore it
                                        is desirable to find the best feature (channel) set given the channels that you
                                        will have to acquire anyways.)
            permitted_features=None:    set of feature ids that specify which feature ids will be searched for
                                        adding/removing features.
                                        Default value None: all features
                                        (one could easily crop the X array instead of using this parameter but using
                                        permitted_features instead has two advantages: 1) it avoids copying of the data,
                                        2) feature ID correspondences are trivial (in the cropped dataset feature
                                        IDs 2, 3, 6 correspond to different features than in the uncropped dataset))
            overshoot=3:                amount of iterations to continue running although no improvement over the evaluation
                                        function could be achieved. Increasing this number may help overcome potential local minima.
                                        Default value: 3
            epsilon=0.:                 threshold that determines by how much the evaluation function of a set must improve over the
                                        currently best scoring set in order for the new set to be adopted. Default value 0.0
        :return: tuple consisting of the best found feature set and the value of its corresponding evaluation function

        """
        if self._method == "SFS":
            return self.__sequential_feature_selection(direction="forward", **kwargs)
        elif self._method == "SBE":
            return self.__sequential_feature_selection(direction="backward", **kwargs)
        else:
            return self.__best_first_search(**kwargs)

    def __sequential_feature_selection(self, indices=None, direction="forward", do_advanced_search=False, initial_features=None,
                                     mandatory_features=None, permitted_features=None, overshoot=3, epsilon=0.):
        n_features = self._X.shape[1]
        n_samples = self._X.shape[0]

        # this whole section is just to check whether all arguments are valid ------------------------------------------
        if n_samples != len(self._Y):
            raise AttributeError("Y must have the same length as X has rows (n_samples)")

        if indices == None:
            indices = np.arange(n_samples)

        if not ((indices.dtype == np.dtype('int64')) | (indices.dtype == np.dtype('int32'))):
            raise ValueError("indices must be either None or a numpy array of integer values")

        if direction not in ["forward", "backward"]:
            raise ValueError("direction must be either \"forward\" or \"backward\"")

        # here we set the default values for constant_feature_ids, feature_search_space and initial_feature_set
        # depending on the selected search direction -------------------------------------------------------------------
        if mandatory_features is None:
            mandatory_features = set([])
        if permitted_features is None:
            permitted_features = set(list(np.arange(n_features))).difference(mandatory_features)

        if direction == "forward":
            if initial_features is None:
                initial_features = set([])
            remaining_features = permitted_features.difference(initial_features)
            set_operation = 1
        else:
            if initial_features is None:
                initial_features = set(permitted_features.difference(mandatory_features))
            remaining_features = set([])
            set_operation = -1

        # check whether the entries of constant_feature_ids, feature_search_space and initial_feature_set are consistent
        # constant_feature_ids cannot be contained in the initial_feature_set
        if len(initial_features.intersection(mandatory_features)) != 0:
            raise AttributeError("constant_feature_ids cannot be contained in initial_feature_ids")

        # init feature set must be a subset of the feature search space
        if permitted_features.intersection(initial_features) != initial_features:
            raise AttributeError("initial_feature_set must be a subset of feature_search_space")

        # constant features cannot be in the feature_search_space
        if len(permitted_features.intersection(mandatory_features)) != 0:
            raise AttributeError("feature_search_space cannot contain features from constant_feature_ids")

        # score initialization, a higher score is better than a lower one
        if len(initial_features) == 0:
            score_of_current_set = float("-inf")

        else:
            score_of_current_set = self._evaluation_function(self._X, self._Y, indices, initial_features)

        current_features = initial_features
        overall_best_score = score_of_current_set

        overall_best = initial_features
        floating_search_operation = - set_operation

        best_not_changed_in = 0

        #now start the feature selection process
        while (best_not_changed_in <= overshoot):
            logger.info("current best feature set %s", str(overall_best))
            score_of_best_feat_to_modify = float("-inf")
            best_feat_to_modify = None

            # determine which features to look at in this iteration (all features not in current_features (=remaining
            # features) for SFS; all features in current_features for SBE)
            if direction == "forward":
                look_at = set(remaining_features)
            else:
                look_at = set(current_features)

            for i in look_at:
                # modify feature i (set_operation depends on direction=forward/backward) and append constant feature set
                new_feature_set = self.__apply_operation_to_feature_set(current_features, i, set_operation)
                new_feature_set = new_feature_set.union(mandatory_features)

                if len(new_feature_set) == 0:
                    continue

                # evaluate this set
                score_with_new_set = self._evaluation_function(self._X, self._Y, indices, new_feature_set)

                if score_with_new_set > score_of_best_feat_to_modify:
                    best_feat_to_modify = i
                    score_of_best_feat_to_modify = score_with_new_set


            if best_feat_to_modify is not None:
                remaining_features = self.__apply_operation_to_feature_set(remaining_features, best_feat_to_modify, floating_search_operation)
                current_features = self.__apply_operation_to_feature_set(current_features, best_feat_to_modify, set_operation)
                just_modified_feature = best_feat_to_modify
                score_of_current_set = score_of_best_feat_to_modify

                logger.info("curr set is now: %s", str(current_features))

                # the whole part here is for the floating search [Pudil et al 1994]. It is only accessed if adding/removing
                # a feature did improve the evaluation function in the previous step
                if score_of_current_set > overall_best_score:
                    # only actually do this if do_floating_search is TRUE
                    continue_to_float_search = do_advanced_search

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
                            logger.info("floating search: ")
                            best_feat_to_modify = None
                            best_feat_to_modify_score = float("-inf")

                            if direction == "forward":
                                look_at = self.__apply_operation_to_feature_set(current_features, just_modified_feature, -1)
                            else:
                                look_at = self.__apply_operation_to_feature_set(remaining_features, just_modified_feature, -1)
                            for i in look_at:
                                new_feature_set = self.__apply_operation_to_feature_set(current_features, i, floating_search_operation)
                                new_feature_set = new_feature_set.union(mandatory_features)
                                if len(new_feature_set) > 0:
                                    #print new_feature_set
                                    score_with_new_feature_set = self._evaluation_function(self._X, self._Y, indices, new_feature_set)

                                    if score_with_new_feature_set > best_feat_to_modify_score:
                                        best_feat_to_modify = i
                                        best_feat_to_modify_score = score_with_new_feature_set
                            logger.info("best floating search score: %f"%best_feat_to_modify_score)
                            if (best_feat_to_modify_score > score_of_current_set):
                                remaining_features = self.__apply_operation_to_feature_set(remaining_features, best_feat_to_modify, -floating_search_operation)
                                current_features = self.__apply_operation_to_feature_set(current_features, best_feat_to_modify, floating_search_operation)
                                score_of_current_set = best_feat_to_modify_score
                                logger.info("updated feature set thanks to float search: %s", str(current_features))
                                if (direction == "forward") & (len(current_features) < 1):
                                    continue_float_search = False
                                if (direction == "backward") & (len(remaining_features) < 1):
                                    continue_float_search = False
                            else:
                                continue_float_search = False
            logger.info("local best score is %f, overall best score is %f"%(score_of_current_set, overall_best_score))
            if score_of_current_set > (overall_best_score - epsilon):
                overall_best_score = score_of_current_set
                best_not_changed_in = 0
                overall_best = current_features.union(mandatory_features)
            else:
                best_not_changed_in += 1
                logger.info("best set has not changed in %d iterations" % best_not_changed_in)

        return np.sort(list(overall_best)).astype("int"), overall_best_score

    def __best_first_search(self, indices=None, do_advanced_search=False, initial_features=None,
                          mandatory_features=None, permitted_features=None, overshoot=3, epsilon=0.):

        n_samples, n_features = self._X.shape

        def expand_node(node, open_list, closed_list, n_features):
            children = []
            for feature in node:
                new_child = set(node)
                new_child.remove(feature)
                if (not new_child in open_list) and (not new_child in closed_list) and (len(new_child) > 0):
                    children += [new_child]

            features_not_in_node = set(permitted_features).symmetric_difference(node)
            for feature in features_not_in_node:
                new_child = set(node)
                new_child.add(feature)
                if (not new_child in open_list) and (not new_child in closed_list):
                    children += [new_child]
            return children

        def obtain_scores_of_children(children, indices):
            scores = []
            for child in children:
                scores += [self._evaluation_function(self._X, self._Y, indices, np.array(list(child.union(mandatory_features))))]
            return scores

        def pick_next_node(open_list, open_scores, closed_list):
            id_of_best_node = np.argmax(open_scores)
            node = open_list.pop(id_of_best_node)
            open_scores.pop(id_of_best_node)
            #IPython.embed()
            closed_list += [node]
            return node, open_list, open_scores, closed_list #one could slolve this in a much more elegant way if python
            # would allow explicit use of pointers

        # this whole section is just to check whether all arguments are valid ------------------------------------------
        if n_samples != len(self._Y):
            raise AttributeError("Y must have the same length as X has rows (n_samples)")

        if indices == None:
            indices = np.arange(n_samples)

        if not ((indices.dtype == np.dtype('int64')) | (indices.dtype == np.dtype('int32'))):
            raise ValueError("indices must be either None or a numpy array of integer values")

        # here we set the default values for constant_feature_ids, feature_search_space and initial_feature_set
        # depending on the selected search direction -------------------------------------------------------------------
        if mandatory_features is None:
            mandatory_features = set([])
        if permitted_features is None:
            permitted_features = set(list(np.arange(n_features))).difference(mandatory_features)
        if initial_features is None:
            initial_features = set([])

        # check whether the entries of constant_feature_ids, feature_search_space and initial_feature_set are consistent
        # constant_feature_ids cannot be contained in the initial_feature_set
        if len(initial_features.intersection(mandatory_features)) != 0:
            raise AttributeError("constant_feature_ids cannot be contained in initial_feature_ids")

        # init feature set must be a subset of the feature search space
        if permitted_features.intersection(initial_features) != initial_features:
            raise AttributeError("initial_feature_set mus be a subset of feature_search_space")

        # constant features cannot be in the feature_search_space
        if len(permitted_features.intersection(mandatory_features)) != 0:
            raise AttributeError("feature_search_space cannot contain features from constant_feature_ids")

        # score initialization, a higher score is better than a lower one
        if len(initial_features) == 0:
            score_of_current_set = float("-inf")

        else:
            score_of_current_set = self._evaluation_function(self._X, self._Y, indices, initial_features)

        # initialize open and closed lists
        open_list = [initial_features]
        open_scores = [score_of_current_set]
        closed_list = []


        best_set = initial_features
        score_of_best_set = score_of_current_set

        best_not_changed_in = 0
        while (best_not_changed_in <= overshoot):
            logger.info("current best set: %s with score %f"%(str(best_set), score_of_best_set))
            # - retrieve the best node from the open list,
            # - remove the corresponding entry from the open_list and open_scores,
            # - add the retrieved node to the closed list
            next_node, open_list, open_scores, closed_list = pick_next_node(open_list, open_scores, closed_list)

            # - find all valid expansions of that node (search feature search space for adding features; remove each
            # feature in turn form the node)
            # - valid expansions are those that result in nodes which are not already in the either the open_list
            # or closed_list
            new_children = expand_node(next_node, open_list, closed_list, n_features)

            # calculate the evaluation function for all children
            new_scores = obtain_scores_of_children(new_children, indices)

            # add all children and their score to the respective lists
            open_list += new_children
            open_scores += new_scores

            if len(new_scores) == 0: # if there are only few features (iris dataset) then there may be no valid
            # expansions to a node. In that case jump to the next best node
                best_not_changed_in += 1
                logger.info("The feature set has not been updated in the last %d iterations"%best_not_changed_in)
                continue

            id_of_best_child = np.argmax(new_scores)
            continue_compound = False

            # update best set if such a set is found
            if new_scores[id_of_best_child] > (score_of_best_set + epsilon):
                best_set = new_children.pop(id_of_best_child)
                score_of_best_set = new_scores.pop(id_of_best_child)
                best_not_changed_in = 0
                continue_compound = True
                logger.info("updated best feature set: %s \t score: %f"%(str(best_set), score_of_best_set))
            else:
                best_not_changed_in += 1
                logger.info("The feature set has not been updated in the last %d iterations"%best_not_changed_in)

            # continue only to compound search if 1) it has been activated by the user, 2) the best_set has been updated
            # AND 3) there was more than one child in the new_children list (>0 because one child has already been
            # popped form the list)
            while(do_advanced_search & continue_compound & (len(new_scores) > 0)):
                # find second best set (best one has already been removed)
                id_of_best_child = np.argmax(new_scores)
                best_child = new_children.pop(id_of_best_child)
                best_child_score = new_scores.pop(id_of_best_child)

                #find out operation that led to child (f. ex: '+ feature 5' or '- feature 3')
                modified_feature = best_child.symmetric_difference(next_node)
                if len(best_child) < len(next_node):
                    operation = -1
                else:
                    operation = +1

                # create new child with compound operators
                compound_child = self.__apply_operation_to_feature_set(best_set, list(modified_feature)[0], operation)

                if len(compound_child) < 1:
                    break
                if (compound_child in open_list) or (compound_child in closed_list):
                    break

                # if compound_child is valid then evaluate it and add it to the lists
                score_of_compound_child = self._evaluation_function(self._X, self._Y, indices, compound_child)

                open_list += [compound_child]
                open_scores += [score_of_compound_child]

                if score_of_compound_child > (score_of_best_set + epsilon):
                    best_set = compound_child
                    score_of_best_set = score_of_compound_child
                    logger.info("updated best node thanks to compound operators")
                else:
                    continue_compound = False

        return np.sort(list(best_set.union(mandatory_features))), score_of_best_set