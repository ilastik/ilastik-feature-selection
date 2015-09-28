__author__ = 'fabian'
import sys
sys.path.append('../')
sys.path.append("../mutual_information")
import utils
import filter_feature_selection
import numpy as np
import feast
import mutual_information_old
import unittest
import logging
import ctypes as c
import IPython

logger = logging.Logger('Filter_Selection_testing')
logger.setLevel(logging.DEBUG)

fhandler = logging.FileHandler('Filter_Selection_testing_log.txt', 'w')

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)
MI_Toolbox = c.CDLL("libMIToolbox.so")


class TestFilterFeatureSelection(unittest.TestCase):
    def __calculate_cond_MI(self, data_0, data_1, cond_vec):
        assert(data_0.size == data_1.size)
        assert(data_0.size == cond_vec.size)


        # cast as C types
        c_vector_length = c.c_int(data_0.size)

        data_0 = np.array(data_0, order="C")
        data_1 = np.array(data_1, order="C")
        cond_vec = np.array(cond_vec, order="C")

        data_0 = data_0.astype("float64")
        data_1 = data_1.astype("float64")
        cond_vec = cond_vec.astype("float64")

        c_data_0 = data_0.ctypes.data_as(c.POINTER(c.c_double))
        c_data_1 = data_1.ctypes.data_as(c.POINTER(c.c_double))
        c_cond_vec = cond_vec.ctypes.data_as(c.POINTER(c.c_double))


        MI_Toolbox.calculateConditionalMutualInformation.restype = c.c_double
        cond_mutual_information = MI_Toolbox.calculateConditionalMutualInformation(
            c_data_0,
            c_data_1,
            c_cond_vec,
            c_vector_length
            )
        return cond_mutual_information

    def test_class_cond_mi_calculation(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "CIFE")
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                X1 = np.array(X[:,i])
                X2 = np.array(X[:,j])

                cond_mi_pyfeast = self.__calculate_cond_MI(X1, X2, Y)
                cond_mi_ours = selector._calculate_class_conditional_MI(X1.astype("int"), X2.astype("int"), Y)
                logger.debug("class cond mutual info: features: %d and %d; pyfeast: %f \t ours: %f"%(i, j, cond_mi_pyfeast, cond_mi_ours))
                self.assertTrue(np.isclose(cond_mi_pyfeast, cond_mi_ours, rtol=0.01, atol=0.00001))

    def test_CIFE(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "CIFE")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.CIFE(X, Y, num_feat)).astype("int")
        logger.debug("CIFE")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))


    def test_JMI(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "JMI")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.JMI(X, Y, num_feat)).astype("int")
        logger.debug("JMI")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))

    def test_ICAP(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "ICAP")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.ICAP(X, Y, num_feat)).astype("int")
        logger.debug("ICAP")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))

    def test_CMIM(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "CMIM")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.CMIM(X, Y, num_feat)).astype("int")
        logger.debug("CMIM")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))

    def test_MIFS(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "MIFS")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.BetaGamma(X, Y, num_feat, 1.0, 0.0)).astype("int") # MIFS in feast is buggy
                                                                                    # (beta=0 although it should be =1)
        logger.debug("MIFS")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))


    def test_mRMR(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "mRMR")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.mRMR(X, Y, num_feat)).astype("int")
        logger.debug("mRMR")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))

    def test_change_method(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        X = X.astype("float64")
        selector = filter_feature_selection.FilterFeatureSelection(X, Y, "mRMR")
        num_feat = 10
        our_set = selector.run(num_feat)
        feast_set = np.array(feast.ICAP(X, Y, num_feat)).astype("int")
        selector.change_method("ICAP")
        our_set = selector.run(num_feat)
        logger.debug("mRMR changed to ICAP")
        logger.debug("ours\t%s"%str(our_set))
        logger.debug("feast\t%s"%str(feast_set))
        self.assertEqual(set(our_set), set(feast_set))

if __name__ == "__main__":
    unittest.main()