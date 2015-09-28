__author__ = 'fabian'

import sys
sys.path.append("../")
import numpy as np
import ctypes as c
import mutual_information_old
import utils
import unittest
import logging


logger = logging.Logger('MI_testing')
logger.setLevel(logging.DEBUG)

fhandler = logging.FileHandler('MI_testing_log.txt', 'w')

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)

logger.addHandler(fhandler)

MI_Toolbox = c.CDLL("libMIToolbox.so")

class TestMutualInformation(unittest.TestCase):
    def __calculate_MI(self, data_0, data_1):
        assert(data_0.size == data_1.size)

        # cast as C types
        c_vector_length = c.c_int(data_0.size)

        data_0 = np.array(data_0, order="C")
        data_1 = np.array(data_1, order="C")

        data_0 = data_0.astype("float64")
        data_1 = data_1.astype("float64")

        c_data_0 = data_0.ctypes.data_as(c.POINTER(c.c_double))
        c_data_1 = data_1.ctypes.data_as(c.POINTER(c.c_double))


        MI_Toolbox.calculateMutualInformation.restype = c.c_double
        mutual_information = MI_Toolbox.calculateMutualInformation(
            c_data_0,
            c_data_1,
            c_vector_length
            )

        return mutual_information

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


    def test_mutual_info_calculation(self):
        X, Y = utils.load_digits()
        X = mutual_information_old.normalize_data_for_MI(X)
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                X1 = np.array(X[:,i])
                X2 = np.array(X[:,j])
                mi_pyfeast = self.__calculate_MI(X1, X2)
                mi_ours = mutual_information_old.calculate_mutual_information_histogram_binning(X1, X2)
                logger.debug("mutual_info: features: %d and %d; pyfeast: %f \t ours: %f"%(i, j, mi_pyfeast, mi_ours))
                self.assertTrue(np.isclose(mi_pyfeast, mi_ours, rtol=0.01, atol=0.00001))

    def test_cond_mutual_info(self):
        X, Y = utils.load_digits()
        Y = Y.astype("int")
        X = mutual_information_old.normalize_data_for_MI(X)
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                X1 = np.array(X[:,i])
                X2 = np.array(X[:,j])

                cond_mi_pyfeast = self.__calculate_cond_MI(X1, X2, Y)
                cond_mi_ours = mutual_information_old.calculate_conditional_MI(X1, X2, Y)
                logger.debug("class cond mutual info: features: %d and %d; pyfeast: %f \t ours: %f"%(i, j, cond_mi_pyfeast, cond_mi_ours))
                self.assertTrue(np.isclose(cond_mi_pyfeast, cond_mi_ours, rtol=0.01, atol=0.00001))

if __name__ == "__main__":
    unittest.main()