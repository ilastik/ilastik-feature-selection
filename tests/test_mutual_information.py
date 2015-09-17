__author__ = 'fabian'


import numpy as np
import ctypes as c
import mutual_information
import utils

MI_Toolbox = c.CDLL("libMIToolbox.so");

# double calculateMutualInformation(double *dataVector, double *targetVector, int vectorLength)
def calculate_MI(data_0, data_1):
    assert(data_0.size == data_1.size)

    # normalize data
    data_0 -= np.min(data_0)
    if data_0.std() != 0:
        data_0 /= np.std(data_0)
    data_1 -= np.min(data_1)
    if data_1.std() != 0:
        data_1 /= np.std(data_1)

    data_0 = np.floor(data_0).astype("int")
    data_1 = np.floor(data_1).astype("int")
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

def calculate_cond_MI(data_0, data_1, cond_vec):
    assert(data_0.size == data_1.size)
    assert(data_0.size == cond_vec.size)
    # normalize data
    data_0 -= np.min(data_0)
    if data_0.std() != 0:
        data_0 /= np.std(data_0)
    data_1 -= np.min(data_1)
    if data_1.std() != 0:
        data_1 /= np.std(data_1)

    data_0 = np.floor(data_0).astype("int")
    data_1 = np.floor(data_1).astype("int")

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

def test_mutual_info_calculation():
    X, Y = utils.load_iris()
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            X1 = np.array(X[:,i])
            X2 = np.array(X[:,j])
            X1 = mutual_information.normalize_data(X1)
            X2 = mutual_information.normalize_data(X2)
            mi_pyfeast = calculate_MI(X1, X2)
            mi_ours = mutual_information.calculate_mutual_information_histogram_binning(X1, X2)
            # assert np.isclose(mi_pyfeast, mi_ours, rtol=0.01, atol=0.00001)
            print("features: %d and %d; pyfeast: %f \t ours: %f"%(i, j, mi_pyfeast, mi_ours))

def test_cond_mutual_info():
    X, Y = utils.load_iris()
    Y = Y.astype("int")
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            X1 = np.array(X[:,i])
            X2 = np.array(X[:,j])
            X1 = mutual_information.normalize_data(X1)
            X2 = mutual_information.normalize_data(X2)

            cond_mi_pyfeast = calculate_cond_MI(X1, X2, Y)
            cond_mi_ours = mutual_information.calculate_conditional_MI(X1, X2, Y)
            # assert np.isclose(cond_mi_pyfeast, cond_mi_ours, rtol=0.01, atol=0.00001)
            print("features: %d and %d; pyfeast: %f \t ours: %f"%(i, j, cond_mi_pyfeast, cond_mi_ours))
