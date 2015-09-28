from libc.stdlib cimport calloc, free
import numpy as np
cimport numpy as np
import math

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def calculate_mutual_information(np.ndarray x1, np.ndarray x2, float base=2.):
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("x1 and x2 must have the same dimension")
    assert x1.dtype == DTYPE and x2.dtype == DTYPE

    cdef int n_samples = x1.shape[0]
    x1 -= np.min(x1)
    x2 -= np.min(x2)

    cdef int i, j

    cdef int *x1_cumstates = <int*>calloc(n_samples, sizeof(int))
    cdef int *x2_cumstates = <int*>calloc(n_samples, sizeof(int))

    cdef int numstates_x1 = np.max(x1) + 1
    cdef int numstates_x2 = np.max(x2) + 1

    cdef int *histogram = <int*>calloc(numstates_x1 * numstates_x2, sizeof(int))
    cdef float mutual_information = 0

    cdef int histogram_val
    cdef float p_xy
    cdef float p_x
    cdef float p_y

    try:
        for i in range(n_samples):
            x1_cumstates[x1[i]] += 1
            x2_cumstates[x2[i]] += 1

        for i in range(n_samples):
            histogram[x2[i] * numstates_x1 + x1[i]] += 1

        for i in range(numstates_x1):
            for j in range(numstates_x2):
                histogram_val = histogram[j * numstates_x1 + i]
                # print histogram_val
                if histogram_val != 0:
                    # p(x, y) * log(p(x, y)/p(x)/p(y))
                    p_x = <float>x1_cumstates[i] / <float>n_samples
                    p_y = <float>x2_cumstates[j] / <float>n_samples
                    p_xy = <float>histogram_val / <float>n_samples
                    # print p_x, p_y
                    mutual_information += p_xy * math.log(p_xy / p_x / p_y) / math.log(base)



    finally:
        free(x1_cumstates)
        free(x2_cumstates)
        free(histogram)

    return mutual_information
