
import numpy as np
cimport numpy as np 

np.import_array()

from cython.parallel cimport parallel, prange

cimport openmp

cdef gram_schmidt(double* gauss_mat, int mat_len):
    cdef double sum_vecs = 0.0

    for i in prange(mat_len, nogil=True):
        sum_vecs += i
#        num_threads = openmp.omp_get_num_threads()
    return sum_vecs