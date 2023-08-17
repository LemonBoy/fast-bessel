import numpy as np
import cython

cdef extern from *:
    """
    extern "C" {
        double jn(int n, double x);
        double jv(double n, double x);
    }
    """
    double jn(int n, double x) nogil
    double jv(double n, double x) nogil

from scipy.special.cython_special cimport jv as scipy_jv

cdef extern from "zbessel/zbessel.hh" namespace "zbessel":
    int zbesj(double zr, double zi, double fnu, int kode, int n, double *cyr, double *cyi, int *nz) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def jnv_AMOS(long[::1] x, double[::1] a):
    cdef double[:, ::1] out = np.empty((x.shape[0], a.shape[0]))
    cdef int i, j, n_zeros
    cdef double im_part
    with nogil:
        for i in range(x.shape[0]):
            for j in range(a.shape[0]):
                zbesj(a[j], 0, x[i], 1, 1, &out[i, j], &im_part, &n_zeros)
    return np.asarray(out)

@cython.boundscheck(False)
@cython.wraparound(False)
def jnv_AMOS_vec(long[::1] x, double[::1] a):
    cdef double[:, ::1] out = np.empty((a.shape[0], x.shape[0]))
    cdef int j, n_zeros
    cdef double[:] im_part = np.empty((x.shape[0],))
    with nogil:
        # Yes, this is partially cheating by not looking at `x`... but what I'm interested in is a 
        # contiguous set of orders.
        for j in range(a.shape[0]):
            zbesj(a[j], 0, x[0], 1, x.shape[0], &out[j, 0], &im_part[0], &n_zeros)
    return np.asarray(out).T

@cython.boundscheck(False)
@cython.wraparound(False)
def jnv_CEPHES(long[::1] x, double[::1] a):
    cdef double[:, ::1] out = np.zeros((x.shape[0], a.shape[0]))
    with nogil:
        for i in range(x.shape[0]):
            for j in range(a.shape[0]):
                out[i, j] = jv(x[i], a[j])
    return np.asarray(out)

@cython.boundscheck(False)
@cython.wraparound(False)
def jnv_SCIPY(long[::1] x, double[::1] a):
    cdef double[:, ::1] out = np.empty((x.shape[0], a.shape[0]))
    with nogil:
        for i in range(x.shape[0]):
            for j in range(a.shape[0]):
                out[i, j] = scipy_jv(x[i], a[j])
    return np.asarray(out)