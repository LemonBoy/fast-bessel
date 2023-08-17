import numpy as np
from fastb import jnv_AMOS, jnv_AMOS_vec, jnv_CEPHES, jnv_SCIPY
import timeit

N_ORDERS = 10
N_TRIALS = 3
SKIP_VALIDITY_CHECK = False

n = np.arange(N_ORDERS, dtype=int)
x = np.arange(1e-3, 200, 1e-3)


def try_AMOS():
    jnv_AMOS(n, x)


def try_AMOSv():
    jnv_AMOS_vec(n, x)


def try_CEPHES():
    jnv_CEPHES(n, x)


def try_SCIPY():
    jnv_SCIPY(n, x)


if not SKIP_VALIDITY_CHECK:
    print("Checking validity...")
    new_AMOS = jnv_AMOS(n, x)
    new_AMOS_vec = jnv_AMOS_vec(n, x)
    new_CEPHES = jnv_CEPHES(n, x)
    new_SCIPY = jnv_SCIPY(n, x)
    assert np.allclose(new_AMOS, new_AMOS_vec)
    print("SCIPY vs CEPHES", np.allclose(new_SCIPY, new_CEPHES))
    print("SCIPY vs AMOS  ", np.allclose(new_SCIPY, new_AMOS_vec))
    print("CEPHES vs AMOS ", np.allclose(new_CEPHES, new_AMOS))

print("Measuring the execution time")
elapsed = timeit.timeit(try_CEPHES, number=N_TRIALS)
print("CEPHES:    s/loop", elapsed / N_TRIALS)
elapsed = timeit.timeit(try_AMOS, number=N_TRIALS)
print("AMOS:      s/loop", elapsed / N_TRIALS)
elapsed = timeit.timeit(try_AMOSv, number=N_TRIALS)
print("AMOS(vec): s/loop", elapsed / N_TRIALS)
elapsed = timeit.timeit(try_SCIPY, number=N_TRIALS)
print("SCIPY:     s/loop", elapsed / N_TRIALS)
