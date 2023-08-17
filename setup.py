from setuptools import setup, Extension
from Cython.Build import cythonize


sourcefiles = [
    "fastb.pyx",
    "cephes/ieee/mtherr.c",
    "cephes/ellf/const.c",
    "cephes/bessel/airy.c",
    "cephes/bessel/jn.c",
    "cephes/bessel/jv.c",
    "cephes/bessel/j0.c",
    "cephes/bessel/j1.c",
    "cephes/misc/polevl.c",
    "cephes/misc/beta.c",
    "cephes/cprob/gamma.c",
    "cephes/cmath/isnan.c",
    "zbessel/zbessel.cc",
]

extensions = [
    Extension(
        "fastb",
        sourcefiles,
        language="c++",
    )
]

setup(ext_modules=cythonize(extensions))
