from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='t amplitudes nuclear',
    ext_modules=cythonize("t_amps_n_only.pyx"),
    include_dirs=[numpy.get_include()]
)
