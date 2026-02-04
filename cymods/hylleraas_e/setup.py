from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Hylleraas energy electronic',
    ext_modules=cythonize("hylleraas_e.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()]
)

