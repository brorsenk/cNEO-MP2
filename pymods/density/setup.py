from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='one dimensional density comparison',
    ext_modules=cythonize("density_comp.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()]
)
