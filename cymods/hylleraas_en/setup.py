from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Hylleraas energy electronic nuclear',
    ext_modules=cythonize("hylleraas_en.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()]
)
