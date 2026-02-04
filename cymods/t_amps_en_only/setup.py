from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='t amplitudes electronic nuclear',
    ext_modules=cythonize("t_amps_en_only.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()]
)
