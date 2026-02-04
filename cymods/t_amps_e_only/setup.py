from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='t amplitudes electronic',
    ext_modules=cythonize("t_amps_e_only.pyx", compiler_directives={'boundscheck': False, 'auto_pickle':True}),
    include_dirs=[numpy.get_include()]
)
