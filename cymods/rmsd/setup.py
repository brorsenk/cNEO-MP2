from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='root mean square differences',
    ext_modules=cythonize("rmsd.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()]
)
