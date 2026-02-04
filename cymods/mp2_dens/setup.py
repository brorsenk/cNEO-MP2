from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='nuclear mp2 density',
    ext_modules=cythonize("mp2_density.pyx", compiler_directives={'boundscheck': False}),
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp']


)
