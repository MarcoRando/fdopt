import numpy 
from Cython.Build import cythonize
from distutils.core import setup,Extension

exts = cythonize([
    Extension("gram_schmidt",  
              sources=['direction_matrices/gram_schmidt.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])   
], language_level=3)

setup(ext_modules=exts)