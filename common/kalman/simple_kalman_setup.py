from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='Simple Kalman Implementation',
      ext_modules=cythonize(Extension("simple_kalman_impl", ["simple_kalman_impl.pyx"])))