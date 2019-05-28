from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(name='CAN Packer API Implementation',
      ext_modules=cythonize(Extension("packer_impl", ["packer_impl.pyx"], language="c++", extra_compile_args=["-std=c++11"])))
