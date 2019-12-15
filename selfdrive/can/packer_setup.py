from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix

setup(name='CAN Packer API Implementation',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(Extension("packer_impl", ["packer_impl.pyx"], language="c++", extra_compile_args=["-std=c++11"])))
