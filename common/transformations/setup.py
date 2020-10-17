import numpy

from Cython.Build import cythonize
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module
from common.cython_hacks import BuildExtWithoutPlatformSuffix

setup(
  name='Cython transformations wrapper',
  cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
  ext_modules=cythonize(
    Extension(
      "transformations",
      sources=["transformations.pyx"],
      language="c++",
      extra_compile_args=["-std=c++14", "-Wno-cpp"],
      include_dirs=[numpy.get_include()],
    ),
    nthreads=4,
  )
)
