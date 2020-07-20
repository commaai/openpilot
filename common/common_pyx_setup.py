from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module
from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix

sourcefiles = ['clock.pyx']
extra_compile_args = ["-std=c++11"]

setup(name='Common',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "common_pyx",
          language="c++",
          sources=sourcefiles,
          extra_compile_args=extra_compile_args,
        )
      ),
      nthreads=4,
)
