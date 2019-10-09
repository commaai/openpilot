import subprocess
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix

sourcefiles = ['parser_pyx.pyx']
extra_compile_args = ["-std=c++11"]
ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()  # pylint: disable=unexpected-keyword-arg

if ARCH == "aarch64":
  extra_compile_args += ["-Wno-deprecated-register"]

setup(name='Radard Thread',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "parser_pyx",
          sources=sourcefiles,
          extra_compile_args=extra_compile_args
        ),
        nthreads=4))
