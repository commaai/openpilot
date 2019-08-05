from distutils.core import setup, Extension
from Cython.Build import cythonize
import subprocess

sourcefiles = ['parser_pyx.pyx']
extra_compile_args = ["-std=c++11"]
ARCH = subprocess.check_output(["uname", "-m"]).rstrip()
if ARCH == "aarch64":
  extra_compile_args += ["-Wno-deprecated-register"]

setup(name='Radard Thread',
      ext_modules=cythonize(
        Extension(
          "parser_pyx",
          sources=sourcefiles,
          extra_compile_args=extra_compile_args
        ),
        nthreads=4))
