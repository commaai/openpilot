import os
import subprocess
from distutils.core import Extension, setup
from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix
from common.basedir import BASEDIR
from common.hardware import TICI

ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()  # pylint: disable=unexpected-keyword-arg

sourcefiles = ['params_pyx.pyx']
extra_compile_args = ["-std=c++11"]

if ARCH == "aarch64":
  if TICI:
    extra_compile_args += ["-DQCOM2"]
  else:
    extra_compile_args += ["-DQCOM"]


setup(name='common',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "params_pyx",
          language="c++",
          sources=sourcefiles,
          include_dirs=[BASEDIR, os.path.join(BASEDIR, 'selfdrive')],
          extra_compile_args=extra_compile_args
        )
      )
)
