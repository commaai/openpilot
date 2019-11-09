import os
import subprocess
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

from Cython.Build import cythonize

from common.basedir import BASEDIR
from common.cython_hacks import BuildExtWithoutPlatformSuffix

sourcefiles = ['messaging_pyx.pyx']
extra_compile_args = ["-std=c++11"]
libraries = []
ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()  # pylint: disable=unexpected-keyword-arg

if ARCH == "aarch64":
  extra_compile_args += ["-Wno-deprecated-register"]
  libraries += ['gnustl_shared']

setup(name='CAN parser',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "messaging_pyx",
          language="c++",
          sources=sourcefiles,
          extra_compile_args=extra_compile_args,
          libraries=libraries,
          extra_objects=[
            os.path.join(BASEDIR, 'selfdrive', 'messaging', 'messaging.a'),
          ]
        )
      ),
      nthreads=4,
)
