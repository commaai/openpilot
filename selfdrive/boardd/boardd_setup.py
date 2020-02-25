import subprocess
import platform
from distutils.core import Extension, setup

from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix
from common.basedir import BASEDIR
import os

PHONELIBS = os.path.join(BASEDIR, 'phonelibs')

ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

if ARCH == "x86_64":
  if platform.system() == "Darwin":
    libraries = ['can_list_to_can_capnp', 'capnp', 'kj']
    ARCH_DIR = 'mac'
  else:
    libraries = [':libcan_list_to_can_capnp.a', ':libcapnp.a', ':libkj.a']
    ARCH_DIR = 'x64'
else:
  libraries = [':libcan_list_to_can_capnp.a', 'capnp', 'kj']
  ARCH_DIR = 'aarch64'

setup(name='Boardd API Implementation',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "boardd_api_impl",
          libraries=libraries,
          library_dirs=[
            './',
            PHONELIBS + '/capnp-cpp/' + ARCH_DIR + '/lib/',
            PHONELIBS + '/capnp-c/' + ARCH_DIR + '/lib/'
          ],
          sources=['boardd_api_impl.pyx'],
          language="c++",
          extra_compile_args=["-std=c++11"],
        )
      )
)
