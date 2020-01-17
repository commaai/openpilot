import subprocess
from distutils.core import Extension, setup

from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix
from common.basedir import BASEDIR
import os

PHONELIBS = os.path.join(BASEDIR, 'phonelibs')

ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
ARCH_DIR = 'x64' if ARCH == "x86_64" else 'aarch64'

setup(name='Boardd API Implementation',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "boardd_api_impl",
          libraries=[':libcan_list_to_can_capnp.a', ':libcapnp.a', ':libkj.a'] if ARCH == "x86_64" else [':libcan_list_to_can_capnp.a', 'capnp', 'kj'],
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
