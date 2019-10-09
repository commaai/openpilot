import subprocess
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix

PHONELIBS = '../../phonelibs'

ARCH = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()  # pylint: disable=unexpected-keyword-arg
ARCH_DIR = 'x64' if ARCH == "x86_64" else 'aarch64'

setup(name='Boardd API Implementation',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "boardd_api_impl",
          libraries=[':libcan_list_to_can_capnp.a', ':libcapnp.a', ':libcapnp.a', ':libkj.a'],
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
