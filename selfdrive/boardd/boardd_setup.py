from distutils.core import Extension, setup
from Cython.Build import cythonize
import os

from common.cython_hacks import BuildExtWithoutPlatformSuffix

CROSS_COMPILATION = os.getenv("CROSS_COMPILATION") is not None
sysroot_args=[]

if CROSS_COMPILATION:
  os.environ['CC'] = 'aarch64-linux-gnu-gcc'
  os.environ['CXX'] = 'aarch64-linux-gnu-g++'
  os.environ['LDSHARED'] = 'aarch64-linux-gnu-gcc -shared'
  os.environ['LDCXXSHARED'] = 'aarch64-linux-gnu-g++ -shared'
  os.environ["LD_LIBRARY_PATH"] = "/usr/aarch64-linux-gnu/lib"
  sysroot_args=['--sysroot', '/usr/aarch64-linux-gnu']

libraries = ['can_list_to_can_capnp', 'capnp', 'kj']

extra_compile_args = ["-std=c++11"]
extra_compile_args += sysroot_args

setup(name='Boardd API Implementation',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "boardd_api_impl",
          libraries=libraries,
          library_dirs=[
            './',
          ],
          sources=['boardd_api_impl.pyx'],
          language="c++",
          extra_compile_args=extra_compile_args,
        )
      )
)
