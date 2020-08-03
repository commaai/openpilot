from distutils.core import Extension, setup
from Cython.Build import cythonize

from common.cython_hacks import BuildExtWithoutPlatformSuffix

libraries = ['can_list_to_can_capnp', 'capnp', 'kj']

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
          extra_compile_args=["-std=c++11"],
        )
      )
)
