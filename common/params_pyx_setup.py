from distutils.core import Extension, setup 
from Cython.Build import cythonize
from common.cython_hacks import BuildExtWithoutPlatformSuffix

sourcefiles = ['params_wrapper.pyx']
extra_compile_args = ["-std=c++17"]

setup(name='Common',
      cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
      ext_modules=cythonize(
        Extension(
          "params",
          language="c++",
          sources=sourcefiles,
          extra_compile_args=extra_compile_args
        )
      )
)
