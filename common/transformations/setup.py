import os
import numpy
import sysconfig

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

def get_ext_filename_without_platform_suffix(filename):
  name, ext = os.path.splitext(filename)
  ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

  if ext_suffix == ext:
    return filename

  ext_suffix = ext_suffix.replace(ext, '')
  idx = name.find(ext_suffix)

  if idx == -1:
    return filename
  else:
    return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
  def get_ext_filename(self, ext_name):
    filename = super().get_ext_filename(ext_name)
    return get_ext_filename_without_platform_suffix(filename)


setup(
  name='Cython transformations wrapper',
  cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
  ext_modules=cythonize(
  Extension(
    "transformations",
    sources=["transformations.pyx"],
    language="c++",
    extra_compile_args=["-std=c++14"],
    include_dirs=[numpy.get_include()],
  )
))
