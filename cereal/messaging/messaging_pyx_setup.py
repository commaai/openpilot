import os
import subprocess
import sysconfig
from distutils.core import Extension, setup  # pylint: disable=import-error,no-name-in-module

from Cython.Build import cythonize
from Cython.Distutils import build_ext


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


sourcefiles = ['messaging_pyx.pyx']
extra_compile_args = ["-std=c++11"]
libraries = ['zmq']
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
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'libmessaging.a'),
          ]
        )
      ),
      nthreads=4,
)
