import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class SConsBuildExt(build_ext):
  def build_extensions(self):
    subprocess.check_call([sys.executable, "-m", "SCons", "--minimal", "-j", str(self.parallel or os.cpu_count() or 1)])
    for ext in self.extensions:
      source = Path(*ext.name.split(".")).with_suffix(".so")
      target = Path(self.get_ext_fullpath(ext.name))
      target.parent.mkdir(parents=True, exist_ok=True)
      self.copy_file(str(source), str(target))


setup(
  ext_modules=[
    Extension("msgq.ipc_pyx", sources=[]),
    Extension("msgq.visionipc.visionipc_pyx", sources=[]),
  ],
  cmdclass={"build_ext": SConsBuildExt},
)
