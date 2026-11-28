import os
from pathlib import Path
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class SConsBuildExt(build_ext):
  def build_extensions(self):
    subprocess.check_call([sys.executable, "-m", "SCons", "--minimal", "-j", str(self.parallel or os.cpu_count() or 1)])
    for ext in self.extensions:
      module = Path(*ext.name.split("."))
      source = module.with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))
      target = Path(self.get_ext_fullpath(ext.name))
      target.parent.mkdir(parents=True, exist_ok=True)
      self.copy_file(str(source), str(target))
      stub = module.with_suffix(".pyi")
      self.copy_file(str(stub), str(target.parent / stub.name))


setup(
  ext_modules=[
    Extension("msgq._ipc", sources=[]),
    Extension("msgq.visionipc._visionipc", sources=[]),
  ],
  cmdclass={"build_ext": SConsBuildExt},
)
