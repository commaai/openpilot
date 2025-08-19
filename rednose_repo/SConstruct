import os
import subprocess
import sysconfig
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

common = ''

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '#rednose',
  '#rednose/examples/generated',
  '/usr/lib/include',
  python_path,
  np.get_include(),
]

env = Environment(
  ENV=os.environ,
  CC='clang',
  CXX='clang++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Werror=implicit-function-declaration",
    "-Werror=incompatible-pointer-types",
    "-Werror=int-conversion",
    "-Werror=return-type",
    "-Werror=format-extra-args",
    "-Wshadow",
  ],
  LIBPATH=["#rednose/examples/generated"],
  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++1z",
  CPPPATH=cpppath,
  REDNOSE_ROOT=Dir("#").abspath,
  tools=["default", "cython", "rednose_filter"],
)

# Cython build enviroment
envCython = env.Clone()
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(python_path)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('env', 'envCython', 'common')

SConscript(['#rednose/SConscript'])
SConscript(['#examples/SConscript'])
