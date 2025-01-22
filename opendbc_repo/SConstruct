import os
import subprocess
import sysconfig
import platform
import numpy as np

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '/usr/lib/include',
  python_path
]

AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=True,
          help='the minimum build. no tests, tools, etc.')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

ccflags_asan = ["-fsanitize=address", "-fno-omit-frame-pointer"] if GetOption('asan') else []
ldflags_asan = ["-fsanitize=address"] if GetOption('asan') else []

env = Environment(
  ENV=os.environ,
  CC='gcc',
  CXX='g++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow",
    "-Wno-vla-cxx-extension",
    "-Wno-unknown-warning-option",  # for compatibility across compiler versions
  ] + ccflags_asan,
  LDFLAGS=ldflags_asan,
  LINKFLAGS=ldflags_asan,
  LIBPATH=[
    "#opendbc/can/",
  ],
  CFLAGS="-std=gnu11",
  CXXFLAGS=["-std=c++1z"],
  CPPPATH=cpppath,
  CYTHONCFILESUFFIX=".cpp",
  tools=["default", "cython"]
)

common = ''
Export('env', 'arch', 'common')

envCython = env.Clone()
envCython["CPPPATH"] += [np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

python_libs = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]

  python_libs.append(os.path.basename(python_path))
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

envCython["LIBS"] = python_libs

Export('envCython')

SConscript(['SConscript'])
