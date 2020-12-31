import Cython
import distutils
import os
import subprocess
import sys

zmq = 'zmq'
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

# Rebuild cython extensions if python, distutils, or cython change
cython_dependencies = [Value(v) for v in (sys.version, distutils.__version__, Cython.__version__)]
Export('cython_dependencies')

cereal_dir = Dir('.')

cpppath = [
    cereal_dir,
    '/usr/lib/include',
]

AddOption('--test',
          action='store_true',
          help='build test files')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

ccflags_asan = ["-fsanitize=address", "-fno-omit-frame-pointer"] if GetOption('asan') else []
ldflags_asan = ["-fsanitize=address"] if GetOption('asan') else []

env = Environment(
  ENV=os.environ,
  CC='clang',
  CXX='clang++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
  ] + ccflags_asan,
  LDFLAGS=ldflags_asan,
  LINKFLAGS=ldflags_asan,

  CFLAGS="-std=gnu11",
  CXXFLAGS="-std=c++14",
  CPPPATH=cpppath,
)


Export('env', 'zmq', 'arch')
SConscript(['SConscript'])
