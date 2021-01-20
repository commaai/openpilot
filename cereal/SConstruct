import os
import platform
import subprocess
import sysconfig

zmq = 'zmq'

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

cereal_dir = Dir('.')
messaging_dir = Dir('./messaging')

cpppath = [
  cereal_dir,
  messaging_dir,
  '/usr/lib/include',
  sysconfig.get_paths()['include'],
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
  CXXFLAGS="-std=c++1z",
  CPPPATH=cpppath,
  CYTHONCFILESUFFIX=".cpp",
  tools=["default", "cython"]
)

QCOM_REPLAY = False
Export('env', 'zmq', 'arch', 'QCOM_REPLAY')


envCython = env.Clone(LIBS=[])
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-deprecated-declarations"]
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(python_path)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('envCython')


SConscript(['SConscript'])
