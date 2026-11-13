import os
import platform
import subprocess
import sysconfig
import nanobind
import eigen

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()

common = ''

python_path = sysconfig.get_paths()['include']
cpppath = [
  '#',
  '#rednose',
  '#rednose/examples/generated',
  '/usr/lib/include',
  python_path,
  eigen.INCLUDE_DIR,
]

env = Environment(
  ENV=os.environ,
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
  tools=["default", "rednose_filter"],
)

# Python extension build environment
envNanobind = env.Clone()
nanobind_dir = os.path.dirname(nanobind.__file__)
envNanobind.Append(
  CPPPATH=[os.path.join(eigen.INCLUDE_DIR, "eigen3"), nanobind.include_dir(), os.path.join(nanobind_dir, "ext", "robin_map", "include")],
  CCFLAGS=["-fvisibility=hidden", "-Wno-shadow"],
)
envNanobind["NANOBIND_SOURCE"] = os.path.join(nanobind_dir, "src", "nb_combined.cpp")

envNanobind["LIBS"] = []
if platform.system() == "Darwin":
  envNanobind["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envNanobind["LINKFLAGS"] = ["-shared"]
  envNanobind["LIBS"] = [os.path.basename(python_path)]
else:
  envNanobind["LINKFLAGS"] = ["-pthread", "-shared"]

Export('env', 'envNanobind', 'common')

SConscript(['#rednose/SConscript'])
SConscript(['#examples/SConscript'])
