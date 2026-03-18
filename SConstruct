import os
import subprocess
import sys
import sysconfig
import platform
import shlex
import importlib
import numpy as np

import SCons.Errors
from SCons.Defaults import _stripixes

SCons.Warnings.warningAsException(True)

Decider('MD5-timestamp')

SetOption('num_jobs', max(1, int(os.cpu_count()/2)))

AddOption('--ccflags', action='store', type='string', default='', help='pass arbitrary flags over the command line')
AddOption('--verbose', action='store_true', default=False, help='show full build commands')
AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=os.path.exists(File('#.gitattributes').abspath), # minimal by default on release branch (where there's no LFS)
          help='the minimum build to run openpilot. no tests, tools, etc.')

# Detect platform
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"
elif arch == "aarch64" and os.path.isfile('/TICI'):
  arch = "larch64"
assert arch in [
  "larch64",  # linux tici arm64
  "aarch64",  # linux pc arm64
  "x86_64",   # linux pc x64
  "Darwin",   # macOS arm64 (x86 not supported)
]

pkg_names = ['bzip2', 'capnproto', 'eigen', 'ffmpeg', 'libjpeg', 'libyuv', 'ncurses', 'zeromq', 'zstd']
pkgs = [importlib.import_module(name) for name in pkg_names]


# ***** enforce a whitelist of system libraries *****
# this prevents silently relying on a 3rd party package,
# e.g. apt-installed libusb. all libraries should either
# be distributed with all Linux distros and macOS, or
# vendored in commaai/dependencies.
allowed_system_libs = {
  "EGL", "GLESv2", "GL", "Qt5Charts", "Qt5Core", "Qt5Gui", "Qt5Widgets",
  "dl", "drm", "gbm", "m", "pthread",
}

def _resolve_lib(env, name):
  for d in env.Flatten(env.get('LIBPATH', [])):
    p = Dir(str(d)).abspath
    for ext in ('.a', '.so', '.dylib'):
      f = File(os.path.join(p, f'lib{name}{ext}'))
      if f.exists() or f.has_builder():
        return name
  if name in allowed_system_libs:
    return name
  raise SCons.Errors.UserError(f"Unexpected non-vendored library '{name}'")

def _libflags(target, source, env, for_signature):
  libs = []
  lp = env.subst('$LIBLITERALPREFIX')
  for lib in env.Flatten(env.get('LIBS', [])):
    if isinstance(lib, str):
      if os.sep in lib or lib.startswith('#'):
        libs.append(File(lib))
      elif lib.startswith('-') or (lp and lib.startswith(lp)):
        libs.append(lib)
      else:
        libs.append(_resolve_lib(env, lib))
    else:
      libs.append(lib)
  return _stripixes(env['LIBLINKPREFIX'], libs, env['LIBLINKSUFFIX'],
                    env['LIBPREFIXES'], env['LIBSUFFIXES'], env, env['LIBLITERALPREFIX'])

env = Environment(
  ENV={
    "PATH": os.environ['PATH'],
    "PYTHONPATH": Dir("#").abspath + ':' + Dir(f"#third_party/acados").abspath,
    "ACADOS_SOURCE_DIR": Dir("#third_party/acados").abspath,
    "ACADOS_PYTHON_INTERFACE_PATH": Dir("#third_party/acados/acados_template").abspath,
    "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{arch}/t_renderer"
  },
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow" if arch in ("Darwin", "larch64") else "-Wshadow=local",
    "-Wno-unknown-warning-option",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
    "-Wno-vla-cxx-extension",
  ],
  CFLAGS=["-std=gnu11"],
  CXXFLAGS=["-std=c++1z"],
  CPPPATH=[
    "#",
    "#msgq",
    "#third_party",
    "#third_party/json11",
    "#third_party/linux/include",
    "#third_party/acados/include",
    "#third_party/acados/include/blasfeo/include",
    "#third_party/acados/include/hpipm/include",
    "#third_party/catch2/include",
    [x.INCLUDE_DIR for x in pkgs],
  ],
  LIBPATH=[
    "#common",
    "#msgq_repo",
    "#third_party",
    "#selfdrive/pandad",
    "#rednose/helpers",
    f"#third_party/acados/{arch}/lib",
    [x.LIB_DIR for x in pkgs],
  ],
  RPATH=[],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  REDNOSE_ROOT="#",
  tools=["default", "cython", "compilation_db", "rednose_filter"],
  toolpath=["#site_scons/site_tools", "#rednose_repo/site_scons/site_tools"],
)
if arch != "larch64":
  env['_LIBFLAGS'] = _libflags

# Arch-specific flags and paths
if arch == "larch64":
  env["CC"] = "clang"
  env["CXX"] = "clang++"
  env.Append(LIBPATH=[
    "/usr/lib/aarch64-linux-gnu",
  ])
  arch_flags = ["-D__TICI__", "-mcpu=cortex-a57"]
  env.Append(CCFLAGS=arch_flags)
  env.Append(CXXFLAGS=arch_flags)
elif arch == "Darwin":
  env.Append(LIBPATH=[
    "/System/Library/Frameworks/OpenGL.framework/Libraries",
  ])
  env.Append(CCFLAGS=["-DGL_SILENCE_DEPRECATION"])
  env.Append(CXXFLAGS=["-DGL_SILENCE_DEPRECATION"])

_extra_cc = shlex.split(GetOption('ccflags') or '')
if _extra_cc:
  env.Append(CCFLAGS=_extra_cc)

# no --as-needed on mac linker
if arch != "Darwin":
  env.Append(LINKFLAGS=["-Wl,--as-needed", "-Wl,--no-undefined"])

# Shorter build output: show brief descriptions instead of full commands.
# Full command lines are still printed on failure by scons.
if not GetOption('verbose'):
  for action, short in (
    ("CC",     "CC"),
    ("CXX",    "CXX"),
    ("LINK",   "LINK"),
    ("SHCC",   "CC"),
    ("SHCXX",  "CXX"),
    ("SHLINK", "LINK"),
    ("AR",     "AR"),
    ("RANLIB", "RANLIB"),
    ("AS",     "AS"),
  ):
    env[f"{action}COMSTR"] = f"  [{short}] $TARGET"

# progress output
node_interval = 5
node_count = 0
def progress_function(node):
  global node_count
  node_count += node_interval
  sys.stderr.write("progress: %d\n" % node_count)
if os.environ.get('SCONS_PROGRESS'):
  Progress(progress_function, interval=node_interval)

# ********** Cython build environment **********
envCython = env.Clone()
envCython["CPPPATH"] += [sysconfig.get_paths()['include'], np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-cpp", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = env["LINKFLAGS"] + ["-bundle", "-undefined", "dynamic_lookup"]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

np_version = SCons.Script.Value(np.__version__)
Export('envCython', 'np_version')

Export('env', 'arch')

# Setup cache dir
cache_dir = '/data/scons_cache' if arch == "larch64" else '/tmp/scons_cache'
CacheDir(cache_dir)
Clean(["."], cache_dir)

# ********** start building stuff **********

# Build common module
SConscript(['common/SConscript'])
Import('_common')
common = [_common, 'json11', 'zmq']
Export('common')

# Build messaging (cereal + msgq + socketmaster + their dependencies)
# Enable swaglog include in submodules
env_swaglog = env.Clone()
env_swaglog['CXXFLAGS'].append('-DSWAGLOG="\\"common/swaglog.h\\""')
SConscript(['msgq_repo/SConscript'], exports={'env': env_swaglog})

SConscript(['cereal/SConscript'])

Import('socketmaster', 'msgq')
messaging = [socketmaster, msgq, 'capnp', 'kj',]
Export('messaging')


# Build other submodules
SConscript(['panda/SConscript'])

# Build rednose library
SConscript(['rednose/SConscript'])

# Build system services
SConscript([
  'system/loggerd/SConscript',
])

if arch == "larch64":
  SConscript(['system/camerad/SConscript'])

# Build openpilot
SConscript(['third_party/SConscript'])

# Build selfdrive
SConscript([
  'selfdrive/pandad/SConscript',
  'selfdrive/controls/lib/lateral_mpc_lib/SConscript',
  'selfdrive/controls/lib/longitudinal_mpc_lib/SConscript',
  'selfdrive/locationd/SConscript',
  'selfdrive/modeld/SConscript',
  'selfdrive/ui/SConscript',
])

if Dir('#tools/cabana/').exists() and arch != "larch64":
  SConscript(['tools/cabana/SConscript'])


env.CompilationDatabase('compile_commands.json')
