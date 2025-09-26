import os
import subprocess
import sys
import sysconfig
import platform
import shlex
import numpy as np
import panda

import SCons.Errors

SCons.Warnings.warningAsException(True)

Decider('MD5-timestamp')

SetOption('num_jobs', max(1, int(os.cpu_count()/2)))

AddOption('--kaitai', action='store_true', help='Regenerate kaitai struct parsers')
AddOption('--asan', action='store_true', help='turn on ASAN')
AddOption('--ubsan', action='store_true', help='turn on UBSan')
AddOption('--mutation', action='store_true', help='generate mutation-ready code')
AddOption('--ccflags', action='store', type='string', default='', help='pass arbitrary flags over the command line')
AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=os.path.exists(File('#.lfsconfig').abspath), # minimal by default on release branch (where there's no LFS)
          help='the minimum build to run openpilot. no tests, tools, etc.')

# Detect platform
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"
  brew_prefix = subprocess.check_output(['brew', '--prefix'], encoding='utf8').strip()
elif arch == "aarch64" and os.path.isfile('/TICI'):
  arch = "larch64"
assert arch in [
  "larch64",  # linux tici arm64
  "aarch64",  # linux pc arm64
  "x86_64",   # linux pc x64
  "Darwin",   # macOS arm64 (x86 not supported)
]

env = Environment(
  ENV={
    "PATH": os.environ['PATH'],
    "PYTHONPATH": Dir("#").abspath + ':' + Dir(f"#third_party/acados").abspath,
    "ACADOS_SOURCE_DIR": Dir("#third_party/acados").abspath,
    "ACADOS_PYTHON_INTERFACE_PATH": Dir("#third_party/acados/acados_template").abspath,
    "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{arch}/t_renderer"
  },
  CC='clang',
  CXX='clang++',
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow",
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
    "#third_party/libyuv/include",
  ],
  LIBPATH=[
    "#common",
  ],
  LIBPATH=[
    "#common",
    "#msgq_repo",
    "#third_party",
    "#selfdrive/pandad",
    "#rednose/helpers",
    f"#third_party/libyuv/{arch}/lib",
    f"#third_party/acados/{arch}/lib",
  ],
  RPATH=[],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  REDNOSE_ROOT="#",
  tools=["default", "cython", "compilation_db", "rednose_filter"],
  toolpath=["#site_scons/site_tools", "#rednose_repo/site_scons/site_tools"],
)

env.AppendUnique(CPPPATH=[panda.INCLUDE_PATH])

# Arch-specific flags and paths
if arch == "larch64":
  env.Append(CPPPATH=["#third_party/opencl/include"])
  env.Append(LIBPATH=[
    "/usr/local/lib",
    "/system/vendor/lib64",
    "/usr/lib/aarch64-linux-gnu",
  ])
  arch_flags = ["-D__TICI__", "-mcpu=cortex-a57"]
  env.Append(CCFLAGS=arch_flags)
  env.Append(CXXFLAGS=arch_flags)
elif arch == "Darwin":
  env.Append(LIBPATH=[
    f"{brew_prefix}/lib",
    f"{brew_prefix}/opt/openssl@3.0/lib",
    f"{brew_prefix}/opt/llvm/lib/c++",
    "/System/Library/Frameworks/OpenGL.framework/Libraries",
  ])
  env.Append(CCFLAGS=["-DGL_SILENCE_DEPRECATION"])
  env.Append(CXXFLAGS=["-DGL_SILENCE_DEPRECATION"])
  env.Append(CPPPATH=[
    f"{brew_prefix}/include",
    f"{brew_prefix}/opt/openssl@3.0/include",
  ])
else:
  env.Append(LIBPATH=[
    "/usr/lib",
    "/usr/local/lib",
  ])

# Sanitizers and extra CCFLAGS from CLI
if GetOption('asan'):
  env.Append(CCFLAGS=["-fsanitize=address", "-fno-omit-frame-pointer"])
  env.Append(LINKFLAGS=["-fsanitize=address"])
elif GetOption('ubsan'):
  env.Append(CCFLAGS=["-fsanitize=undefined"])
  env.Append(LINKFLAGS=["-fsanitize=undefined"])

_extra_cc = shlex.split(GetOption('ccflags') or '')
if _extra_cc:
  env.Append(CCFLAGS=_extra_cc)

# no --as-needed on mac linker
if arch != "Darwin":
  env.Append(LINKFLAGS=["-Wl,--as-needed", "-Wl,--no-undefined"])

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
py_include = sysconfig.get_paths()['include']
envCython = env.Clone()
envCython["CPPPATH"] += [py_include, np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = env["LINKFLAGS"] + ["-bundle", "-undefined", "dynamic_lookup"]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

np_version = SCons.Script.Value(np.__version__)
Export('envCython', 'np_version')

# ********** Qt build environment **********
qt_env = env.Clone()
qt_modules = ["Widgets", "Gui", "Core", "Network", "Concurrent", "DBus", "Xml"]

qt_libs = []
if arch == "Darwin":
  qt_env['QTDIR'] = f"{brew_prefix}/opt/qt@5"
  qt_dirs = [
    os.path.join(qt_env['QTDIR'], "include"),
  ]
  qt_dirs += [f"{qt_env['QTDIR']}/include/Qt{m}" for m in qt_modules]
  qt_env["LINKFLAGS"] += ["-F" + os.path.join(qt_env['QTDIR'], "lib")]
  qt_env["FRAMEWORKS"] += [f"Qt{m}" for m in qt_modules] + ["OpenGL"]
  qt_env.AppendENVPath('PATH', os.path.join(qt_env['QTDIR'], "bin"))
else:
  qt_install_prefix = subprocess.check_output(['qmake', '-query', 'QT_INSTALL_PREFIX'], encoding='utf8').strip()
  qt_install_headers = subprocess.check_output(['qmake', '-query', 'QT_INSTALL_HEADERS'], encoding='utf8').strip()

  qt_env['QTDIR'] = qt_install_prefix
  qt_dirs = [
    f"{qt_install_headers}",
  ]

  qt_gui_path = os.path.join(qt_install_headers, "QtGui")
  qt_gui_dirs = [d for d in os.listdir(qt_gui_path) if os.path.isdir(os.path.join(qt_gui_path, d))]
  qt_dirs += [f"{qt_install_headers}/QtGui/{qt_gui_dirs[0]}/QtGui", ] if qt_gui_dirs else []
  qt_dirs += [f"{qt_install_headers}/Qt{m}" for m in qt_modules]

  qt_libs = [f"Qt5{m}" for m in qt_modules]
  if arch == "larch64":
    qt_libs += ["GLESv2", "wayland-client"]
    qt_env.PrependENVPath('PATH', Dir("#third_party/qt5/larch64/bin/").abspath)
  elif arch != "Darwin":
    qt_libs += ["GL"]
qt_env['QT3DIR'] = qt_env['QTDIR']
qt_env.Tool('qt3')

qt_env['CPPPATH'] += qt_dirs + ["#third_party/qrcode"]
qt_flags = [
  "-D_REENTRANT",
  "-DQT_NO_DEBUG",
  "-DQT_WIDGETS_LIB",
  "-DQT_GUI_LIB",
  "-DQT_CORE_LIB",
  "-DQT_MESSAGELOGCONTEXT",
]
qt_env['CXXFLAGS'] += qt_flags
qt_env['LIBPATH'] += ['#selfdrive/ui', ]
qt_env['LIBS'] = qt_libs

Export('env', 'qt_env', 'arch')

# Setup cache dir
cache_dir = '/data/scons_cache' if arch == "larch64" else '/tmp/scons_cache'
CacheDir(cache_dir)
Clean(["."], cache_dir)

# ********** start building stuff **********

# Build common module
SConscript(['common/SConscript'])
Import('_common')
Import('_common')
common = [_common, 'json11', 'zmq']
Export('common')
Export('common')

# Build messaging (cereal + msgq + socketmaster + their dependencies)
# Enable swaglog include in submodules
env_swaglog = env.Clone()
env_swaglog['CXXFLAGS'].append('-DSWAGLOG="\\"common/swaglog.h\\""')
SConscript(['msgq_repo/SConscript'], exports={'env': env_swaglog})
SConscript(['opendbc_repo/SConscript'], exports={'env': env_swaglog})

SConscript(['cereal/SConscript'])

Import('socketmaster', 'msgq')
messaging = [socketmaster, msgq, 'capnp', 'kj',]
Export('messaging')

# Build rednose library
SConscript(['rednose/SConscript'])

# Build system services
SConscript([
  'system/ubloxd/SConscript',
  'system/loggerd/SConscript',
])

if arch == "larch64":
  SConscript(['system/camerad/SConscript'])

# Build openpilot
SConscript(['third_party/SConscript'])

SConscript(['selfdrive/SConscript'])

if Dir('#tools/cabana/').exists() and GetOption('extras'):
  SConscript(['tools/replay/SConscript'])
  if arch != "larch64":
    SConscript(['tools/cabana/SConscript'])


env.CompilationDatabase('compile_commands.json')
