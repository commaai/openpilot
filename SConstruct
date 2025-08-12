import os
import subprocess
import sys
import sysconfig
import platform
import numpy as np

import SCons.Errors

SCons.Warnings.warningAsException(True)

# pending upstream fix - https://github.com/SCons/scons/issues/4461
#SetOption('warn', 'all')

TICI = os.path.isfile('/TICI')
AGNOS = TICI

Decider('MD5-timestamp')

SetOption('num_jobs', max(1, int(os.cpu_count()/2)))

AddOption('--kaitai',
          action='store_true',
          help='Regenerate kaitai struct parsers')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

AddOption('--ubsan',
          action='store_true',
          help='turn on UBSan')

AddOption('--coverage',
          action='store_true',
          help='build with test coverage options')

AddOption('--clazy',
          action='store_true',
          help='build with clazy')

AddOption('--ccflags',
          action='store',
          type='string',
          default='',
          help='pass arbitrary flags over the command line')

AddOption('--external-sconscript',
          action='store',
          metavar='FILE',
          dest='external_sconscript',
          help='add an external SConscript to the build')

AddOption('--mutation',
          action='store_true',
          help='generate mutation-ready code')

AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=os.path.exists(File('#.lfsconfig').abspath), # minimal by default on release branch (where there's no LFS)
          help='the minimum build to run openpilot. no tests, tools, etc.')

## Architecture name breakdown (arch)
## - larch64: linux tici aarch64
## - aarch64: linux pc aarch64
## - x86_64:  linux pc x64
## - Darwin:  mac x64 or arm64
real_arch = arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"
  brew_prefix = subprocess.check_output(['brew', '--prefix'], encoding='utf8').strip()
elif arch == "aarch64" and AGNOS:
  arch = "larch64"
assert arch in ["larch64", "aarch64", "x86_64", "Darwin"]

lenv = {
  "PATH": os.environ['PATH'],
  "PYTHONPATH": Dir("#").abspath + ':' + Dir(f"#third_party/acados").abspath,

  "ACADOS_SOURCE_DIR": Dir("#third_party/acados").abspath,
  "ACADOS_PYTHON_INTERFACE_PATH": Dir("#third_party/acados/acados_template").abspath,
  "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{arch}/t_renderer"
}

rpath = []

if arch == "larch64":
  cpppath = [
    "#third_party/opencl/include",
  ]

  libpath = [
    "/usr/local/lib",
    "/system/vendor/lib64",
    f"#third_party/acados/{arch}/lib",
  ]

  libpath += [
    "#third_party/libyuv/larch64/lib",
    "/usr/lib/aarch64-linux-gnu"
  ]
  cflags = ["-DQCOM2", "-mcpu=cortex-a57"]
  cxxflags = ["-DQCOM2", "-mcpu=cortex-a57"]
  rpath += ["/usr/local/lib"]
else:
  cflags = []
  cxxflags = []
  cpppath = []
  rpath += []

  # MacOS
  if arch == "Darwin":
    libpath = [
      f"#third_party/libyuv/{arch}/lib",
      f"#third_party/acados/{arch}/lib",
      f"{brew_prefix}/lib",
      f"{brew_prefix}/opt/openssl@3.0/lib",
      "/System/Library/Frameworks/OpenGL.framework/Libraries",
    ]

    cflags += ["-DGL_SILENCE_DEPRECATION"]
    cxxflags += ["-DGL_SILENCE_DEPRECATION"]
    cpppath += [
      f"{brew_prefix}/include",
      f"{brew_prefix}/opt/openssl@3.0/include",
    ]
  # Linux
  else:
    libpath = [
      f"#third_party/acados/{arch}/lib",
      f"#third_party/libyuv/{arch}/lib",
      "/usr/lib",
      "/usr/local/lib",
    ]

if GetOption('asan'):
  ccflags = ["-fsanitize=address", "-fno-omit-frame-pointer"]
  ldflags = ["-fsanitize=address"]
elif GetOption('ubsan'):
  ccflags = ["-fsanitize=undefined"]
  ldflags = ["-fsanitize=undefined"]
else:
  ccflags = []
  ldflags = []

# no --as-needed on mac linker
if arch != "Darwin":
  ldflags += ["-Wl,--as-needed", "-Wl,--no-undefined"]

ccflags_option = GetOption('ccflags')
if ccflags_option:
  ccflags += ccflags_option.split(' ')

env = Environment(
  ENV=lenv,
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
  ] + cflags + ccflags,

  CPPPATH=cpppath + [
    "#",
    "#third_party/acados/include",
    "#third_party/acados/include/blasfeo/include",
    "#third_party/acados/include/hpipm/include",
    "#third_party/catch2/include",
    "#third_party/libyuv/include",
    "#third_party/json11",
    "#third_party/linux/include",
    "#third_party",
    "#msgq",
  ],

  CC='clang',
  CXX='clang++',
  LINKFLAGS=ldflags,

  RPATH=rpath,

  CFLAGS=["-std=gnu11"] + cflags,
  CXXFLAGS=["-std=c++1z"] + cxxflags,
  LIBPATH=libpath + [
    "#msgq_repo",
    "#third_party",
    "#selfdrive/pandad",
    "#common",
    "#rednose/helpers",
  ],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  REDNOSE_ROOT="#",
  tools=["default", "cython", "compilation_db", "rednose_filter"],
  toolpath=["#site_scons/site_tools", "#rednose_repo/site_scons/site_tools"],
)

if arch == "Darwin":
  # RPATH is not supported on macOS, instead use the linker flags
  darwin_rpath_link_flags = [f"-Wl,-rpath,{path}" for path in env["RPATH"]]
  env["LINKFLAGS"] += darwin_rpath_link_flags

env.CompilationDatabase('compile_commands.json')

# Setup cache dir
cache_dir = '/data/scons_cache' if AGNOS else '/tmp/scons_cache'
CacheDir(cache_dir)
Clean(["."], cache_dir)

node_interval = 5
node_count = 0
def progress_function(node):
  global node_count
  node_count += node_interval
  sys.stderr.write("progress: %d\n" % node_count)

if os.environ.get('SCONS_PROGRESS'):
  Progress(progress_function, interval=node_interval)

# Cython build environment
py_include = sysconfig.get_paths()['include']
envCython = env.Clone()
envCython["CPPPATH"] += [py_include, np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"] + darwin_rpath_link_flags
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

np_version = SCons.Script.Value(np.__version__)
Export('envCython', 'np_version')

# Qt build environment
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

if GetOption("clazy"):
  checks = [
    "level0",
    "level1",
    "no-range-loop",
    "no-non-pod-global-static",
  ]
  qt_env['CXX'] = 'clazy'
  qt_env['ENV']['CLAZY_IGNORE_DIRS'] = qt_dirs[0]
  qt_env['ENV']['CLAZY_CHECKS'] = ','.join(checks)

Export('env', 'qt_env', 'arch', 'real_arch')

# Build common module
SConscript(['common/SConscript'])
Import('_common', '_gpucommon')

common = [_common, 'json11', 'zmq']
gpucommon = [_gpucommon]

Export('common', 'gpucommon')

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


# Build other submodules
SConscript(['panda/SConscript'])

# Build rednose library
SConscript(['rednose/SConscript'])

# Build system services
SConscript([
  'system/ubloxd/SConscript',
  'system/loggerd/SConscript',
])
if arch != "Darwin":
  SConscript([
    'system/logcatd/SConscript',
    'system/proclogd/SConscript',
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

external_sconscript = GetOption('external_sconscript')
if external_sconscript:
  SConscript([external_sconscript])
