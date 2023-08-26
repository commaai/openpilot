import os
import subprocess
import sys
import sysconfig
import platform
import numpy as np

import SCons.Errors

SCons.Warnings.warningAsException(True)

TICI = os.path.isfile('/TICI')
AGNOS = TICI

Decider('MD5-timestamp')

AddOption('--kaitai',
          action='store_true',
          help='Regenerate kaitai struct parsers')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

AddOption('--ubsan',
          action='store_true',
          help='turn on UBSan')

AddOption('--clazy',
          action='store_true',
          help='build with clazy')

AddOption('--compile_db',
          action='store_true',
          help='build clang compilation database')

AddOption('--snpe',
          action='store_true',
          help='use SNPE on PC')

AddOption('--external-sconscript',
          action='store',
          metavar='FILE',
          dest='external_sconscript',
          help='add an external SConscript to the build')

AddOption('--no-thneed',
          action='store_true',
          dest='no_thneed',
          help='avoid using thneed')

AddOption('--pc-thneed',
          action='store_true',
          dest='pc_thneed',
          help='use thneed on pc')

AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=os.path.islink(Dir('#laika/').abspath),
          help='the minimum build to run openpilot. no tests, tools, etc.')

# *** Target and Architecture

## Target name breakdown (target)
## - agnos-aarch64: linux tici aarch64
## - linux-aarch64: linux pc aarch64
## - linux-x86_64:  linux pc x64
## - Darwin:        mac x64 or arm64
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
system = platform.system()
if system == "Darwin":
  target = "Darwin"
  brew_prefix = subprocess.check_output(['brew', '--prefix'], encoding='utf8').strip()
elif system == "Linux":
  if arch == "aarch64" and AGNOS:
    target = "agnos-aarch64"
  else:
    target = f"linux-{arch}"
else:
  raise Exception(f"Unsupported platform: {system}")

assert target in ["agnos-aarch64", "linux-aarch64", "linux-x86_64", "Darwin"]

# *** Environment setup
lenv = {
  "PATH": os.environ['PATH'],
  "LD_LIBRARY_PATH": [Dir(f"#third_party/acados/{target}/lib").abspath],
  "PYTHONPATH": Dir("#").abspath + ':' + Dir(f"#third_party/acados").abspath,

  "ACADOS_SOURCE_DIR": Dir("#third_party/acados").abspath,
  "ACADOS_PYTHON_INTERFACE_PATH": Dir("#third_party/acados/acados_template").abspath,
  "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{target}/t_renderer"
}
rpath = lenv["LD_LIBRARY_PATH"].copy()

if target == "agnos-aarch64":
  cflags = ["-DQCOM2", "-mcpu=cortex-a57"]
  cxxflags = ["-DQCOM2", "-mcpu=cortex-a57"]
  cpppath = [
    "#third_party/opencl/include",
  ]
  libpath = [
    "#third_party/acados/agnos-aarch64/lib",
    "#third_party/snpe/agnos-aarch64",
    "#third_party/libyuv/agnos-aarch64/lib",
    "/usr/local/lib",
    "/usr/lib",
    "/system/vendor/lib64",
    "/usr/lib/aarch64-linux-gnu"
  ]
  rpath += ["/usr/local/lib"]
  lenv["LD_LIBRARY_PATH"] += ['/data/data/com.termux/files/usr/lib']
else:
  cflags = []
  cxxflags = []
  cpppath = []
  rpath += [
    Dir("#cereal").abspath,
    Dir("#common").abspath
  ]

  # MacOS
  if target == "Darwin":
    libpath = [
      f"#third_party/libyuv/{target}/lib",
      f"#third_party/acados/{target}/lib",
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
    lenv["DYLD_LIBRARY_PATH"] = lenv["LD_LIBRARY_PATH"]
  # Linux
  else:
    libpath = [
      f"#third_party/acados/{target}/lib",
      f"#third_party/libyuv/{target}/lib",
      f"#third_party/mapbox-gl-native-qt/{target}",
      "/usr/local/lib",
      "/usr/lib",
    ]

    if target == "linux-x86_64":
      libpath += [
        f"#third_party/snpe/{target}"
      ]
      rpath += [
        Dir(f"#third_party/snpe/{target}").abspath,
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

## no --as-needed on mac linker
if target != "Darwin":
  ldflags += ["-Wl,--as-needed", "-Wl,--no-undefined"]

## Enable swaglog include in submodules
cflags += ['-DSWAGLOG="\\"common/swaglog.h\\""']
cxxflags += ['-DSWAGLOG="\\"common/swaglog.h\\""']

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
    "-Wno-deprecated-register",
    "-Wno-register",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
    "-Wno-error=unused-but-set-variable",
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
    "#third_party/snpe/include",
    "#third_party/mapbox-gl-native-qt/include",
    "#third_party/qrcode",
    "#third_party",
    "#cereal",
    "#opendbc/can",
  ],

  CC='clang',
  CXX='clang++',
  LINKFLAGS=ldflags,

  RPATH=rpath,

  CFLAGS=["-std=gnu11"] + cflags,
  CXXFLAGS=["-std=c++1z"] + cxxflags,
  LIBPATH=libpath + [
    "#cereal",
    "#third_party",
    "#opendbc/can",
    "#selfdrive/boardd",
    "#common",
  ],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  tools=["default", "cython", "compilation_db"],
)

## RPATH is not supported on macOS, instead use the linker flags
if target == "Darwin":
  Darwin_rpath_link_flags = [f"-Wl,-rpath,{path}" for path in env["RPATH"]]
  env["LINKFLAGS"] += Darwin_rpath_link_flags

if GetOption('compile_db'):
  env.CompilationDatabase('compile_commands.json')

## Setup cache dir
cache_dir = '/data/scons_cache' if AGNOS else '/tmp/scons_cache'
CacheDir(cache_dir)
Clean(["."], cache_dir)

## Setup optional progress bar
node_interval = 5
node_count = 0
def progress_function(node):
  global node_count
  node_count += node_interval
  sys.stderr.write("progress: %d\n" % node_count)

if os.environ.get('SCONS_PROGRESS'):
  Progress(progress_function, interval=node_interval)

# *** Cython build environment
py_include = sysconfig.get_paths()['include']
envCython = env.Clone()
envCython["CPPPATH"] += [py_include, np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

envCython["LIBS"] = []
if target == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"] + Darwin_rpath_link_flags
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

# *** Qt build environment
qt_env = env.Clone()
qt_modules = ["Widgets", "Gui", "Core", "Network", "Concurrent", "Multimedia", "Quick", "Qml", "QuickWidgets", "Location", "Positioning", "DBus", "Xml"]

qt_libs = []
if target == "Darwin":
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
    f"{qt_install_headers}/QtGui/5.12.8/QtGui",
  ]
  qt_dirs += [f"{qt_install_headers}/Qt{m}" for m in qt_modules]

  qt_libs = [f"Qt5{m}" for m in qt_modules] + ["GL"]
  if target == "agnos-aarch64":
    qt_libs += ["GLESv2", "wayland-client"]
    qt_env.PrependENVPath('PATH', Dir("#third_party/qt5/agnos-aarch64/bin/").abspath)
qt_env['QT3DIR'] = qt_env['QTDIR']

## compatibility for older SCons versions
try:
  qt_env.Tool('qt3')
except SCons.Errors.UserError:
  qt_env.Tool('qt')

qt_env['CPPPATH'] += qt_dirs + ["#selfdrive/ui/qt/"]
qt_flags = [
  "-D_REENTRANT",
  "-DQT_NO_DEBUG",
  "-DQT_WIDGETS_LIB",
  "-DQT_GUI_LIB",
  "-DQT_QUICK_LIB",
  "-DQT_QUICKWIDGETS_LIB",
  "-DQT_QML_LIB",
  "-DQT_CORE_LIB",
  "-DQT_MESSAGELOGCONTEXT",
]
qt_env['CXXFLAGS'] += qt_flags
qt_env['LIBPATH'] += ['#selfdrive/ui']
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

# *** Export environments and global variables
Export('env', 'envCython', 'qt_env', 'target', 'system', 'arch')

# *** Build common code
SConscript(['common/SConscript'])

# *** Build cereal and messaging
SConscript(['cereal/SConscript'])
cereal = [File('#cereal/libcereal.a')]
messaging = [File('#cereal/libmessaging.a')]
visionipc = [File('#cereal/libvisionipc.a')]
Export('cereal', 'messaging', 'visionipc')

# *** Build other submodules
SConscript([
  'body/board/SConscript',
  'opendbc/can/SConscript',
  'panda/SConscript',
])

# *** Build rednose library and ekf models
rednose_deps = [
  "#selfdrive/locationd/models/constants.py",
  "#selfdrive/locationd/models/gnss_helpers.py",
]

rednose_config = {
  'generated_folder': '#selfdrive/locationd/models/generated',
  'to_build': {
    'gnss': ('#selfdrive/locationd/models/gnss_kf.py', True, [], rednose_deps),
    'live': ('#selfdrive/locationd/models/live_kf.py', True, ['live_kf_constants.h'], rednose_deps),
    'car': ('#selfdrive/locationd/models/car_kf.py', True, [], rednose_deps),
  },
}

if target != "agnos-aarch64":
  rednose_config['to_build'].update({
    'loc_4': ('#selfdrive/locationd/models/loc_kf.py', True, [], rednose_deps),
    'lane': ('#selfdrive/locationd/models/lane_kf.py', True, [], rednose_deps),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', False, [], []),
  })

Export('rednose_config')
SConscript(['rednose/SConscript'])

# *** Build system services
SConscript([
  'system/clocksd/SConscript',
  'system/proclogd/SConscript',
  'system/ubloxd/SConscript',
  'system/loggerd/SConscript',
])
if target != "Darwin":
  SConscript([
    'system/camerad/SConscript',
    'system/sensord/SConscript',
    'system/logcatd/SConscript',
  ])

# *** Build openpilot
SConscript(['third_party/SConscript'])

SConscript(['common/kalman/SConscript'])
SConscript(['common/transformations/SConscript'])

SConscript(['selfdrive/boardd/SConscript'])
SConscript(['selfdrive/controls/lib/lateral_mpc_lib/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc_lib/SConscript'])
SConscript(['selfdrive/locationd/SConscript'])
SConscript(['selfdrive/navd/SConscript'])
SConscript(['selfdrive/modeld/SConscript'])
SConscript(['selfdrive/ui/SConscript'])

if (target in ['linux-x86_64', 'linux-aarch64', 'Darwin'] and Dir('#tools/cabana/').exists()) or GetOption('extras'):
  SConscript(['tools/replay/SConscript'])
  SConscript(['tools/cabana/SConscript'])

external_sconscript = GetOption('external_sconscript')
if external_sconscript:
  SConscript([external_sconscript])
