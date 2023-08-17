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

AddOption('--extras',
          action='store_true',
          help='build misc extras, like setup and installer files')

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

AddOption('--no-test',
          action='store_false',
          dest='test',
          default=os.path.islink(Dir('#laika/').abspath),
          help='skip building test files')

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
  "LD_LIBRARY_PATH": [Dir(f"#third_party/acados/{arch}/lib").abspath],
  "PYTHONPATH": Dir("#").abspath + ':' + Dir(f"#third_party/acados").abspath,

  "ACADOS_SOURCE_DIR": Dir("#third_party/acados").abspath,
  "ACADOS_PYTHON_INTERFACE_PATH": Dir("#third_party/acados/acados_template").abspath,
  "TERA_PATH": Dir("#").abspath + f"/third_party/acados/{arch}/t_renderer"
}

rpath = lenv["LD_LIBRARY_PATH"].copy()

if arch == "larch64":
  lenv["LD_LIBRARY_PATH"] += ['/data/data/com.termux/files/usr/lib']

  cpppath = [
    "#third_party/opencl/include",
  ]

  libpath = [
    "/usr/local/lib",
    "/usr/lib",
    "/system/vendor/lib64",
    f"#third_party/acados/{arch}/lib",
  ]

  libpath += [
    "#third_party/snpe/larch64",
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
  rpath += [
    Dir("#cereal").abspath,
    Dir("#common").abspath
  ]

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
    lenv["DYLD_LIBRARY_PATH"] = lenv["LD_LIBRARY_PATH"]
  # Linux
  else:
    libpath = [
      f"#third_party/acados/{arch}/lib",
      f"#third_party/libyuv/{arch}/lib",
      f"#third_party/mapbox-gl-native-qt/{arch}",
      "#cereal",
      "#common",
      "/usr/lib",
      "/usr/local/lib",
    ]

    if arch == "x86_64":
      libpath += [
        f"#third_party/snpe/{arch}"
      ]
      rpath += [
        Dir(f"#third_party/snpe/{arch}").abspath,
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

# Enable swaglog include in submodules
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

if arch == "Darwin":
  # RPATH is not supported on macOS, instead use the linker flags
  darwin_rpath_link_flags = [f"-Wl,-rpath,{path}" for path in env["RPATH"]]
  env["LINKFLAGS"] += darwin_rpath_link_flags

if GetOption('compile_db'):
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

SHARED = False

# TODO: this can probably be removed
def abspath(x):
  if arch == 'aarch64':
    pth = os.path.join("/data/pythonpath", x[0].path)
    env.Depends(pth, x)
    return File(pth)
  else:
    # rpath works elsewhere
    return x[0].path.rsplit("/", 1)[1][:-3]

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

Export('envCython')

# Qt build environment
qt_env = env.Clone()
qt_modules = ["Widgets", "Gui", "Core", "Network", "Concurrent", "Multimedia", "Quick", "Qml", "QuickWidgets", "Location", "Positioning", "DBus", "Xml"]

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
    f"{qt_install_headers}/QtGui/5.12.8/QtGui",
  ]
  qt_dirs += [f"{qt_install_headers}/Qt{m}" for m in qt_modules]

  qt_libs = [f"Qt5{m}" for m in qt_modules]
  if arch == "larch64":
    qt_libs += ["GLESv2", "wayland-client"]
    qt_env.PrependENVPath('PATH', Dir("#third_party/qt5/larch64/bin/").abspath)
  elif arch != "Darwin":
    qt_libs += ["GL"]
qt_env['QT3DIR'] = qt_env['QTDIR']

# compatibility for older SCons versions
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
qt_env['QT3_MOCHPREFIX'] = cache_dir + '/moc_files/moc_'

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

Export('env', 'qt_env', 'arch', 'real_arch', 'SHARED')

SConscript(['common/SConscript'])
Import('_common', '_gpucommon')

if SHARED:
  common, gpucommon = abspath(common), abspath(gpucommon)
else:
  common = [_common, 'json11']
  gpucommon = [_gpucommon]

Export('common', 'gpucommon')

# cereal and messaging are shared with the system
SConscript(['cereal/SConscript'])
if SHARED:
  cereal = abspath([File('cereal/libcereal_shared.so')])
  messaging = abspath([File('cereal/libmessaging_shared.so')])
else:
  cereal = [File('#cereal/libcereal.a')]
  messaging = [File('#cereal/libmessaging.a')]
  visionipc = [File('#cereal/libvisionipc.a')]

Export('cereal', 'messaging', 'visionipc')

# Build rednose library and ekf models

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

if arch != "larch64":
  rednose_config['to_build'].update({
    'loc_4': ('#selfdrive/locationd/models/loc_kf.py', True, [], rednose_deps),
    'lane': ('#selfdrive/locationd/models/lane_kf.py', True, [], rednose_deps),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', False, [], []),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', False, [], []),
  })

Export('rednose_config')
SConscript(['rednose/SConscript'])

# Build system services
SConscript([
  'system/clocksd/SConscript',
  'system/proclogd/SConscript',
  'system/ubloxd/SConscript',
  'system/loggerd/SConscript',
])
if arch != "Darwin":
  SConscript([
    'system/camerad/SConscript',
    'system/sensord/SConscript',
    'system/logcatd/SConscript',
  ])

# Build openpilot

# build submodules
SConscript([
  'body/board/SConscript',
  'cereal/SConscript',
  'opendbc/can/SConscript',
  'panda/SConscript',
])

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

if (arch in ['x86_64', 'aarch64', 'Darwin'] and Dir('#tools/cabana/').exists()) or GetOption('extras'):
  SConscript(['tools/replay/SConscript'])
  SConscript(['tools/cabana/SConscript'])

external_sconscript = GetOption('external_sconscript')
if external_sconscript:
  SConscript([external_sconscript])
