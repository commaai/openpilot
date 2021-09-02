import os
import shutil
import subprocess
import sys
import sysconfig
import platform
import numpy as np

TICI = os.path.isfile('/TICI')
Decider('MD5-timestamp')

AddOption('--test',
          action='store_true',
          help='build test files')

AddOption('--setup',
          action='store_true',
          help='build setup and installer files')

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

AddOption('--mpc-generate',
          action='store_true',
          help='regenerates the mpc sources')

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

real_arch = arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

if arch == "aarch64" and TICI:
  arch = "larch64"

USE_WEBCAM = os.getenv("USE_WEBCAM") is not None

lenv = {
  "PATH": os.environ['PATH'],
  "LD_LIBRARY_PATH": [Dir(f"#phonelibs/acados/{arch}/lib").abspath],
  "PYTHONPATH": Dir("#").abspath + ":" + Dir("#pyextra/").abspath,

  "ACADOS_SOURCE_DIR": Dir("#phonelibs/acados/acados").abspath,
  "TERA_PATH": Dir("#").abspath + f"/phonelibs/acados/{arch}/t_renderer",
}

rpath = lenv["LD_LIBRARY_PATH"].copy()

if arch == "aarch64" or arch == "larch64":
  lenv["LD_LIBRARY_PATH"] += ['/data/data/com.termux/files/usr/lib']

  if arch == "aarch64":
    # android
    lenv["ANDROID_DATA"] = os.environ['ANDROID_DATA']
    lenv["ANDROID_ROOT"] = os.environ['ANDROID_ROOT']

  cpppath = [
    "#phonelibs/opencl/include",
  ]

  libpath = [
    "/usr/local/lib",
    "/usr/lib",
    "/system/vendor/lib64",
    "/system/comma/usr/lib",
    "#phonelibs/nanovg",
    f"#phonelibs/acados/{arch}/lib",
  ]

  if arch == "larch64":
    libpath += [
      "#phonelibs/snpe/larch64",
      "#phonelibs/libyuv/larch64/lib",
      "/usr/lib/aarch64-linux-gnu"
    ]
    cpppath += [
      "#selfdrive/camerad/include",
    ]
    cflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    rpath += ["/usr/local/lib"]
  else:
    rpath = []
    libpath += [
      "#phonelibs/snpe/aarch64",
      "#phonelibs/libyuv/lib",
      "/system/vendor/lib64"
    ]
    cflags = ["-DQCOM", "-D_USING_LIBCXX", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM", "-D_USING_LIBCXX", "-mcpu=cortex-a57"]
else:
  cflags = []
  cxxflags = []
  cpppath = []

  if arch == "Darwin":
    yuv_dir = "mac" if real_arch != "arm64" else "mac_arm64"
    libpath = [
      f"#phonelibs/libyuv/{yuv_dir}/lib",
      "/usr/local/lib",
      "/opt/homebrew/lib",
      "/usr/local/opt/openssl/lib",
      "/opt/homebrew/opt/openssl/lib",
      "/System/Library/Frameworks/OpenGL.framework/Libraries",
    ]
    cflags += ["-DGL_SILENCE_DEPRECATION"]
    cxxflags += ["-DGL_SILENCE_DEPRECATION"]
    cpppath += [
      "/opt/homebrew/include",
      "/usr/local/opt/openssl/include",
      "/opt/homebrew/opt/openssl/include"
    ]
  else:
    libpath = [
      "#phonelibs/acados/x86_64/lib",
      "#phonelibs/snpe/x86_64-linux-clang",
      "#phonelibs/libyuv/x64/lib",
      "#phonelibs/mapbox-gl-native-qt/x86_64",
      "#cereal",
      "#selfdrive/common",
      "/usr/lib",
      "/usr/local/lib",
    ]

  rpath += [
    Dir("#phonelibs/snpe/x86_64-linux-clang").abspath,
    Dir("#cereal").abspath,
    Dir("#selfdrive/common").abspath
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
cflags += ["-DSWAGLOG"]
cxxflags += ["-DSWAGLOG"]

env = Environment(
  ENV=lenv,
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wno-unknown-warning-option",
    "-Wno-deprecated-register",
    "-Wno-register",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
  ] + cflags + ccflags,

  CPPPATH=cpppath + [
    "#",
    "#phonelibs/acados/include",
    "#phonelibs/acados/include/blasfeo/include",
    "#phonelibs/acados/include/hpipm/include",
    "#phonelibs/catch2/include",
    "#phonelibs/bzip2",
    "#phonelibs/libyuv/include",
    "#phonelibs/openmax/include",
    "#phonelibs/json11",
    "#phonelibs/curl/include",
    "#phonelibs/libgralloc/include",
    "#phonelibs/android_frameworks_native/include",
    "#phonelibs/android_hardware_libhardware/include",
    "#phonelibs/android_system_core/include",
    "#phonelibs/linux/include",
    "#phonelibs/snpe/include",
    "#phonelibs/mapbox-gl-native-qt/include",
    "#phonelibs/nanovg",
    "#phonelibs/qrcode",
    "#phonelibs",
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
    "#phonelibs",
    "#opendbc/can",
    "#selfdrive/boardd",
    "#selfdrive/common",
  ],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  tools=["default", "cython", "compilation_db"],
)

if GetOption('compile_db'):
  env.CompilationDatabase('compile_commands.json')

# Setup cache dir
cache_dir = '/data/scons_cache' if TICI else '/tmp/scons_cache'
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

def abspath(x):
  if arch == 'aarch64':
    pth = os.path.join("/data/pythonpath", x[0].path)
    env.Depends(pth, x)
    return File(pth)
  else:
    # rpath works elsewhere
    return x[0].path.rsplit("/", 1)[1][:-3]

# Cython build enviroment
py_include = sysconfig.get_paths()['include']
envCython = env.Clone()
envCython["CPPPATH"] += [py_include, np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-deprecated-declarations"]

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"] = ["-shared"]
  envCython["LIBS"] = [os.path.basename(py_include)]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

Export('envCython')

# Qt build environment
qt_env = env.Clone()
qt_modules = ["Widgets", "Gui", "Core", "Network", "Concurrent", "Multimedia", "Quick", "Qml", "QuickWidgets", "Location", "Positioning"]
if arch != "aarch64":
  qt_modules += ["DBus"]

qt_libs = []
if arch == "Darwin":
  if real_arch == "arm64":
    qt_env['QTDIR'] = "/opt/homebrew/opt/qt@5"
  else:
    qt_env['QTDIR'] = "/usr/local/opt/qt@5"
  qt_dirs = [
    os.path.join(qt_env['QTDIR'], "include"),
  ]
  qt_dirs += [f"{qt_env['QTDIR']}/include/Qt{m}" for m in qt_modules]
  qt_env["LINKFLAGS"] += ["-F" + os.path.join(qt_env['QTDIR'], "lib")]
  qt_env["FRAMEWORKS"] += [f"Qt{m}" for m in qt_modules] + ["OpenGL"]
elif arch == "aarch64":
  qt_env['QTDIR'] = "/system/comma/usr"
  qt_dirs = [
    f"/system/comma/usr/include/qt",
  ]
  qt_dirs += [f"/system/comma/usr/include/qt/Qt{m}" for m in qt_modules]

  qt_libs = [f"Qt5{m}" for m in qt_modules]
  qt_libs += ['EGL', 'GLESv3', 'c++_shared']
else:
  qt_env['QTDIR'] = "/usr"
  qt_dirs = [
    f"/usr/include/{real_arch}-linux-gnu/qt5",
    f"/usr/include/{real_arch}-linux-gnu/qt5/QtGui/5.12.8/QtGui",
  ]
  qt_dirs += [f"/usr/include/{real_arch}-linux-gnu/qt5/Qt{m}" for m in qt_modules]

  qt_libs = [f"Qt5{m}" for m in qt_modules]
  if arch == "larch64":
    qt_libs += ["GLESv2", "wayland-client"]
  elif arch != "Darwin":
    qt_libs += ["GL"]

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

Export('env', 'qt_env', 'arch', 'real_arch', 'SHARED', 'USE_WEBCAM')

SConscript(['selfdrive/common/SConscript'])
Import('_common', '_gpucommon', '_gpu_libs')

if SHARED:
  common, gpucommon = abspath(common), abspath(gpucommon)
else:
  common = [_common, 'json11']
  gpucommon = [_gpucommon] + _gpu_libs

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

rednose_config = {
  'generated_folder': '#selfdrive/locationd/models/generated',
  'to_build': {
    'live': ('#selfdrive/locationd/models/live_kf.py', True, ['live_kf_constants.h']),
    'car': ('#selfdrive/locationd/models/car_kf.py', True, []),
  },
}

if arch != "aarch64":
  rednose_config['to_build'].update({
    'gnss': ('#selfdrive/locationd/models/gnss_kf.py', True, []),
    'loc_4': ('#selfdrive/locationd/models/loc_kf.py', True, []),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', False, []),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', False, []),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', False, []),
    'lane': ('#xx/pipeline/lib/ekf/lane_kf.py', True, []),
  })

Export('rednose_config')
SConscript(['rednose/SConscript'])

# Build openpilot

SConscript(['cereal/SConscript'])
SConscript(['panda/board/SConscript'])
SConscript(['opendbc/can/SConscript'])

SConscript(['phonelibs/SConscript'])

SConscript(['common/SConscript'])
SConscript(['common/kalman/SConscript'])
SConscript(['common/transformations/SConscript'])

SConscript(['selfdrive/camerad/SConscript'])
SConscript(['selfdrive/modeld/SConscript'])

SConscript(['selfdrive/controls/lib/cluster/SConscript'])
SConscript(['selfdrive/controls/lib/lateral_mpc_lib/SConscript'])
SConscript(['selfdrive/controls/lib/lead_mpc_lib/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc_lib/SConscript'])

SConscript(['selfdrive/boardd/SConscript'])
SConscript(['selfdrive/proclogd/SConscript'])
SConscript(['selfdrive/clocksd/SConscript'])

SConscript(['selfdrive/loggerd/SConscript'])

SConscript(['selfdrive/locationd/SConscript'])
SConscript(['selfdrive/sensord/SConscript'])
SConscript(['selfdrive/ui/SConscript'])

if arch != "Darwin":
  SConscript(['selfdrive/logcatd/SConscript'])

external_sconscript = GetOption('external_sconscript')
if external_sconscript:
  SConscript([external_sconscript])
