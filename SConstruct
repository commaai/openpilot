import os
import subprocess
import sys
import platform

AddOption('--test',
          action='store_true',
          help='build test files')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"
if arch == "aarch64" and not os.path.isdir("/system"):
  arch = "larch64"

webcam = bool(ARGUMENTS.get("use_webcam", 0))

if arch == "aarch64" or arch == "larch64":
  lenv = {
    "LD_LIBRARY_PATH": '/data/data/com.termux/files/usr/lib',
    "PATH": os.environ['PATH'],
  }

  if arch == "aarch64":
    # android
    lenv["ANDROID_DATA"] = os.environ['ANDROID_DATA']
    lenv["ANDROID_ROOT"] = os.environ['ANDROID_ROOT']

  cpppath = [
    "#phonelibs/opencl/include",
  ]

  libpath = [
    "/usr/lib",
    "/data/data/com.termux/files/usr/lib",
    "/system/vendor/lib64",
    "/system/comma/usr/lib",
    "#phonelibs/nanovg",
  ]

  if arch == "larch64":
    cpppath += ["#phonelibs/capnp-cpp/include", "#phonelibs/capnp-c/include"]
    libpath += ["#phonelibs/snpe/larch64"]
    libpath += ["#phonelibs/libyuv/larch64/lib"]
    libpath += ["#external/capnparm/lib", "/usr/lib/aarch64-linux-gnu"]
    cflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    rpath = ["/usr/local/lib"]
  else:
    libpath += ["#phonelibs/snpe/aarch64"]
    libpath += ["#phonelibs/libyuv/lib"]
    cflags = ["-DQCOM", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM", "-mcpu=cortex-a57"]
    rpath = ["/system/vendor/lib64"]

else:
  lenv = {
    "PATH": "#external/bin:" + os.environ['PATH'],
  }
  cpppath = [
    "#phonelibs/capnp-cpp/include",
    "#phonelibs/capnp-c/include",
    "#phonelibs/zmq/x64/include",
    "#external/tensorflow/include",
  ]

  if arch == "Darwin":
    libpath = [
      "#phonelibs/capnp-cpp/mac/lib",
      "#phonelibs/capnp-c/mac/lib",
      "#phonelibs/libyuv/mac/lib",
      "#cereal",
      "#selfdrive/common",
      "/usr/local/lib",
      "/System/Library/Frameworks/OpenGL.framework/Libraries",
    ]
  else:
    libpath = [
      "#phonelibs/capnp-cpp/x64/lib",
      "#phonelibs/capnp-c/x64/lib",
      "#phonelibs/snpe/x86_64-linux-clang",
      "#phonelibs/zmq/x64/lib",
      "#phonelibs/libyuv/x64/lib",
      "#external/zmq/lib",
      "#external/tensorflow/lib",
      "#cereal",
      "#selfdrive/common",
      "/usr/lib",
      "/usr/local/lib",
    ]

  rpath = ["phonelibs/capnp-cpp/x64/lib",
           "phonelibs/zmq/x64/lib",
           "external/tensorflow/lib",
           "cereal",
           "selfdrive/common"]

  # allows shared libraries to work globally
  rpath = [os.path.join(os.getcwd(), x) for x in rpath]

  cflags = []
  cxxflags = []

ccflags_asan = ["-fsanitize=address", "-fno-omit-frame-pointer"] if GetOption('asan') else []
ldflags_asan = ["-fsanitize=address"] if GetOption('asan') else []

# change pythonpath to this
lenv["PYTHONPATH"] = Dir("#").path

env = Environment(
  ENV=lenv,
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-O2",
    "-Werror=implicit-function-declaration",
    "-Werror=incompatible-pointer-types",
    "-Werror=int-conversion",
    "-Werror=return-type",
    "-Werror=format-extra-args",
  ] + cflags + ccflags_asan,

  CPPPATH=cpppath + [
    "#",
    "#selfdrive",
    "#phonelibs/bzip2",
    "#phonelibs/libyuv/include",
    "#phonelibs/openmax/include",
    "#phonelibs/json/src",
    "#phonelibs/json11",
    "#phonelibs/eigen",
    "#phonelibs/curl/include",
    #"#phonelibs/opencv/include", # use opencv4 instead
    "#phonelibs/libgralloc/include",
    "#phonelibs/android_frameworks_native/include",
    "#phonelibs/android_hardware_libhardware/include",
    "#phonelibs/android_system_core/include",
    "#phonelibs/linux/include",
    "#phonelibs/snpe/include",
    "#phonelibs/nanovg",
    "#selfdrive/common",
    "#selfdrive/camerad",
    "#selfdrive/camerad/include",
    "#selfdrive/loggerd/include",
    "#selfdrive/modeld",
    "#cereal/messaging",
    "#cereal",
    "#opendbc/can",
  ],

  CC='clang',
  CXX='clang++',
  LINKFLAGS=ldflags_asan,

  RPATH=rpath,

  CFLAGS=["-std=gnu11"] + cflags,
  CXXFLAGS=["-std=c++14"] + cxxflags,
  LIBPATH=libpath +
  [
    "#cereal",
    "#selfdrive/common",
    "#phonelibs",
  ]
)

if os.environ.get('SCONS_CACHE'):
  CacheDir('/tmp/scons_cache')

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

# still needed for apks
if arch == 'larch64':
  zmq = 'zmq'
else:
  zmq = FindFile("libzmq.a", libpath)
Export('env', 'arch', 'zmq', 'SHARED', 'webcam')

# cereal and messaging are shared with the system
SConscript(['cereal/SConscript'])
if SHARED:
  cereal = abspath([File('cereal/libcereal_shared.so')])
  messaging = abspath([File('cereal/libmessaging_shared.so')])
else:
  cereal = [File('#cereal/libcereal.a')]
  messaging = [File('#cereal/libmessaging.a')]
Export('cereal', 'messaging')

SConscript(['selfdrive/common/SConscript'])
Import('_common', '_visionipc', '_gpucommon', '_gpu_libs')

if SHARED:
  common, visionipc, gpucommon = abspath(common), abspath(visionipc), abspath(gpucommon)
else:
  common = [_common, 'json']
  visionipc = _visionipc
  gpucommon = [_gpucommon] + _gpu_libs

Export('common', 'visionipc', 'gpucommon')

SConscript(['opendbc/can/SConscript'])

SConscript(['common/SConscript'])
SConscript(['common/kalman/SConscript'])
SConscript(['phonelibs/SConscript'])

if arch != "Darwin":
  SConscript(['selfdrive/camerad/SConscript'])
  SConscript(['selfdrive/modeld/SConscript'])

SConscript(['selfdrive/controls/lib/cluster/SConscript'])
SConscript(['selfdrive/controls/lib/lateral_mpc/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc_model/SConscript'])

SConscript(['selfdrive/boardd/SConscript'])
SConscript(['selfdrive/proclogd/SConscript'])

SConscript(['selfdrive/ui/SConscript'])
SConscript(['selfdrive/loggerd/SConscript'])

if arch == "aarch64":
  SConscript(['selfdrive/logcatd/SConscript'])
  SConscript(['selfdrive/sensord/SConscript'])
  SConscript(['selfdrive/clocksd/SConscript'])

SConscript(['selfdrive/locationd/SConscript'])
SConscript(['selfdrive/locationd/kalman/SConscript'])

# TODO: finish cereal, dbcbuilder, MPC
