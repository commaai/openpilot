import Cython
import distutils
import os
import shutil
import subprocess
import sys
import platform
import numpy as np
from sysconfig import get_paths

TICI = os.path.isfile('/TICI')
Decider('MD5-timestamp')

AddOption('--test',
          action='store_true',
          help='build test files')

AddOption('--asan',
          action='store_true',
          help='turn on ASAN')

# Rebuild cython extensions if python, distutils, or cython change
cython_dependencies = [Value(v) for v in (sys.version, distutils.__version__, Cython.__version__)]
Export('cython_dependencies')

real_arch = arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"

if arch == "aarch64" and TICI:
  arch = "larch64"

USE_WEBCAM = os.getenv("USE_WEBCAM") is not None
QCOM_REPLAY = arch == "aarch64" and os.getenv("QCOM_REPLAY") is not None

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
    "/system/vendor/lib64",
    "/system/comma/usr/lib",
    "#phonelibs/nanovg",
  ]

  if arch == "larch64":
    libpath += [
      "#phonelibs/snpe/larch64",
      "#phonelibs/libyuv/larch64/lib",
      "/usr/lib/aarch64-linux-gnu"
    ]
    cflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM2", "-mcpu=cortex-a57"]
    rpath = ["/usr/local/lib"]
  else:
    libpath += [
      "#phonelibs/snpe/aarch64",
      "#phonelibs/libyuv/lib",
      "/system/vendor/lib64"
    ]
    cflags = ["-DQCOM", "-mcpu=cortex-a57"]
    cxxflags = ["-DQCOM", "-mcpu=cortex-a57"]
    rpath = []

    if QCOM_REPLAY:
      cflags += ["-DQCOM_REPLAY"]
      cxxflags += ["-DQCOM_REPLAY"]

else:
  cflags = []
  cxxflags = []

  lenv = {
    "PATH": "#external/bin:" + os.environ['PATH'],
  }
  cpppath = [
    "#external/tensorflow/include",
  ]

  if arch == "Darwin":
    libpath = [
      "#phonelibs/libyuv/mac/lib",
      "#cereal",
      "#selfdrive/common",
      "/usr/local/lib",
      "/System/Library/Frameworks/OpenGL.framework/Libraries",
    ]
    cflags += ["-DGL_SILENCE_DEPRECATION"]
    cxxflags += ["-DGL_SILENCE_DEPRECATION"]
  else:
    libpath = [
      "#phonelibs/snpe/x86_64-linux-clang",
      "#phonelibs/libyuv/x64/lib",
      "#external/tensorflow/lib",
      "#cereal",
      "#selfdrive/common",
      "/usr/lib",
      "/usr/local/lib",
    ]

  rpath = [
    "phonelibs/snpe/x86_64-linux-clang",
    "external/tensorflow/lib",
    "cereal",
    "selfdrive/common"
  ]

  # allows shared libraries to work globally
  rpath = [os.path.join(os.getcwd(), x) for x in rpath]

if GetOption('asan'):
  ccflags_asan = ["-fsanitize=address", "-fno-omit-frame-pointer"]
  ldflags_asan = ["-fsanitize=address"]
else:
  ccflags_asan = []
  ldflags_asan = []

# change pythonpath to this
lenv["PYTHONPATH"] = Dir("#").path

#Get the path for Python.h for cython linking
python_path = get_paths()['include']
numpy_path = np.get_include()

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
  ] + cflags + ccflags_asan,

  CPPPATH=cpppath + [
    "#",
    "#selfdrive",
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
    "#phonelibs/nanovg",
    "#selfdrive/boardd",
    "#selfdrive/common",
    "#selfdrive/camerad",
    "#selfdrive/camerad/include",
    "#selfdrive/loggerd/include",
    "#selfdrive/modeld",
    "#selfdrive/sensord",
    "#selfdrive/ui",
    "#cereal/messaging",
    "#cereal",
    "#opendbc/can",
  ],

  CC='clang',
  CXX='clang++',
  LINKFLAGS=ldflags_asan,

  RPATH=rpath,

  CFLAGS=["-std=gnu11"] + cflags,
  CXXFLAGS=["-std=c++1z"] + cxxflags,
  LIBPATH=libpath + [
    "#cereal",
    "#selfdrive/boardd",
    "#selfdrive/common",
    "#phonelibs",
  ],
  CYTHONCFILESUFFIX=".cpp",
  tools=["default", "cython"]
)
  
if os.environ.get('SCONS_CACHE'):
  cache_dir = '/tmp/scons_cache'

  if os.getenv('CI'):
    branch = os.getenv('GIT_BRANCH')

    if QCOM_REPLAY:
      cache_dir = '/tmp/scons_cache_qcom_replay'
    elif branch is not None and branch != 'master':
      cache_dir_branch = '/tmp/scons_cache_' + branch
      if not os.path.isdir(cache_dir_branch) and os.path.isdir(cache_dir):
        shutil.copytree(cache_dir, cache_dir_branch)
      cache_dir = cache_dir_branch
  CacheDir(cache_dir)

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

#Cython build enviroment
envCython = env.Clone()
envCython["CPPPATH"] += [python_path, numpy_path]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-deprecated-declarations"]

python_libs = []
if arch == "Darwin":
  envCython["LINKFLAGS"]=["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "aarch64":
  envCython["LINKFLAGS"]=["-shared"]

  python_libs.append(os.path.basename(python_path))
else:
  envCython["LINKFLAGS"]=["-pthread", "-shared"]

envCython["LIBS"] = python_libs

Export('envCython')

# still needed for apks
zmq = 'zmq'
Export('env', 'arch', 'real_arch', 'zmq', 'SHARED', 'USE_WEBCAM', 'QCOM_REPLAY')

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
  common = [_common, 'json11']
  visionipc = _visionipc
  gpucommon = [_gpucommon] + _gpu_libs

Export('common', 'visionipc', 'gpucommon')

SConscript(['opendbc/can/SConscript'])

SConscript(['common/SConscript'])
SConscript(['common/kalman/SConscript'])
SConscript(['common/transformations/SConscript'])
SConscript(['phonelibs/SConscript'])

SConscript(['selfdrive/camerad/SConscript'])
SConscript(['selfdrive/modeld/SConscript'])

SConscript(['selfdrive/controls/lib/cluster/SConscript'])
SConscript(['selfdrive/controls/lib/lateral_mpc/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc/SConscript'])
SConscript(['selfdrive/controls/lib/longitudinal_mpc_model/SConscript'])

SConscript(['selfdrive/boardd/SConscript'])
SConscript(['selfdrive/proclogd/SConscript'])
SConscript(['selfdrive/clocksd/SConscript'])

SConscript(['selfdrive/loggerd/SConscript'])

SConscript(['selfdrive/locationd/SConscript'])
SConscript(['selfdrive/locationd/models/SConscript'])
SConscript(['selfdrive/sensord/SConscript'])
SConscript(['selfdrive/ui/SConscript'])

if arch != "Darwin":
  SConscript(['selfdrive/logcatd/SConscript'])

if arch == "x86_64":
  SConscript(['tools/lib/index_log/SConscript'])
