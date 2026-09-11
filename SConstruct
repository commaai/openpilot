import os
import subprocess
import sys
import sysconfig
import platform
import shlex
import shutil
import tempfile
import importlib
import numpy as np

import SCons.Errors
from SCons.Defaults import _stripixes

COMMA_HARDWARE = os.path.isfile('/AGNOS')

SCons.Warnings.warningAsException(True)

Decider('MD5-timestamp')

SetOption('num_jobs', max(1, int(os.cpu_count()/(1 if "CI" in os.environ else 2))))

AddOption('--ccflags', action='store', type='string', default='', help='pass arbitrary flags over the command line')
AddOption('--verbose', action='store_true', default=False, help='show full build commands')
release = not os.path.exists(File('#.gitattributes').abspath) # file absent on release branch, see release_files.py
AddOption('--minimal',
          action='store_false',
          dest='extras',
          default=(not COMMA_HARDWARE and not release),
          help='the minimum build to run openpilot. no tests, tools, etc.')

submodule_python_paths = [
  Dir("#").abspath,
  Dir("#msgq_repo").abspath,
  Dir("#opendbc_repo").abspath,
  Dir("#rednose_repo").abspath,
  Dir("#teleoprtc_repo").abspath,
  Dir("#tinygrad_repo").abspath,
]
for p in reversed(submodule_python_paths):
  if p not in sys.path:
    sys.path.insert(0, p)

if external_pythonpath := os.environ.get("PYTHONPATH"):
  submodule_python_paths += [p for p in external_pythonpath.split(os.pathsep) if p and p not in submodule_python_paths]

# Detect platform
arch = subprocess.check_output(["uname", "-m"], encoding='utf8').rstrip()
if platform.system() == "Darwin":
  arch = "Darwin"
elif platform.system() == "Windows":
  arch = "Windows"
elif arch == "aarch64" and COMMA_HARDWARE:
  arch = "comma_arm64"
assert arch in [
  "comma_arm64",  # linux comma hardware (AGNOS) arm64
  "aarch64",      # linux pc arm64
  "x86_64",       # linux pc x64
  "Darwin",       # macOS arm64 (x86 not supported)
  "Windows",      # windows pc x64, development only (MSYS2 clang64 toolchain)
]
WINDOWS = arch == "Windows"

pkg_names = ['acados', 'capnproto', 'ffmpeg', 'json11', 'ncurses', 'zeromq', 'zstd']
pkgs = [importlib.import_module(name) for name in pkg_names]
acados = pkgs[pkg_names.index('acados')]
ffmpeg = pkgs[pkg_names.index('ffmpeg')]
# Shared package ships .so/.dylib; older device venvs still have static .a only.
# Keep static link deps (x264/z/va/drm) when the installed package is static so
# COMMA_HARDWARE CI works without upgrading the device venv yet.
# TODO: drop the static fallback once device venvs have comma-deps-ffmpeg>=7.1.0.post94
_ffmpeg_lib_names = os.listdir(ffmpeg.LIB_DIR) if os.path.isdir(ffmpeg.LIB_DIR) else []
ffmpeg_shared = any(
  n.startswith('libavcodec.so') or (n.startswith('libavcodec') and n.endswith(('.dylib', '.dll.a')))
  for n in _ffmpeg_lib_names
)
ffmpeg_libs = ['avformat', 'avcodec', 'swresample', 'avutil']
if not ffmpeg_shared:
  ffmpeg_libs += ['x264', 'z']
  if arch not in ("Darwin", "Windows"):
    ffmpeg_libs += ['va', 'va-drm', 'drm']
acados_include_dirs = [
  acados.INCLUDE_DIR,
  os.path.join(acados.INCLUDE_DIR, "blasfeo", "include"),
  os.path.join(acados.INCLUDE_DIR, "hpipm", "include"),
]


# ***** enforce a whitelist of system libraries *****
# this prevents silently relying on a 3rd party package,
# e.g. apt-installed libusb. all libraries should either
# be distributed with all Linux distros and macOS, or
# vendored in commaai/dependencies.
allowed_system_libs = {
  "EGL", "GLESv2", "GL",
  "dl", "drm", "gbm", "m", "pthread",
  "opengl32", "gdi32", "winmm", "shell32", "user32", "setupapi",  # Windows SDK import libraries
}
# static libzmq/capnp need these on every Windows link; import libs only pull in what is referenced
windows_link_libs = ["pthread", "ws2_32", "iphlpapi", "rpcrt4", "bcrypt", "advapi32"] if WINDOWS else []

def _resolve_lib(env, name):
  for d in env.Flatten(env.get('LIBPATH', [])):
    p = Dir(str(d)).abspath
    for ext in ('.a', '.so', '.dylib', '.dll.a'):
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
  libs += windows_link_libs
  return _stripixes(env['LIBLINKPREFIX'], libs, env['LIBLINKSUFFIX'],
                    env['LIBPREFIXES'], env['LIBSUFFIXES'], env, env['LIBLITERALPREFIX'])

if WINDOWS:
  # build commands run through MSYS2 bash: the SConscripts use POSIX shell syntax, the submodules' Environments too
  import SCons.Platform.posix
  import SCons.Platform.win32
  _bash = shutil.which("bash")
  if not _bash or "system32" in _bash.lower():  # System32's bash.exe is the WSL launcher
    raise SCons.Errors.UserError("run scons from an MSYS2 CLANG64 shell")

  def _bash_spawn(sh, escape, cmd, args, env):
    args = [a if '"' in a else a.replace("\\", "/") for a in args]  # bash reads backslashes as escapes; quoted defines stay
    return subprocess.call([_bash, "-c", " ".join(args)], env=env)
  SCons.Platform.win32.spawn = _bash_spawn
  SCons.Platform.win32.escape = SCons.Platform.posix.escape

env = Environment(
  ENV={
    # Windows processes need the system root (DLLs), a temp dir and ccache's cache dir
    **({k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "LOCALAPPDATA") if k in os.environ} if WINDOWS else {}),
    "PATH": os.environ['PATH'],
    "PYTHONPATH": os.pathsep.join(submodule_python_paths),
    "ACADOS_SOURCE_DIR": acados.DIR,
    "ACADOS_PYTHON_INTERFACE_PATH": acados.TEMPLATE_DIR,
    "TERA_PATH": acados.TERA_PATH
  },
  CCFLAGS=[
    "-g",
    "-fPIC",
    "-pipe",
    "-O2",
    "-Wunused",
    "-Werror",
    "-Wshadow" if arch in ("Darwin", "comma_arm64", "Windows") else "-Wshadow=local",
    "-Wno-unknown-warning-option",
    "-Wno-inconsistent-missing-override",
    "-Wno-c99-designator",
    "-Wno-reorder-init-list",
    "-Wno-vla-cxx-extension",
  ],
  CFLAGS=["-std=gnu11"],
  CXXFLAGS=["-std=c++1z"],
  CPPPATH=[
    "#openpilot",
    "#msgq_repo",            # #include "msgq/..."
    "#opendbc_repo",         # #include "opendbc/..."
    "#rednose_repo",         # #include "rednose/..."
    "#rednose_repo/rednose", # #include "logger/..." (rednose package root)
    "#openpilot/cereal/gen/cpp",
    acados_include_dirs,
    [x.INCLUDE_DIR for x in pkgs],
    "#",
  ],
  LIBPATH=[
    "#openpilot/common",
    "#msgq_repo",
    "#openpilot/selfdrive/pandad",
    "#rednose_repo/rednose/helpers",
    [x.LIB_DIR for x in pkgs],
  ],
  RPATH=[ffmpeg.LIB_DIR] if ffmpeg_shared else [],
  CYTHONCFILESUFFIX=".cpp",
  COMPILATIONDB_USE_ABSPATH=True,
  REDNOSE_ROOT="#rednose_repo",
  tools=["mingw" if WINDOWS else "default", "cython", "compilation_db", "rednose_filter"],
  toolpath=["#msgq_repo/site_scons/site_tools", "#rednose_repo/site_scons/site_tools"],
)
# SCons' Darwin linker tool doesn't define the variables used to expand RPATH.
if arch == "Darwin":
  env["RPATHPREFIX"] = "-Wl,-rpath,"
  env["RPATHSUFFIX"] = ""
  env["_RPATH"] = "${_concat(RPATHPREFIX, RPATH, RPATHSUFFIX, __env__)}"
if arch != "comma_arm64":
  env['_LIBFLAGS'] = _libflags
if WINDOWS:
  # clang and lld through the mingw tool, whose defaults are gcc; shared libraries keep the lib prefix the SConscripts expect
  env["CC"], env["CXX"] = "clang", "clang++"
  env["SHLIBPREFIX"] = "lib"
  # PE has no rpath; DLLs are found next to the executable or via PATH
  env["_RPATH"] = ""
  # static runtime: Python does not search PATH for the DLLs an extension module needs
  env.Append(LINKFLAGS=["-static"])

# Arch-specific flags and paths
if arch == "comma_arm64":
  env["CC"] = "clang"
  env["CXX"] = "clang++"
  env.Append(LIBPATH=[
    "/usr/lib/aarch64-linux-gnu",
  ])
  arch_flags = ["-D__COMMA_HARDWARE__", "-mcpu=cortex-a57"]
  env.Append(CCFLAGS=arch_flags)
  env.Append(CXXFLAGS=arch_flags)
elif arch == "Darwin":
  env.Append(LIBPATH=[
    "/System/Library/Frameworks/OpenGL.framework/Libraries",
  ])
  env.Append(CCFLAGS=["-DGL_SILENCE_DEPRECATION"])
  env.Append(CXXFLAGS=["-DGL_SILENCE_DEPRECATION"])
elif arch == "Windows":
  # strict -std=c++1z hides vasprintf and M_PI in mingw's headers; the vendored libzmq is a static archive
  env.Append(CCFLAGS=["-D_GNU_SOURCE", "-D_USE_MATH_DEFINES", "-DZMQ_STATIC"])

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

# ********** Cython build environment **********
envCython = env.Clone()
envCython["CPPPATH"] += [sysconfig.get_paths()['include'], np.get_include()]
envCython["CCFLAGS"] += ["-Wno-#warnings", "-Wno-cpp", "-Wno-shadow", "-Wno-deprecated-declarations"]
envCython["CCFLAGS"].remove("-Werror")

envCython["LIBS"] = []
if arch == "Darwin":
  envCython["LINKFLAGS"] = env["LINKFLAGS"] + ["-bundle", "-undefined", "dynamic_lookup"]
elif arch == "Windows":
  envCython["LINKFLAGS"] = ["-shared", "-static"]
  envCython["LIBS"] += [File(f"{sys.base_prefix}/libs/python{sys.version_info.major}{sys.version_info.minor}.lib")]
else:
  envCython["LINKFLAGS"] = ["-pthread", "-shared"]

np_version = SCons.Script.Value(np.__version__)
Export('envCython', 'np_version')

Export('env', 'arch', 'acados', 'ffmpeg_libs')

# Setup cache dir
cache_dir = '/data/scons_cache' if arch == "comma_arm64" else os.path.join(tempfile.gettempdir(), 'scons_cache')
cache_size_limit = 4e9 if "CI" in os.environ else 2e9
CacheDir(cache_dir)
Clean(["."], cache_dir)

def prune_cache_dir(target=None, source=None, env=None):
  cache_files = sorted((os.path.join(root, f) for root, _, files in os.walk(cache_dir) for f in files), key=os.path.getmtime)
  cache_size = sum(os.path.getsize(f) for f in cache_files)
  for f in cache_files:
    if cache_size < cache_size_limit:
      break
    cache_size -= os.path.getsize(f)
    os.unlink(f)

# ********** start building stuff **********

# Build common module
SConscript(['openpilot/common/SConscript'])
Import('_common')
common = [_common, 'json11', 'zmq']
Export('common')

# Build messaging (cereal + msgq + socketmaster + their dependencies)
# Enable swaglog include in submodules
env_swaglog = env.Clone()
env_swaglog['CXXFLAGS'].append('-DSWAGLOG="\\"common/swaglog.h\\""')
SConscript(['msgq_repo/SConscript'], exports={'env': env_swaglog})

SConscript(['openpilot/cereal/SConscript'])

Import('socketmaster', 'msgq')
messaging = [socketmaster, msgq, 'capnp', 'kj',]
Export('messaging')


# Build other submodules
SConscript(['panda/SConscript'])

# Build rednose library
SConscript(['rednose_repo/rednose/SConscript'])

# Build system services
SConscript([
  'openpilot/system/loggerd/SConscript',
])

if arch == "comma_arm64":
  SConscript(['openpilot/system/camerad/SConscript'])

# Build selfdrive
SConscript([
  'openpilot/selfdrive/pandad/SConscript',
  'openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/SConscript',
  'openpilot/selfdrive/locationd/SConscript',
  'openpilot/selfdrive/ui/SConscript',
])
if arch != "Windows":  # modeld needs tinygrad's compiled model, Linux/macOS only
  SConscript(['openpilot/selfdrive/modeld/SConscript'])

# Build desktop-only tools
if GetOption('extras') and arch != "comma_arm64":
  SConscript([
    'openpilot/tools/replay/SConscript',
    'openpilot/tools/cabana/SConscript',
    'openpilot/tools/jotpluggler/SConscript',
  ])


env.CompilationDatabase('compile_commands.json')

# progress output
def count_scons_nodes(nodes):
  seen = set()
  stack = list(nodes)

  while stack:
    node = stack.pop().disambiguate()
    if node in seen:
      continue
    seen.add(node)
    if hasattr(node, 'has_builder') and node.has_builder():
      build_product_nodes.add(node)
    executor = node.get_executor()
    if executor is not None:
      stack += executor.get_all_prerequisites() + executor.get_all_children()

  return len(seen)

progress_interval = 5
progress_count = 0
build_product_nodes = set()
progress_total = max(1, count_scons_nodes(env.arg2nodes(BUILD_TARGETS or [Dir('.')], env.fs.Entry)))

def progress_function(node):
  global progress_count
  if progress_count >= progress_total:
    return
  progress_count = min(progress_count + progress_interval, progress_total)
  progress = round(100. * progress_count / progress_total, 1)
  sys.stderr.write("\rBuilding: %5.1f%%" % progress if sys.stderr.isatty() else "progress: %.1f\n" % progress)
  if progress == 100. and sys.stderr.isatty():
    sys.stderr.write("\n")
  sys.stderr.flush()

Progress(progress_function, interval=progress_interval)
AddPostAction(BUILD_TARGETS or [Dir('.')], prune_cache_dir)

def check_build_product_size(target, source, env):
  limit = 50 * 1024 * 1024  # GitHub max size
  for t in target:
    if hasattr(t, 'isfile') and t.isfile() and (size := os.path.getsize(t.abspath)) > limit:
      raise SCons.Errors.UserError(f"{t} is {size / (1024 * 1024):.1f} MiB, exceeding the {limit / (1024 * 1024):.1f} MiB limit")
if not GetOption('extras'):
  AddPostAction(list(build_product_nodes), Action(check_build_product_size, None))
