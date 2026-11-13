import sys
import sysconfig
from pathlib import Path

import nanobind

Import('env', 'arch', 'common')

envNanobind = env.Clone(LIBS=[])
nb_dir = Path(nanobind.source_dir()).parent
envNanobind.Append(CPPPATH=[sysconfig.get_paths()['include'], nanobind.include_dir(), str(nb_dir / "ext/robin_map/include")],
                   CCFLAGS=["-fvisibility=hidden", "-fno-strict-aliasing"])
if arch == "Darwin":
  envNanobind.Append(LINKFLAGS=["-bundle", "-undefined", "dynamic_lookup"])
else:
  # Python symbols resolve at import time.
  envNanobind["LINKFLAGS"] = [f for f in env["LINKFLAGS"] if f != "-Wl,--no-undefined"] + ["-pthread", "-shared"]

# Third-party runtime sources do not follow our shadowing warning policy.
envNanobindRuntime = envNanobind.Clone()
envNanobindRuntime['CCFLAGS'] = [flag for flag in envNanobindRuntime['CCFLAGS'] if not flag.startswith('-Wshadow')]
nanobind_lib = envNanobindRuntime.SharedObject('msgq/nb_combined.os', str(nb_dir / "src/nb_combined.cpp"))


visionipc_dir = Dir('msgq/visionipc')
python_suffix = sysconfig.get_config_var('EXT_SUFFIX')

# Build msgq
msgq_objects = env.SharedObject([
  'msgq/ipc.cc',
  'msgq/event.cc',
  'msgq/msgq.cc',
])
msgq = env.Library('msgq', msgq_objects)
msgq_python = envNanobind.Program(f'msgq/_ipc{python_suffix}', ['msgq/ipc_py.cc', nanobind_lib], LIBS=envNanobind["LIBS"]+[msgq]+common)

# Build Vision IPC
vipc_files = ['visionipc.cc', 'visionipc_server.cc', 'visionipc_client.cc']
if File('/dev/ion').exists():
  vipc_files += ['visionbuf_ion.cc']
else:
  vipc_files += ['visionbuf.cc']
vipc_sources = [f'{visionipc_dir.abspath}/{f}' for f in vipc_files]

vipc_objects = env.SharedObject(vipc_sources)
visionipc = env.Library('visionipc', vipc_objects)


vipc_libs = envNanobind["LIBS"] + [visionipc, msgq] + common
visionipc_python = envNanobind.Program(f'{visionipc_dir.abspath}/_visionipc{python_suffix}', [f'{visionipc_dir.abspath}/visionipc_py.cc', nanobind_lib],
                  LIBS=vipc_libs)

# ASan modules require a preloaded runtime before Python can import them.
if "-fsanitize=address" not in env["LINKFLAGS"]:
  root = Dir('.').abspath
  for module in ('msgq._ipc', 'msgq.visionipc._visionipc'):
    env.Command(module.replace('.', '/') + '.pyi', [msgq_python, visionipc_python],
                f'"{sys.executable}" -m nanobind.stubgen -i "{root}" -m {module} -o "$TARGET"')

if GetOption('extras'):
  env.Program('msgq/test_runner', ['msgq/msgq_tests.cc'], LIBS=[msgq]+common)

Export('visionipc', 'msgq', 'msgq_python')
