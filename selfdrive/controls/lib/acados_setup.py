import os
import sys
import ctypes
import platform

def get_acados_dir():
  # current file: openpilot/selfdrive/controls/lib/acados_setup.py
  # acados is at openpilot/third_party/acados
  current_dir = os.path.dirname(os.path.abspath(__file__))
  op_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
  return os.path.join(op_dir, 'third_party', 'acados')

def get_acados_lib_path():
  acados_dir = get_acados_dir()
  if sys.platform.startswith('linux'):
    machine = platform.machine()
    if machine == 'x86_64':
      return os.path.join(acados_dir, 'x86_64', 'lib')
    elif machine == 'aarch64':
      return os.path.join(acados_dir, 'larch64', 'lib')
  elif sys.platform.startswith('darwin'):
    return os.path.join(acados_dir, 'Darwin', 'lib')

  # Fallback
  return os.path.join(acados_dir, 'x86_64', 'lib')

def acados_preload():
  lib_path = get_acados_lib_path()
  if not os.path.exists(lib_path):
    return

  if sys.platform.startswith('linux'):
    libs = ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so.3.1']
    mode = ctypes.RTLD_GLOBAL
  elif sys.platform.startswith('darwin'):
    libs = ['libblasfeo.dylib', 'libhpipm.dylib', 'libqpOASES_e.3.1.dylib']
    mode = ctypes.RTLD_GLOBAL
  else:
    libs = []
    mode = 0

  for lib in libs:
    full_path = os.path.join(lib_path, lib)
    if os.path.exists(full_path):
      try:
        ctypes.CDLL(full_path, mode=mode)
      except OSError:
        pass

def prepare_acados_ocp_json(json_file):
  import json
  import tempfile

  with open(json_file) as f:
    data = json.load(f)

  data['acados_lib_path'] = get_acados_lib_path()

  fd, path = tempfile.mkstemp(suffix='.json', text=True)
  with os.fdopen(fd, 'w') as f:
    json.dump(data, f, indent=4)

  return path
