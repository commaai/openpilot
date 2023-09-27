import os
import platform
from cffi import FFI

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))


def write_code(folder, name, code, header):
  if not os.path.exists(folder):
    os.mkdir(folder)

  open(os.path.join(folder, f"{name}.cpp"), 'w').write(code)
  open(os.path.join(folder, f"{name}.h"), 'w').write(header)


def load_code(folder, name, lib_name=None):
  if lib_name is None:
    lib_name = name
  shared_ext = "dylib" if platform.system() == "Darwin" else "so"
  shared_fn = os.path.join(folder, f"lib{lib_name}.{shared_ext}")
  header_fn = os.path.join(folder, f"{name}.h")

  with open(header_fn) as f:
    header = f.read()

  # is the only thing that can be parsed by cffi
  header = "\n".join([line for line in header.split("\n") if line.startswith("void ")])

  ffi = FFI()
  ffi.cdef(header)
  return (ffi, ffi.dlopen(shared_fn))


class KalmanError(Exception):
  pass
