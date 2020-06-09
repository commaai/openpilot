import os
from cffi import FFI

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))


def write_code(folder, name, code, header):
  if not os.path.exists(folder):
    os.mkdir(folder)

  open(os.path.join(folder, f"{name}.cpp"), 'w').write(code)
  open(os.path.join(folder, f"{name}.h"), 'w').write(header)


def load_code(folder, name):
  shared_fn = os.path.join(folder, f"lib{name}.so")
  header_fn = os.path.join(folder, f"{name}.h")

  with open(header_fn) as f:
    header = f.read()

  ffi = FFI()
  ffi.cdef(header)
  return (ffi, ffi.dlopen(shared_fn))


class KalmanError(Exception):
  pass
