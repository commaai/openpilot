
import os
from cffi import FFI

import sip  # pylint: disable=import-error

from common.ffi_wrapper import suffix
from common.basedir import BASEDIR


def get_ffi():
  lib = os.path.join(BASEDIR, "selfdrive", "ui", "qt", "libpython_helpers" + suffix())

  ffi = FFI()
  ffi.cdef("void set_main_window(void *w);")
  return ffi, ffi.dlopen(lib)


def set_main_window(widget):
  ffi, lib = get_ffi()
  lib.set_main_window(ffi.cast('void*', sip.unwrapinstance(widget)))
