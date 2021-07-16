#!/usr/bin/env python3

import os
from cffi import FFI

import sip  # pylint: disable=import-error
from PyQt5.QtWidgets import QApplication, QLabel  # pylint: disable=no-name-in-module

from common.ffi_wrapper import suffix
from common.basedir import BASEDIR


def get_ffi():
  lib = os.path.join(BASEDIR, "selfdrive", "ui", "qt", "libpython_helpers" + suffix())

  ffi = FFI()
  ffi.cdef("void set_main_window(void *w);")
  return ffi, ffi.dlopen(lib)


if __name__ == "__main__":
  ffi, lib = get_ffi()

  app = QApplication([])
  label = QLabel('Hello World!')

  # Set full screen and rotate
  lib.set_main_window(ffi.cast('void*', sip.unwrapinstance(label)))

  app.exec_()
