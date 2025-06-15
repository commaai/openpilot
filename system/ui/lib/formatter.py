import cffi
import os


def build_formatter():
  try:
    from openpilot.system.ui.lib import _format_string

    return _format_string.ffi, _format_string.lib
  except ImportError:
    pass

  current_dir = os.path.dirname(os.path.abspath(__file__))
  ffibuilder = cffi.FFI()
  ffibuilder.cdef("""
        int vasprintf(char **strp, const char *fmt, ...);
        void free(void *ptr);
    """)

  ffibuilder.set_source(
    "_format_string",
    """
        #include <stdio.h>
        #include <stdlib.h>
    """,
  )

  ffibuilder.compile(tmpdir=current_dir, verbose=False)
  from openpilot.system.ui.lib import _format_string

  return _format_string.ffi, _format_string.lib
