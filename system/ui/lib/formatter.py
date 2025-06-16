import cffi
import tempfile
from pathlib import Path


def _get_formatter():
  try:
    from openpilot.system.ui.lib import _format_string

    return _format_string.ffi, _format_string.lib
  except ImportError:
    ffibuilder = cffi.FFI()
    ffibuilder.cdef("""
        void free(void *ptr);
        int format_with_va_list(char **strp, const char *fmt, void *ap);
    """)
    ffibuilder.set_source(
      "_format_string",
      """
        #include <stdio.h>
        #include <stdlib.h>
        #include <stdarg.h>

        int format_with_va_list(char **strp, const char *fmt, void *ap) {
          return vasprintf(strp, fmt, *(va_list*)ap);
        }
      """,
    )
    target = Path(__file__).parent / "_format_string.so"
    tmp_dir = tempfile.gettempdir()
    ffibuilder.compile(tmpdir=tmp_dir, target=str(target), verbose=False)
    from openpilot.system.ui.lib import _format_string

    return _format_string.ffi, _format_string.lib

ffi, lib = _get_formatter()
