import os
import sys

from typing import no_type_check

class FdRedirect:
  def __init__(self, file_prefix: str, fd: int):
    fname = os.path.join("/tmp", f"{file_prefix}.{fd}")
    if os.path.exists(fname):
      os.unlink(fname)
    self.dest_fd = os.open(fname, os.O_WRONLY | os.O_CREAT)
    self.dest_fname = fname
    self.source_fd = fd
    os.set_inheritable(self.dest_fd, True)

  def __del__(self):
    os.close(self.dest_fd)

  def purge(self) -> None:
    os.unlink(self.dest_fname)

  def read(self) -> bytes:
    with open(self.dest_fname, "rb") as f:
      return f.read() or b""

  def link(self) -> None:
    os.dup2(self.dest_fd, self.source_fd)


class ProcessOutputCapture:
  def __init__(self, proc_name: str, prefix: str):
    prefix = f"{proc_name}_{prefix}"
    self.stdout_redirect = FdRedirect(prefix, 1)
    self.stderr_redirect = FdRedirect(prefix, 2)

  def __del__(self):
    self.stdout_redirect.purge()
    self.stderr_redirect.purge()

  @no_type_check # ipython classes have incompatible signatures
  def link_with_current_proc(self) -> None:
    try:
      # prevent ipykernel from redirecting stdout/stderr of python subprocesses
      from ipykernel.iostream import OutStream
      if isinstance(sys.stdout, OutStream):
        sys.stdout = sys.__stdout__
      if isinstance(sys.stderr, OutStream):
        sys.stderr = sys.__stderr__
    except ImportError:
      pass

    # link stdout/stderr to the fifo
    self.stdout_redirect.link()
    self.stderr_redirect.link()

  def read_outerr(self) -> tuple[str, str]:
    out_str = self.stdout_redirect.read().decode()
    err_str = self.stderr_redirect.read().decode()
    return out_str, err_str
