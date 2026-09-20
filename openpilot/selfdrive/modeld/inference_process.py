from contextlib import contextmanager
import ctypes
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import signal
import struct
import sys
import threading
import time
import traceback


@contextmanager
def _defer_sigint():
  if threading.current_thread() is not threading.main_thread():
    yield
    return
  previous_handler = signal.getsignal(signal.SIGINT)
  interrupted = None

  def defer_interrupt(signum, frame):
    nonlocal interrupted
    interrupted = (signum, frame)

  # Another device thread can receive SIGINT, so a main-thread signal mask
  # alone cannot protect Python ownership bookkeeping from its handler.
  signal.signal(signal.SIGINT, defer_interrupt)
  try:
    yield
  finally:
    signal.signal(signal.SIGINT, previous_handler)
    if interrupted is not None:
      if callable(previous_handler):
        previous_handler(*interrupted)
      elif previous_handler == signal.SIG_DFL:
        os.kill(os.getpid(), signal.SIGINT)


def _write(buffer, value):
  data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
  if len(data) > len(buffer) - 8:
    raise ValueError("inference message exceeds shared buffer")
  view = memoryview(buffer).cast('B')
  view[8:8 + len(data)] = data
  struct.pack_into('Q', view, 0, len(data))


def _read(buffer):
  size, = struct.unpack_from('Q', buffer)
  if size > len(buffer) - 8:
    raise ValueError("invalid inference message length")
  return pickle.loads(memoryview(buffer).cast('B')[8:8 + size])


def _worker(connection, request, response, factory, args, parent_pid):
  # A manager SIGKILL must not leave a worker holding the inference device.
  if sys.platform == 'linux':
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
      raise OSError(ctypes.get_errno(), "setting inference worker parent-death signal")
    if os.getppid() != parent_pid:
      return
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  try:
    model = factory(*args)
    connection.send_bytes(b'R')
    while connection.recv_bytes(1) == b'R':
      _write(response, model(*_read(request)))
      connection.send_bytes(b'R')
  except EOFError:
    pass
  except Exception:
    try:
      _write(response, traceback.format_exc())
      connection.send_bytes(b'E')
    except (BrokenPipeError, EOFError, OSError):
      pass
  finally:
    connection.close()
  # Return normally so device atexit handlers can release their resources.


class InferenceProcess:
  """One in-flight request, with a deadline independent of inference and IPC payload writes."""
  def __init__(self, factory, args=(), *, startup_timeout=60.0, cleanup=None):
    ctx = mp.get_context('spawn')
    self.request = ctx.RawArray('B', 64 * 1024)
    self.response = ctx.RawArray('B', 1024 * 1024)
    self.connection, child = ctx.Pipe()
    self.stopping = False
    self.reaper = None
    self.process = ctx.Process(target=_worker, args=(child, self.request, self.response, factory, args, os.getpid()),
                               name='modeld-inference', daemon=True)
    try:
      # Record ownership before delivering SIGINT, including the caller's
      # cleanup registration. Initialization remains interruptible below.
      with _defer_sigint():
        self.process.start()
        if cleanup is not None:
          cleanup.callback(self.close)
      child.close()
      self._receive(startup_timeout)
    except BaseException:
      child.close()
      if self.process.pid is not None:
        self.close()
      else:
        self.connection.close()
      raise

  def _receive(self, timeout):
    deadline = time.monotonic() + max(0., timeout)
    ready = wait([self.connection, self.process.sentinel], timeout=max(0., deadline - time.monotonic()))
    if not ready or time.monotonic() > deadline:
      raise TimeoutError("inference worker deadline exceeded")
    if self.connection not in ready:
      raise RuntimeError("inference worker exited")
    try:
      status = self.connection.recv_bytes(1)
    except (EOFError, OSError) as e:
      raise RuntimeError("inference worker disconnected") from e
    if status == b'E':
      raise RuntimeError(_read(self.response))
    if status != b'R':
      raise RuntimeError("invalid inference completion")

  def call(self, *args, timeout=0.15):
    if self.stopping:
      raise RuntimeError("inference worker is stopped")
    deadline = time.monotonic() + timeout
    try:
      _write(self.request, args)
      # Only a one-byte notification enters the pipe. A worker stopped while
      # writing a large result cannot strand the supervisor in recv().
      try:
        self.connection.send_bytes(b'R')
      except (EOFError, OSError) as e:
        raise RuntimeError("inference worker disconnected") from e
      self._receive(deadline - time.monotonic())
      result = _read(self.response)
      if time.monotonic() > deadline:
        raise TimeoutError("inference result exceeded deadline")
      return result
    except BaseException:
      self.stop()
      raise

  def _reap(self):
    # Leave room for device finalizers and interpreter teardown within manager's 5s stop budget.
    self.process.join(2.)
    if self.process.is_alive():
      self.process.terminate()
      self.process.join(.5)
    if self.process.is_alive():
      self.process.kill()
      self.process.join(.5)
    self.connection.close()

  def stop(self):
    if not self.stopping:
      with _defer_sigint():
        self.stopping = True
        try:
          self.connection.send_bytes(b'S')
        except (BrokenPipeError, EOFError, OSError):
          pass
        self.reaper = threading.Thread(target=self._reap, name='inference-reaper', daemon=True)
        self.reaper.start()

  def close(self):
    self.stop()
    self.reaper.join()
    if self.process.is_alive():
      raise RuntimeError("inference worker could not be stopped")

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()
