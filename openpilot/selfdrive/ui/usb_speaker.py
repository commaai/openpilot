"""Hot-pluggable USB speech output, independent of PortAudio's cached devices."""
import ctypes
import ctypes.util
import errno
from pathlib import Path
import threading
import time

import numpy as np

from openpilot.common.swaglog import cloudlog


def find_usb_output(root=Path('/proc/asound')):
  for path in sorted(root.glob('card[0-9]*/stream[0-9]*')):
    try:
      description = path.read_text()
    except OSError:
      continue  # Device disappeared during discovery.
    if 'USB Audio' in description and 'Playback:' in description:
      return f"plughw:{int(path.parent.name[4:])},{int(path.name[6:])}"
  return None


class USBSpeaker:
  def __init__(self, device, callback):
    self.callback = callback
    self.stop = threading.Event()
    self.thread = None
    self.active = False
    self.pcm = ctypes.c_void_p()
    self.lib = ctypes.CDLL(ctypes.util.find_library('asound') or 'libasound.so.2')
    ptr, integer, uint = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
    for name, args, result in (
      ('snd_pcm_open', [ctypes.POINTER(ptr), ctypes.c_char_p, integer, integer], integer),
      ('snd_pcm_set_params', [ptr, integer, integer, uint, uint, integer, uint], integer),
      ('snd_pcm_avail_update', [ptr], ctypes.c_long),
      ('snd_pcm_writei', [ptr, ptr, ctypes.c_ulong], ctypes.c_long),
      ('snd_pcm_recover', [ptr, integer, integer], integer),
      ('snd_pcm_close', [ptr], integer),
    ):
      fn = getattr(self.lib, name)
      fn.argtypes, fn.restype = args, result
    try:
      self.check(self.lib.snd_pcm_open(ctypes.byref(self.pcm), device.encode(), 0, 1))  # playback, nonblocking
      self.check(self.lib.snd_pcm_set_params(self.pcm, 2, 3, 2, 48000, 1, 60000))  # S16_LE, interleaved, stereo, 60 ms
    except Exception:
      self.close()
      raise

  @staticmethod
  def check(result):
    if result < 0:
      raise OSError(-result, 'USB audio I/O failed')
    return result

  def start(self):
    self.active = True
    self.thread = threading.Thread(target=self.run, name='usb-speech', daemon=True)
    self.thread.start()

  def run(self):
    output = np.empty((960, 2), dtype=np.float32)
    pending = np.empty((0, 2), dtype='<i2')
    pending_at = 0.0
    try:
      while not self.stop.wait(0.005):
        for _ in range(3):
          available = self.lib.snd_pcm_avail_update(self.pcm)
          if available == -errno.EAGAIN:
            break
          if available < 0:
            self.check(self.lib.snd_pcm_recover(self.pcm, available, 1))
            pending = pending[:0]
            break
          if available < 960:
            break
          if len(pending) and time.monotonic() - pending_at > 0.2:
            pending = pending[:0]
          if not len(pending):
            self.callback(output, 960, None, None)
            pending = (np.clip(output, -1, 1) * 32767).astype('<i2')
            pending_at = time.monotonic()
          written = self.lib.snd_pcm_writei(self.pcm, pending.ctypes.data, len(pending))
          if written == -errno.EAGAIN:
            break
          if written < 0:
            self.check(self.lib.snd_pcm_recover(self.pcm, written, 1))
            pending = pending[:0]
            break
          pending = pending[written:]
    except Exception:
      cloudlog.exception('USB speech output disconnected or failed')
    finally:
      self.active = False
      self.release()

  def release(self):
    if self.pcm:
      self.lib.snd_pcm_close(self.pcm)
      self.pcm = ctypes.c_void_p()

  def close(self):
    self.stop.set()
    if self.thread is not None:
      self.thread.join()
    else:
      self.release()
    self.active = False
