"""Unfiltered, non-grabbing Linux touchscreen capture for the typing study."""
import fcntl
import os
from pathlib import Path
import select
import struct
import threading
import time

from openpilot.common.realtime import drop_realtime, set_core_affinity

INPUT_EVENT = struct.Struct('@llHHi')
EVIOCSCLOCKID = 0x400445A0  # _IOW('E', 0xa0, int), linux/input.h
EVIOCGABS_BASE = 0x80184540  # _IOR('E', 0x40 + axis, struct input_absinfo)


class TouchCapture:
  def __init__(self, write):
    self.write = write
    self._stop = threading.Event()
    self._thread = None
    self._fd = None
    self.error = None

  def start(self):
    candidates = [entry for entry in Path('/sys/class/input').glob('event*')
                  if (entry / 'device/name').read_text().strip() == 'fts_ts']
    if len(candidates) != 1:
      raise OSError('Expected one fts_ts touchscreen')
    path = Path('/dev/input') / candidates[0].name
    self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    clock = 'monotonic'
    try:
      fcntl.ioctl(self._fd, EVIOCSCLOCKID, struct.pack('i', time.CLOCK_MONOTONIC))
    except OSError:
      clock = 'realtime (driver did not accept monotonic clock)'
    axes = {}
    capabilities = int((candidates[0] / 'device/capabilities/abs').read_text().strip().replace(' ', ''), 16)
    for axis in range(64):
      if not capabilities & (1 << axis):
        continue
      data = bytearray(24)
      try:
        fcntl.ioctl(self._fd, EVIOCGABS_BASE + axis, data)
        axes[str(axis)] = list(struct.unpack('iiiiii', data))
      except OSError:
        continue
    self.write({'type': 'evdev_start', 'device': str(path), 'clock': clock, 'axes': axes,
                'monotonic': time.monotonic(), 'realtime': time.clock_gettime(time.CLOCK_REALTIME),
                'format': '[seconds,microseconds,event_type,event_code,value]', 'exclusive_grab': False})
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name='keyboard-touch-log')
    self._thread.start()

  def _run(self):
    drop_realtime()
    set_core_affinity([0, 1, 2, 3])
    pending = b''
    try:
      while True:
        ready, _, _ = select.select([self._fd], [], [], 0 if self._stop.is_set() else 0.1)
        if not ready:
          if self._stop.is_set():
            break
          continue
        block = os.read(self._fd, INPUT_EVENT.size * 256)
        if not block:
          raise OSError('Touchscreen event stream closed')
        pending += block
        complete = len(pending) // INPUT_EVENT.size * INPUT_EVENT.size
        events = list(INPUT_EVENT.iter_unpack(pending[:complete]))
        pending = pending[complete:]
        if events:
          self.write({'type': 'evdev', 'read_monotonic': time.monotonic(), 'events': events})
      if pending:
        raise OSError('Incomplete Linux input event at end of capture')
    except OSError as error:
      self.error = str(error)
      self.write({'type': 'capture_error', 'message': self.error})

  def stop(self):
    self._stop.set()
    if self._thread is not None:
      self._thread.join()
      self._thread = None
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None
      self.write({'type': 'evdev_end', 'monotonic': time.monotonic()})


def encode_mouse_event(event):
  return [event.t, event.slot, event.pos.x, event.pos.y, event.left_pressed, event.left_released, event.left_down]
