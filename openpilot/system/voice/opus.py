import ctypes
import ctypes.util

SR = 16000
FRAME = 320  # 20 ms, opus only takes 2.5 to 60 ms frames
BITRATE = 32000
OPUS_APPLICATION_VOIP = 2048
OPUS_SET_BITRATE_REQUEST = 4002


def _lib() -> ctypes.CDLL:
  lib = ctypes.CDLL(ctypes.util.find_library('opus') or 'libopus.so.0')
  lib.opus_encoder_create.restype = ctypes.c_void_p
  lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int32]
  return lib


class Encoder:
  # 16 kHz mono int16 in, one opus packet per 20 ms out, about 10x less than raw on a cell uplink
  def __init__(self):
    self.lib = _lib()
    err = ctypes.c_int()
    self.enc = self.lib.opus_encoder_create(SR, 1, OPUS_APPLICATION_VOIP, ctypes.byref(err))
    if err.value != 0:
      raise RuntimeError(f'opus_encoder_create {err.value}')
    self.lib.opus_encoder_ctl(ctypes.c_void_p(self.enc), OPUS_SET_BITRATE_REQUEST, ctypes.c_int(BITRATE))
    self.pending = b''
    self.out = ctypes.create_string_buffer(4000)

  def encode(self, pcm: bytes) -> list[bytes]:
    self.pending += pcm
    packets = []
    while len(self.pending) >= FRAME * 2:
      frame, self.pending = self.pending[:FRAME * 2], self.pending[FRAME * 2:]
      n = self.lib.opus_encode(self.enc, frame, FRAME, self.out, len(self.out))
      if n < 0:
        raise RuntimeError(f'opus_encode {n}')
      packets.append(self.out.raw[:n])
    return packets

