import json
import socket
import struct
import threading
import time

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.system import voiced

State = log.VoiceAssistant.State


def stub_server(sock: socket.socket, reply_pcm: bytes, got: dict) -> None:
  conn, _ = sock.accept()
  audio = b''
  while True:
    t, n = struct.unpack('<BI', voiced.recv_exact(conn, 5))
    p = voiced.recv_exact(conn, n)
    if t == 1:
      got['meta'] = json.loads(p)
    elif t == 2:
      audio += p
    elif t == 3:
      break
  got['audio_s'] = len(audio) / 2 / voiced.SR
  conn.sendall(voiced.frame(10, json.dumps({'transcript': 'clanker, hi', 'verified': True, 'query': 'hi'}).encode()))
  conn.sendall(voiced.frame(12, json.dumps({'sample_rate': 22050}).encode()))
  conn.sendall(voiced.frame(11, reply_pcm))
  conn.sendall(voiced.frame(12, json.dumps({'reply': 'hello'}).encode()))
  conn.sendall(voiced.frame(13, b'{}'))
  conn.close()


class FakeTime:
  def __init__(self):
    self.t = 1000.

  def monotonic(self) -> float:
    return self.t

  def sleep(self, s: float) -> None:
    time.sleep(s)

  def time(self) -> float:
    return time.time()

  def strftime(self, fmt: str) -> str:
    return time.strftime(fmt)


class OneShotKws:
  def __init__(self, fire_at: int):
    self.n, self.fire_at = 0, fire_at

  def __call__(self, audio: np.ndarray) -> float:
    self.n += 1
    return 0.99 if self.n == self.fire_at else 0.0


def test_wake_query_reply(monkeypatch):
  srv = socket.socket()
  srv.bind(('127.0.0.1', 0))
  srv.listen(1)
  reply = (np.sin(np.arange(22050) / 10) * 8000).astype(np.int16).tobytes()  # 1 s
  got: dict = {}
  threading.Thread(target=stub_server, args=(srv, reply, got), daemon=True).start()

  monkeypatch.setattr(voiced, 'Kws', lambda path: OneShotKws(fire_at=40))
  clock = FakeTime()
  monkeypatch.setattr(voiced, 'time', clock)
  v = voiced.Voiced(f'127.0.0.1:{srv.getsockname()[1]}')
  out = messaging.sub_sock('voiceAudioData', conflate=False, timeout=100)
  time.sleep(0.2)

  rng = np.random.default_rng(0)
  states = []
  t_end = None
  for i in range(120):
    # quiet noise, then "speech" a bit after the trigger, then silence ends the query
    x = rng.standard_normal(voiced.HOP).astype(np.float32) * 1e-3
    if 47 <= i < 60:
      x *= 100
    had_query = v.query is not None
    v.step(x)
    states.append(v.state)
    # the reply thread can move past thinking before we look, so take the end from the query itself
    if t_end is None and had_query and v.query is None:
      t_end = clock.t
    clock.t += voiced.HOP / voiced.SR

  for _ in range(100):
    if v.state == State.idle and v.mute_until != float('inf'):
      break
    time.sleep(0.05)

  assert State.listening in states
  assert got['meta']['verify'] is True
  # 3 s preroll plus the query
  assert got['audio_s'] > voiced.PREROLL_S
  assert v.reply == 'hello' and v.transcript == 'clanker, hi'
  # chime first, then the 1 s reply, all through soundd
  played = sum(len(m.voiceAudioData.data) / 2 / m.voiceAudioData.sampleRate for m in messaging.drain_sock(out))
  assert abs(played - (len(voiced.WAKE_CHIME) / voiced.SR + 1.0)) < 0.02, played
  # the reply must not wake us up again
  assert v.mute_until >= t_end + len(voiced.WAKE_CHIME) / voiced.SR + 1.0 + voiced.SELF_MUTE_S - 0.01
