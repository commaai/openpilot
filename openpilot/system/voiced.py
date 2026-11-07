#!/usr/bin/env python3
import os
# openblas threads busy-wait on every core, 1 thread is 8x faster on the SDM845
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import socket
import ssl
import struct
import threading
import time
from collections import deque

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.api import Api
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.voice.kws import Kws, SR, WIN

CONTEXT_SERVICES = ['gpsLocationExternal', 'gpsLocation', 'carState', 'selfdriveState', 'carParams', 'deviceState', 'navInstruction']

State = log.VoiceAssistant.State

MODEL = os.path.join(BASEDIR, 'openpilot/system/voice/kws.npz')
THRESHOLD = 0.9
HOP = 1280  # 80 ms
PREROLL_S = 3.0  # the score peaks after the wake word ends and the server verifies it, so keep enough before the trigger
MAX_QUERY_S = 8.0
END_SILENCE_S = 0.8
NO_SPEECH_S = 3.0
REFRACTORY_S = 2.0
WAKE_TAIL_S = 0.4  # the end of the wake word is not the start of the question
SELF_MUTE_S = 0.7  # the speaker keeps playing after we stop sending
OUT_CHUNK_S = 0.25
# no ping answer for this long and the link is dead, cell links often drop without a reset
LINK_TIMEOUT_S = 5.0
LEVEL_HOP_S = 0.05


def tone(notes: list[tuple[float, float]], db: float, h3: bool = False) -> np.ndarray:
  # sine plus a soft 2nd harmonic, 5 ms attack and exponential decay, notes are (Hz, ms) with 0 Hz as a gap
  out = []
  for f, ms in notes:
    t = np.arange(int(SR * ms / 1000)) / SR
    w = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + (0.18 * np.sin(6 * np.pi * f * t) if h3 else 0)
    out.append(w * np.minimum(t / 0.005, 1) * np.exp(-t / 0.05) if f else np.zeros_like(t))
  x = np.concatenate(out)
  return (x / np.abs(x).max() * 10 ** (db / 20)).astype(np.float32)


# rising fifth when listening starts, falling fourth when it ends, low double for errors
WAKE_CHIME = tone([(659, 70), (0, 15), (988, 110)], -14)
DONE_CHIME = tone([(988, 60), (0, 10), (740, 120)], -17)
ERROR_CHIME = tone([(392, 100), (0, 60), (392, 100)], -14, h3=True)


def level(x: np.ndarray) -> float:
  # -50 to -10 dBFS onto 0 to 1
  return float(np.clip((20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) + 50) / 40, 0, 1))


def frame(t: int, p: bytes) -> bytes:
  return struct.pack('<BI', t, len(p)) + p


def recv_exact(s: socket.socket, n: int) -> bytes:
  b = b''
  while len(b) < n:
    try:
      c = s.recv(n - len(b))
    except TimeoutError:
      # a quiet link between frames is fine, the caller decides, but never drop half a frame
      if not b:
        raise
      continue
    if not c:
      raise EOFError
    b += c
  return b


def to_pcm(x: np.ndarray) -> bytes:
  return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class Voiced:
  def __init__(self, server: str):
    host, _, port = server.partition(':')
    self.server = (host, int(port or 8781))
    self.kws = Kws(MODEL)
    self.pm = messaging.PubMaster(['voiceAssistant', 'voiceAudioData'])
    self.sm = messaging.SubMaster(['gpsLocationExternal', 'gpsLocation', 'carState', 'selfdriveState', 'carParams', 'deviceState', 'navInstruction'],
                                  ignore_alive=CONTEXT_SERVICES, ignore_valid=CONTEXT_SERVICES)
    self.ring: deque[np.ndarray] = deque(maxlen=int(SR * PREROLL_S) // HOP)
    self.pending = np.zeros(0, np.float32)
    self.buf = np.zeros(WIN, np.float32)
    self.noise_db = -60.0
    self.last_trigger = -1e9
    self.mute_until = 0.0
    self.followup = False
    self.query: dict | None = None
    self.state = State.idle
    self.score = 0.
    self.transcript = ''
    self.reply = ''
    self.mic_level, self.out_t0 = 0., 0.
    self.reply_latency_ms = 0.
    self.out_lv: list[float] = []

  def context(self) -> dict:
    # what the assistant may want to know about the drive, sent with every query
    sm, ctx = self.sm, {'unix_time': time.time(), 'tz': time.strftime('%z')}
    gps = next((sm[s] for s in ('gpsLocationExternal', 'gpsLocation') if sm.recv_frame[s] > 0 and sm[s].hasFix), None)
    if gps is not None:
      ctx['gps'] = {'lat': round(gps.latitude, 5), 'lon': round(gps.longitude, 5), 'speed_mps': round(gps.speed, 1),
                    'bearing_deg': round(gps.bearingDeg), 'accuracy_m': round(gps.horizontalAccuracy, 1)}
    if sm.recv_frame['carState'] > 0:
      cs = sm['carState']
      ctx['car'] = {'speed_mps': round(cs.vEgo, 1), 'gear': str(cs.gearShifter), 'cruise_set_mps': round(cs.cruiseState.speed, 1),
                    'left_blinker': cs.leftBlinker, 'right_blinker': cs.rightBlinker}
    if sm.recv_frame['carParams'] > 0:
      ctx.setdefault('car', {})['model'] = sm['carParams'].carFingerprint
    if sm.recv_frame['selfdriveState'] > 0:
      ss = sm['selfdriveState']
      ctx['openpilot'] = {'engaged': ss.enabled, 'personality': str(ss.personality)}
    if sm.recv_frame['navInstruction'] > 0:
      ni = sm['navInstruction']
      ctx['nav'] = {'next': f'{ni.maneuverPrimaryText} {ni.maneuverSecondaryText}'.strip(), 'next_in_m': round(ni.maneuverDistance),
                    'remaining_m': round(ni.distanceRemaining), 'remaining_s': round(ni.timeRemaining)}
    if sm.recv_frame['deviceState'] > 0:
      ds = sm['deviceState']
      ctx['device'] = {'max_temp_c': round(ds.maxTempC), 'network': str(ds.networkType)}
    return ctx

  def publish(self) -> None:
    msg = messaging.new_message('voiceAssistant', valid=True)
    va = msg.voiceAssistant
    va.state = self.state
    va.wakeScore = float(self.score)
    va.transcript = self.transcript
    va.reply = self.reply
    va.serverRttMs = float(getattr(self, 'rtt_ms', 0.))
    va.replyLatencyMs = self.reply_latency_ms
    va.level = self.out_level() if self.state == State.speaking else self.mic_level
    self.pm.send('voiceAssistant', msg)

  def play(self, x: np.ndarray, rate: int) -> float:
    # soundd mixes this into its stream, returns how long it will play
    now, hop = time.monotonic(), int(rate * LEVEL_HOP_S)
    if now >= self.out_t0 + len(self.out_lv) * LEVEL_HOP_S:
      self.out_t0, self.out_lv = now, []
    self.out_lv += [level(x[i:i + hop]) for i in range(0, len(x), hop)]
    n = int(rate * OUT_CHUNK_S)
    for i in range(0, len(x), n):
      msg = messaging.new_message('voiceAudioData', valid=True)
      msg.voiceAudioData.data = to_pcm(x[i:i + n])
      msg.voiceAudioData.sampleRate = rate
      self.pm.send('voiceAudioData', msg)
    return len(x) / rate

  def out_level(self) -> float:
    # what the speaker is playing right now, so the indicator moves with the voice
    i = int((time.monotonic() - self.out_t0) / LEVEL_HOP_S)
    return self.out_lv[i] if 0 <= i < len(self.out_lv) else 0.

  def step(self, x: np.ndarray) -> None:
    now = time.monotonic()
    self.ring.append(x)
    if self.followup and now >= self.mute_until:
      self.followup = False
      self.start_query(now, wake=False)
    if self.query is not None:
      self.query_step(x, now)
      return

    lvl = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    self.noise_db = 0.98 * self.noise_db + 0.02 * lvl if lvl < self.noise_db + 6 else self.noise_db + 0.002
    self.buf = np.concatenate([self.buf[HOP:], x])
    if now < self.mute_until:
      # our own chime and reply would wake us up
      self.buf[:] = 0
      return
    self.score = self.kws(self.buf)
    if self.score > THRESHOLD and now - self.last_trigger > REFRACTORY_S:
      self.last_trigger = now
      try:
        self.start_query(now, wake=True)
      except OSError:
        cloudlog.exception("voiced: server unreachable")
        self.query, self.state = None, State.idle

  def start_query(self, now: float, wake: bool) -> None:
    # the visual cue fires here, the chime only after the server verified the wake word
    self.state, self.transcript, self.reply = State.listening, '', ''
    self.publish()
    pre = np.concatenate(list(self.ring)) if wake else np.zeros(0, np.float32)
    s = socket.create_connection(self.server, timeout=5)
    s.sendall(frame(1, json.dumps({'device': os.uname().nodename, 'score': float(self.score), 'verify': wake, 'context': self.context()}).encode()))
    s.sendall(frame(2, to_pcm(pre)))
    self.query = {'sock': s, 't_wake': now, 'speech': False, 'last_voice': now, 'wake': wake}
    cloudlog.info(f"voiced: wake score {self.score:.3f} followup {not wake}")

  def query_step(self, x: np.ndarray, now: float) -> None:
    q = self.query
    lvl = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    if lvl > self.noise_db + 8 and now - q['t_wake'] > WAKE_TAIL_S:
      q['speech'], q['last_voice'] = True, now
    try:
      q['sock'].sendall(frame(2, to_pcm(x)))
    except OSError:
      cloudlog.exception("voiced: upload failed")
      self.query, self.state = None, State.idle
      return
    done = now - q['t_wake'] > MAX_QUERY_S or (q['speech'] and now - q['last_voice'] > END_SILENCE_S) or \
           (not q['speech'] and now - q['t_wake'] > NO_SPEECH_S)
    if done:
      q['sock'].sendall(frame(3, b''))
      # the query audio is still in the window, don't wake on it again
      self.mute_until = float('inf')
      self.state = State.thinking
      threading.Thread(target=self.reply_thread, args=(q['sock'], q['wake']), daemon=True).start()
      self.query = None

  def reply_thread(self, s: socket.socket, wake: bool) -> None:
    play_end = time.monotonic()
    try:
      s.settimeout(30)
      rate = 22050
      while True:
        t, n = struct.unpack('<BI', recv_exact(s, 5))
        p = recv_exact(s, n)
        if t == 10:
          d = json.loads(p)
          self.transcript = d.get('transcript', '')
          if wake and d.get('verified'):
            play_end = max(play_end, time.monotonic()) + self.play(WAKE_CHIME, SR)
        elif t == 12:
          d = json.loads(p)
          rate = d.get('sample_rate', rate)
          self.reply = d.get('reply', self.reply)
        elif t == 11:
          self.state = State.speaking
          x = np.frombuffer(p, np.int16).astype(np.float32) / 32768
          play_end = max(play_end, time.monotonic()) + self.play(x, rate)
        elif t == 13:
          self.followup = bool(json.loads(p or b'{}').get('followup'))
          break
    except Exception:
      cloudlog.exception("voiced: reply failed")
    finally:
      s.close()
      self.mute_until = play_end + SELF_MUTE_S
      self.state = State.idle

  def run(self) -> None:
    audio = messaging.sub_sock('rawAudioData', conflate=False)
    last_pub = 0.
    while True:
      self.sm.update(0)
      for m in messaging.drain_sock(audio, wait_for_one=True):
        self.pending = np.concatenate([self.pending, np.frombuffer(m.rawAudioData.data, np.int16).astype(np.float32) / 32768])
      while len(self.pending) >= HOP:
        x, self.pending = self.pending[:HOP], self.pending[HOP:]
        self.step(x)
      if time.monotonic() - last_pub > 0.08:
        self.publish()
        last_pub = time.monotonic()


class Streamer(Voiced):
  # v1: stream the mic nonstop, the workstation detects the wake word, verifies it and answers
  def __init__(self, server: str):
    host, _, port = server.partition(':')
    self.server = (host, int(port or 8443))
    self.pm = messaging.PubMaster(['voiceAssistant', 'voiceAudioData'])
    self.sm = messaging.SubMaster(CONTEXT_SERVICES, ignore_alive=CONTEXT_SERVICES, ignore_valid=CONTEXT_SERVICES)
    self.state, self.score, self.transcript, self.reply = State.idle, 0., '', ''
    self.mute_until = 0.0
    self.sock: socket.socket | None = None
    self.rtt_ms = 0.
    self.ping_sent: float | None = None
    self.reply_latency_ms = 0.
    self.speak_until = 0.
    self.mic_level, self.out_t0 = 0., 0.
    self.out_lv: list[float] = []
    self.dongle_id = Params().get("DongleId")
    self.tls = ssl.create_default_context()

  def meta(self) -> bytes:
    try:
      ctx = self.context()
    except Exception:
      # missing car data must never stop the assistant
      cloudlog.exception("voiced: context failed")
      ctx = {}
    # the round trip goes along so the server can log the link
    return frame(1, json.dumps({'stream': True, 'device': os.uname().nodename, 'context': ctx, 'rtt_ms': round(self.rtt_ms)}).encode())

  def connect(self) -> None:
    raw = socket.create_connection(self.server, timeout=5)
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # the server is public behind tailscale funnel, tls there and the device's own comma jwt as auth
    s = self.tls.wrap_socket(raw, server_hostname=self.server[0])
    # sends and receives give up on a dead link instead of blocking forever
    s.settimeout(LINK_TIMEOUT_S)
    self.ping_sent = None
    auth = {'stream': True, 'dongle_id': self.dongle_id, 'token': Api(self.dongle_id).get_token(expiry_hours=24)}
    s.sendall(frame(1, json.dumps(auth).encode()))
    s.sendall(self.meta())
    self.sock = s
    threading.Thread(target=self.recv_thread, args=(s,), daemon=True).start()
    cloudlog.info(f"voiced: streaming to {self.server}")

  def recv_thread(self, s: socket.socket) -> None:
    rate, play_end, chimed = 22050, 0.0, False
    timing: dict = {}
    server_lat: dict = {}
    try:
      while True:
        try:
          hdr = recv_exact(s, 5)
        except TimeoutError:
          # nothing comes while the device speaks, the ping watchdog in run() catches a dead link otherwise,
          # but a reply the server owes us that never comes means the link is gone
          if self.sock is not s or self.state == State.thinking:
            raise
          continue
        t, n = struct.unpack('<BI', hdr)
        p = recv_exact(s, n)
        now = time.monotonic()
        if t == 10:
          d = json.loads(p)
          if d.get('wake'):
            # heard the wake word, chime while the driver keeps talking
            self.state, self.transcript, self.reply = State.listening, '', ''
            play_end, chimed = now + self.play(WAKE_CHIME, SR), True
          elif d.get('verified'):
            # stop sending until the reply is done so we never hear ourselves
            self.mute_until = float('inf')
            self.state, self.transcript, self.reply = State.thinking, d.get('transcript', ''), ''
            timing = {'id': d.get('id'), 't0': now}
            if not chimed:
              play_end = now + self.play(WAKE_CHIME, SR)
            elif d.get('query'):
              # the mic is closed, a bare wake word gets "Yes?" instead
              play_end = max(play_end, now) + self.play(DONE_CHIME, SR)
            chimed = False
          elif self.state == State.listening:
            # the full query did not hold up, drop the cue
            self.state, chimed = State.idle, False
        elif t == 12:
          d = json.loads(p)
          rate = d.get('sample_rate', rate)
          server_lat = d.get('latency', server_lat)
          if 'say' in d:
            self.reply = f"{self.reply} {d['say'].strip()}".strip()
        elif t == 11:
          self.state = State.speaking
          if timing and 'first_audio' not in timing:
            # the chime may still be playing, the reply starts after it
            timing['first_audio'], timing['play_start'] = now - timing['t0'], max(play_end, now) - timing['t0']
          play_end = max(play_end, now) + self.play(np.frombuffer(p, np.int16).astype(np.float32) / 32768, rate)
        elif t == 13:
          # audio arrives faster than it plays, stay speaking until the speaker is done
          self.speak_until = max(play_end, now)
          if timing:
            # the server logs it next to its own timing, sent while the mic upload is still paused so no other thread writes
            t0 = timing.pop('t0')
            timing = {k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()}
            timing.update(play_end=round(self.speak_until - t0, 3), rtt_ms=round(self.rtt_ms))
            if 'play_start' in timing:
              # same sum the server logs: wait after the last word, until the transcript went out, one way, until the speaker
              self.reply_latency_ms = (server_lat.get('gap', 0.) + server_lat.get('verified', 0.) + self.rtt_ms / 2000 + timing['play_start']) * 1000
            s.sendall(frame(5, json.dumps(timing).encode()))
            cloudlog.info(f"voiced: playback {timing}")
            timing = {}
          self.mute_until = self.speak_until + SELF_MUTE_S
        elif t == 14:
          self.rtt_ms, self.ping_sent = (now - struct.unpack('<d', p)[0]) * 1000, None
    except Exception:
      cloudlog.exception("voiced: stream closed")
      if self.state in (State.listening, State.thinking):
        self.play(ERROR_CHIME, SR)
    finally:
      s.close()
      if self.sock is s:
        self.sock = None
      self.mute_until, self.state, self.rtt_ms = 0.0, State.idle, 0.

  def run(self) -> None:
    audio = messaging.sub_sock('rawAudioData', conflate=False)
    last_pub = last_meta = last_try = last_ping = 0.
    while True:
      self.sm.update(0)
      now = time.monotonic()
      if self.sock is not None and self.ping_sent is not None and now >= self.mute_until:
        # an unanswered ping shows as a growing round trip, too long and we reconnect
        self.rtt_ms = max(self.rtt_ms, (now - self.ping_sent) * 1000)
        if now - self.ping_sent > LINK_TIMEOUT_S:
          cloudlog.warning(f"voiced: no ping answer for {now - self.ping_sent:.1f} s, reconnecting")
          dead, self.sock = self.sock, None
          try:
            # wakes the receive thread too
            dead.shutdown(socket.SHUT_RDWR)
          except OSError:
            pass
          dead.close()
      if self.sock is None and now - last_try > 2.0:
        last_try = now
        try:
          self.connect()
        except Exception:
          cloudlog.warning(f"voiced: server {self.server} unreachable")
      for m in messaging.drain_sock(audio, wait_for_one=True):
        s = self.sock
        if s is None or time.monotonic() < self.mute_until:
          continue
        if self.state == State.listening:
          self.mic_level = level(np.frombuffer(m.rawAudioData.data, np.int16).astype(np.float32) / 32768)
        try:
          if now - last_meta > 1.0:
            # keep gps and car state fresh for the next query
            s.sendall(self.meta())
            last_meta = now
          if now - last_ping > 1.0:
            s.sendall(frame(4, struct.pack('<d', time.monotonic())))
            last_ping = now
            if self.ping_sent is None:
              self.ping_sent = now
          s.sendall(frame(2, m.rawAudioData.data))
        except OSError:
          cloudlog.exception("voiced: send failed")
          s.close()
          self.sock = None
      if self.state == State.speaking and self.speak_until and now > self.speak_until:
        self.state, self.speak_until = State.idle, 0.
      if now - last_pub > 0.05:
        self.publish()
        last_pub = now


def main():
  server = Params().get("VoiceAssistantServer")
  if isinstance(server, bytes):
    server = server.decode()
  if not server:
    cloudlog.warning("voiced: disabled, no VoiceAssistantServer")
    while True:
      time.sleep(60)
  Streamer(server).run()


if __name__ == "__main__":
  main()
