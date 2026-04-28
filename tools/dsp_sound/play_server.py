#!/usr/bin/env python3
"""DSP-free streaming audio server.

Runs on the comma device. Listens TCP for raw float32 mono 48 kHz samples and
plays them through the same sounddevice OutputStream that selfdrive/ui/soundd.py
uses, with no processing of any kind.

Test hook: if the env var COMMA_SPEAKER_TEST=<path> is set, audio is written to
that WAV file (int16 PCM) instead of the speaker. Used by comma_speaker.py
--test for end-to-end roundtrip verification.
"""
import os
import socket
import sys
import threading
import time
import wave
import numpy as np

SAMPLE_RATE = 48000   # match selfdrive/ui/soundd.py
SAMPLE_BUFFER = 4096  # match selfdrive/ui/soundd.py
HOST = '0.0.0.0'
PORT = 7777


class RingBuffer:
  """Single-writer / single-reader float32 ring buffer."""
  def __init__(self, seconds=2.0, sample_rate=SAMPLE_RATE):
    self.cap = int(seconds * sample_rate)
    self.buf = np.zeros(self.cap, dtype=np.float32)
    self.w = 0
    self.r = 0
    self.fill = 0
    self.lock = threading.Lock()

  def write(self, samples):
    n = len(samples)
    if n == 0:
      return
    with self.lock:
      # if it doesn't fit, drop oldest by advancing r
      if n >= self.cap:
        samples = samples[-self.cap:]
        n = self.cap
      free = self.cap - self.fill
      if n > free:
        drop = n - free
        self.r = (self.r + drop) % self.cap
        self.fill -= drop
      end = self.w + n
      if end <= self.cap:
        self.buf[self.w:end] = samples
      else:
        first = self.cap - self.w
        self.buf[self.w:] = samples[:first]
        self.buf[:n - first] = samples[first:]
      self.w = end % self.cap
      self.fill += n

  def read_into(self, out, frames):
    """Fill `out[:frames]` from buffer; zero-fill on underrun."""
    with self.lock:
      n = min(self.fill, frames)
      if n > 0:
        end = self.r + n
        if end <= self.cap:
          out[:n] = self.buf[self.r:end]
        else:
          first = self.cap - self.r
          out[:first] = self.buf[self.r:]
          out[first:n] = self.buf[:n - first]
        self.r = end % self.cap
        self.fill -= n
      if n < frames:
        out[n:frames] = 0.0


def make_callback(rb, stats):
  def cb(out, frames, time_info, status):
    if status:
      stats['xrun'] += 1
    rb.read_into(out[:, 0], frames)  # PURE PASSTHROUGH — no scale, no clip, no DSP
  return cb


def receive_one_client(conn, rb, stats):
  block_bytes = SAMPLE_BUFFER * 4
  buf = b''
  try:
    while True:
      data = conn.recv(block_bytes)
      if not data:
        break
      stats['rx_bytes'] += len(data)
      buf += data
      while len(buf) >= block_bytes:
        chunk = buf[:block_bytes]
        buf = buf[block_bytes:]
        samples = np.frombuffer(chunk, dtype=np.float32).copy()
        # defensive: NaN/Inf would taint the output stream — not DSP, just safety
        bad = ~np.isfinite(samples)
        if bad.any():
          samples[bad] = 0.0
          stats['nan'] += int(bad.sum())
        stats['rx_rms'] = float(np.sqrt(np.mean(samples * samples)))
        rb.write(samples)
    # any trailing partial block: align to float32 boundary and write
    aligned = len(buf) // 4 * 4
    if aligned:
      tail = np.frombuffer(buf[:aligned], dtype=np.float32).copy()
      bad = ~np.isfinite(tail)
      if bad.any():
        tail[bad] = 0.0
      rb.write(tail)
  except (ConnectionResetError, BrokenPipeError, OSError):
    pass


def serve(rb, stats):
  srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  srv.bind((HOST, PORT))
  srv.listen(1)
  srv.settimeout(0.5)
  print(f"listening on {HOST}:{PORT} (float32 mono @ {SAMPLE_RATE} Hz, no DSP)", flush=True)

  last_status = time.monotonic()
  while True:
    try:
      conn, addr = srv.accept()
    except socket.timeout:
      now = time.monotonic()
      if now - last_status > 5.0:
        print(f"  waiting for client (rx={stats['rx_bytes']//1024} KB, xrun={stats['xrun']})", flush=True)
        last_status = now
      continue
    except OSError:
      break
    stats['clients'] += 1
    print(f"client connected: {addr[0]}:{addr[1]}", flush=True)
    try:
      receive_one_client(conn, rb, stats)
    finally:
      conn.close()
      print("client disconnected", flush=True)


def run_test_mode(test_path, rb, stats):
  """Write to WAV instead of opening sounddevice. Stops on EOF (client disconnect)."""
  import struct
  print(f"TEST MODE: writing to {test_path}", flush=True)
  wf = wave.open(test_path, 'wb')
  wf.setnchannels(1)
  wf.setsampwidth(2)
  wf.setframerate(SAMPLE_RATE)

  srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  srv.bind((HOST, PORT))
  srv.listen(1)
  print(f"listening on {HOST}:{PORT} (TEST MODE)", flush=True)

  conn, addr = srv.accept()
  print(f"client connected: {addr[0]}:{addr[1]}", flush=True)
  block_bytes = SAMPLE_BUFFER * 4
  buf = b''
  try:
    while True:
      data = conn.recv(block_bytes)
      if not data:
        break
      buf += data
      while len(buf) >= block_bytes:
        chunk = buf[:block_bytes]
        buf = buf[block_bytes:]
        samples = np.frombuffer(chunk, dtype=np.float32)
        # int16 PCM, no scaling beyond the standard ±1.0 → ±32767 mapping
        ints = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
        wf.writeframes(ints.tobytes())
    aligned = len(buf) // 4 * 4
    if aligned:
      tail = np.frombuffer(buf[:aligned], dtype=np.float32)
      ints = np.clip(tail * 32767.0, -32768, 32767).astype(np.int16)
      wf.writeframes(ints.tobytes())
  finally:
    conn.close()
    srv.close()
    wf.close()
  print(f"TEST MODE: wrote {test_path}", flush=True)


def main():
  test_path = os.environ.get('COMMA_SPEAKER_TEST')
  rb = RingBuffer(seconds=2.0)
  stats = {'xrun': 0, 'rx_bytes': 0, 'rx_rms': 0.0, 'nan': 0, 'clients': 0}

  if test_path:
    run_test_mode(test_path, rb, stats)
    return

  import sounddevice as sd
  # match selfdrive/ui/soundd.py exactly
  with sd.OutputStream(channels=1, samplerate=SAMPLE_RATE,
                       callback=make_callback(rb, stats),
                       blocksize=SAMPLE_BUFFER, dtype='float32') as stream:
    print(f"stream: {stream.samplerate=} {stream.channels=} {stream.dtype=} "
          f"{stream.device=} {stream.blocksize=}", flush=True)
    serve(rb, stats)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nstopped", flush=True)
    sys.exit(0)
