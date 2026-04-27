#!/usr/bin/env python3
"""Streaming audio server — accepts raw float32 mono 48kHz samples over TCP
and plays them through the full DSP pipeline to the speaker."""
import math
import queue
import select
import socket
import struct
import sys
import termios
import threading
import time
import tty
import numpy as np
import sounddevice as sd
from multiprocessing import Process, Pipe

# reuse everything from play_wav
from play_wav import (
  _eq_worker, _bk_worker, _out_worker, _parse_eq_file, read_key,
  SAMPLE_BUFFER, LOOKAHEAD,
)

SAMPLE_RATE = 48000  # clients must send at this rate
HOST = '0.0.0.0'
PORT = 7777


class StreamingPipeline:
  """Pipeline that reads from a thread-safe input queue instead of a samples array."""
  def __init__(self, sample_rate, eq_path, preamp_db=0.0):
    self.sample_rate = sample_rate
    self.in_q = queue.Queue(maxsize=LOOKAHEAD * 2)  # incoming raw samples
    self.out_buf = queue.Queue(maxsize=LOOKAHEAD)   # processed blocks
    self.flags = {'eq': True, 'bass': True, 'transient': True,
                  'crystal': True, 'loudness': True, 'mbcomp': True}
    self.raw = False
    self.vol = 0.5
    self._stop = threading.Event()
    self._send_lock = threading.Lock()  # serialize writes to _to_eq pipe

    eq_filters, file_preamp_db = _parse_eq_file(eq_path) if eq_path else ([], 0.0)
    total_preamp = file_preamp_db + preamp_db
    if eq_filters:
      print(f"Loaded {len(eq_filters)} EQ bands (preamp: {total_preamp:+.1f} dB)")

    self._to_eq, eq_in = Pipe()
    eq_out, bk_in = Pipe()
    bk_out, out_in = Pipe()
    out_out, self._from_out = Pipe()

    self._eq_proc = Process(target=_eq_worker, args=(eq_in, eq_out, sample_rate, eq_filters, total_preamp), daemon=True)
    self._bk_proc = Process(target=_bk_worker, args=(bk_in, bk_out, sample_rate), daemon=True)
    self._out_proc = Process(target=_out_worker, args=(out_in, out_out, sample_rate), daemon=True)
    self._eq_proc.start(); self._bk_proc.start(); self._out_proc.start()
    eq_in.close(); eq_out.close(); bk_in.close(); bk_out.close(); out_in.close(); out_out.close()

    self._feeder = threading.Thread(target=self._feed, daemon=True)
    self._feeder.start()

  def push_samples(self, samples_1d):
    """Enqueue samples from the network for processing."""
    self.in_q.put(samples_1d)

  def _feed(self):
    """Pulls from in_q, sends through pipeline, pushes results to out_buf."""
    pending = 0
    while not self._stop.is_set():
      if pending > 0 and self._from_out.poll():
        result = self._from_out.recv()
        # ignore non-audio messages (reconfig_eq echo)
        if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str) and result[0] == 'reconfig_eq':
          pending -= 1
          continue
        self.out_buf.put(result)
        pending -= 1

      if pending >= 2 or self.out_buf.full():
        time.sleep(0.0005)
        continue

      try:
        block = self.in_q.get(timeout=0.05)
      except queue.Empty:
        continue

      with self._send_lock:
        self._to_eq.send((np.ascontiguousarray(block, dtype=np.float32),
                          self.flags, self.raw, self.vol))
      pending += 1

  def reconfigure_eq(self, filters, preamp_db):
    """Rebuild the EQ FIR with new filter specs. Message flows through pipeline."""
    with self._send_lock:
      self._to_eq.send(('reconfig_eq', list(filters), float(preamp_db)))

  def get_block(self):
    try:
      return self.out_buf.get_nowait()
    except queue.Empty:
      return None

  def stop(self):
    self._stop.set()
    self._feeder.join(timeout=1)
    with self._send_lock:
      self._to_eq.send(None)
    try: self._from_out.recv()
    except: pass
    self._eq_proc.join(timeout=1)
    self._bk_proc.join(timeout=1)
    self._out_proc.join(timeout=1)


def receive_samples(conn, pipe, stats):
  """Receive float32 samples from socket in SAMPLE_BUFFER chunks and feed pipeline."""
  block_bytes = SAMPLE_BUFFER * 4  # float32 = 4 bytes per sample
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
        # sanitize: any NaN/Inf corrupts biquad state forever
        bad = ~np.isfinite(samples)
        if bad.any():
          stats['nan_count'] = stats.get('nan_count', 0) + int(bad.sum())
          samples[bad] = 0.0
        stats['rx_rms'] = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        pipe.push_samples(samples)
    # leftover samples at end of stream
    if buf:
      samples = np.frombuffer(buf[:len(buf) // 4 * 4], dtype=np.float32)
      if len(samples):
        pipe.push_samples(samples)
  except (ConnectionResetError, BrokenPipeError):
    pass


def _tui(stdscr, pipe, filters, preamp_db, eq_path, cb_stats, client_status):
  """Curses TUI for live EQ editing + pipeline control."""
  import curses
  curses.curs_set(0)
  stdscr.nodelay(True)
  curses.use_default_colors()

  # filters is a mutable list of [ftype, fc, gain_db, q] lists (mutable!)
  sel_filter = 0
  sel_param = 2  # 0=type, 1=fc, 2=gain, 3=q
  ftypes = ['PK', 'LS', 'HS']

  # adjustment step sizes per parameter (small, medium, large)
  STEPS = {
    1: [1.0, 10.0, 100.0],   # fc: 1, 10, 100 Hz
    2: [0.1, 0.5, 1.0],       # gain: 0.1, 0.5, 1.0 dB
    3: [0.05, 0.1, 0.5],      # Q: 0.05, 0.1, 0.5
  }

  def apply():
    tuples = [(f[0], f[1], f[2], f[3]) for f in filters]
    pipe.reconfigure_eq(tuples, preamp_db[0])

  apply()  # initial load

  param_name = ['type', 'Fc', 'gain', 'Q']
  last_refresh = 0.0

  while True:
    key = stdscr.getch()

    if key == ord('q'):
      break
    elif key == curses.KEY_UP:
      sel_filter = (sel_filter - 1) % max(1, len(filters))
    elif key == curses.KEY_DOWN:
      sel_filter = (sel_filter + 1) % max(1, len(filters))
    elif key == curses.KEY_LEFT:
      sel_param = max(0, sel_param - 1)
    elif key == curses.KEY_RIGHT:
      sel_param = min(3, sel_param + 1)
    elif key in (ord('+'), ord('='), ord('-'), ord('_')):
      sign = 1 if key in (ord('+'), ord('=')) else -1
      if sel_filter < len(filters):
        f = filters[sel_filter]
        if sel_param == 0:
          idx = ftypes.index(f[0]) if f[0] in ftypes else 0
          f[0] = ftypes[(idx + sign) % len(ftypes)]
        else:
          step = STEPS[sel_param][1]  # medium step
          f[sel_param] += sign * step
          if sel_param == 1: f[sel_param] = max(20.0, min(20000.0, f[sel_param]))
          elif sel_param == 3: f[sel_param] = max(0.1, min(20.0, f[sel_param]))
        apply()
    elif key in (ord('['), ord(']')):
      sign = -1 if key == ord('[') else 1
      if sel_filter < len(filters) and sel_param > 0:
        step = STEPS[sel_param][2]  # large step
        f = filters[sel_filter]
        f[sel_param] += sign * step
        if sel_param == 1: f[sel_param] = max(20.0, min(20000.0, f[sel_param]))
        elif sel_param == 3: f[sel_param] = max(0.1, min(20.0, f[sel_param]))
        apply()
    elif key in (ord(','), ord('.')):
      sign = -1 if key == ord(',') else 1
      if sel_filter < len(filters) and sel_param > 0:
        step = STEPS[sel_param][0]  # fine step
        f = filters[sel_filter]
        f[sel_param] += sign * step
        if sel_param == 1: f[sel_param] = max(20.0, min(20000.0, f[sel_param]))
        elif sel_param == 3: f[sel_param] = max(0.1, min(20.0, f[sel_param]))
        apply()
    elif key == ord('P'):
      preamp_db[0] -= 0.5; apply()
    elif key == ord('p'):
      preamp_db[0] += 0.5; apply()
    elif key == ord('S') and eq_path:
      # save to file
      with open(eq_path, 'w') as fp:
        fp.write(f"Filter Settings file\n\nPreamp: {preamp_db[0]:.1f} dB\n\n")
        for i, f in enumerate(filters, 1):
          if f[0] == 'PK':
            fp.write(f"Filter {i:>2}: ON  PK       Fc  {f[1]:>6.0f} Hz  Gain  {f[2]:>6.2f} dB  Q  {f[3]:.3f}\n")
          elif f[0] == 'LS':
            fp.write(f"Filter {i:>2}: ON  LS       Fc  {f[1]:>6.0f} Hz  Gain  {f[2]:>6.2f} dB  Q  {f[3]:.3f}\n")
          elif f[0] == 'HS':
            fp.write(f"Filter {i:>2}: ON  HS       Fc  {f[1]:>6.0f} Hz  Gain  {f[2]:>6.2f} dB  Q  {f[3]:.3f}\n")
    # transport / pipeline toggles
    elif key == ord('e'):
      all_on = all(pipe.flags.values())
      for k in pipe.flags: pipe.flags[k] = not all_on
    elif key == ord('r'):
      pipe.raw = not pipe.raw
    elif key == ord('w'):
      pipe.vol = min(pipe.vol + 0.1, 2.0)
    elif key == ord('s'):
      pipe.vol = max(pipe.vol - 0.1, 0.0)
    elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
      k = ['eq', 'bass', 'transient', 'crystal', 'loudness', 'mbcomp'][key - ord('1')]
      pipe.flags[k] = not pipe.flags[k]

    # redraw at ~20Hz
    now = time.monotonic()
    if now - last_refresh < 0.05 and key == -1:
      time.sleep(0.01)
      continue
    last_refresh = now

    stdscr.erase()
    stdscr.addstr(0, 0, f"EQ Live Editor  —  preamp: {preamp_db[0]:+.1f} dB", curses.A_BOLD)
    stdscr.addstr(1, 0, "↑↓ filter  ←→ param  , . +/-  [ ] large   p/P preamp  S save  q quit")

    for i, f in enumerate(filters):
      y = 3 + i
      line = f"  F{i+1:>2}:  "
      stdscr.addstr(y, 0, line)
      params = [f[0], f"{f[1]:>5.0f} Hz", f"{f[2]:+6.2f} dB", f"Q {f[3]:>5.3f}"]
      x = len(line)
      for j, p in enumerate(params):
        cell = f" {p} "
        attr = 0
        if i == sel_filter:
          attr = curses.A_BOLD
          if j == sel_param:
            attr |= curses.A_REVERSE
        stdscr.addstr(y, x, cell, attr)
        x += len(cell)

    # pipeline status
    y = 3 + len(filters) + 1
    short = {'eq': 'EQ', 'bass': 'BA', 'transient': 'TR', 'crystal': 'CR',
             'loudness': 'LD', 'mbcomp': 'MB'}
    fxlabel = ' '.join(s if pipe.flags[k] else '··' for k, s in short.items())
    if pipe.raw: fxlabel += ' RAW'
    stdscr.addstr(y, 0, f"client: {client_status[0]}")
    stdscr.addstr(y + 1, 0, f"vol: {pipe.vol:.1f}   {fxlabel}")
    stdscr.addstr(y + 2, 0, f"rx: {cb_stats['rx_bytes']//1024} KB  rx_rms: {cb_stats['rx_rms']:.3f}")
    stdscr.addstr(y + 3, 0, f"q: {pipe.in_q.qsize()}/{pipe.out_buf.qsize()}  "
                           f"cb fill/silent: {cb_stats['filled']}/{cb_stats['silent']}")

    stdscr.refresh()


def main(eq_path=None, preamp_db=0.0, volume=0.5):
  sd._terminate()
  sd._initialize()

  pipe = StreamingPipeline(SAMPLE_RATE, eq_path, preamp_db)
  pipe.vol = volume

  cb_stats = {'calls': 0, 'silent': 0, 'filled': 0, 'out_rms': 0.0,
              'rx_bytes': 0, 'rx_rms': 0.0}

  def callback(data_out, frames, time_info, status):
    if status:
      print(f"\n  stream status: {status}", file=sys.stderr)
    cb_stats['calls'] += 1
    result = pipe.get_block()
    if result is None:
      cb_stats['silent'] += 1
      data_out[:] = 0
      return
    cb_stats['filled'] += 1
    n = min(len(result), frames)
    data_out[:n, 0] = result[:n].astype(np.float32)
    data_out[n:] = 0
    if cb_stats['calls'] % 10 == 0 and n > 0:
      cb_stats['out_rms'] = float(np.sqrt(np.mean(result[:n] ** 2)))

  srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  srv.bind((HOST, PORT))
  srv.listen(1)
  srv.settimeout(0.1)  # non-blocking-ish accept so we can poll keys
  print(f"Listening on {HOST}:{PORT} (float32 mono @ {SAMPLE_RATE} Hz)")
  print("  e: toggle all DSP  r: raw  w/s: volume  q: quit")
  print("  1:EQ 2:bass 3:transient 4:crystal 5:loudness 6:mbcomp")

  paused = [False]
  client_status = ['waiting']

  def server_thread():
    while not stop_event.is_set():
      try:
        conn, addr = srv.accept()
      except (socket.timeout, OSError):
        continue
      client_status[0] = f'{addr[0]}:{addr[1]}'
      try:
        receive_samples(conn, pipe, cb_stats)
      finally:
        conn.close()
        client_status[0] = 'waiting'

  stop_event = threading.Event()
  sthread = threading.Thread(target=server_thread, daemon=True)
  sthread.start()

  # load filters for TUI editing
  filters_raw, file_pdb = _parse_eq_file(eq_path) if eq_path else ([], 0.0)
  # convert tuples → mutable lists so TUI can edit in place
  filters = [list(f) for f in filters_raw]
  preamp_box = [file_pdb + (preamp_db or 0.0)]

  import curses
  try:
    with sd.OutputStream(channels=1, samplerate=SAMPLE_RATE,
                          callback=callback, blocksize=SAMPLE_BUFFER):
      curses.wrapper(_tui, pipe, filters, preamp_box, eq_path, cb_stats, client_status)
  finally:
    stop_event.set()
    srv.close()
    pipe.stop()


if __name__ == "__main__":
  args = []
  eq_path = None
  preamp_db = None
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == '--eq' and i + 1 < len(sys.argv):
      i += 1
      eq_path = sys.argv[i]
    elif sys.argv[i] == '--preamp' and i + 1 < len(sys.argv):
      i += 1
      preamp_db = float(sys.argv[i])
    elif not sys.argv[i].startswith('--'):
      args.append(sys.argv[i])
    i += 1

  if preamp_db is None and eq_path:
    filters, _ = _parse_eq_file(eq_path)
    if filters:
      gains = sorted(g for _, _, g, _ in filters)
      preamp_db = -gains[len(gains) // 2] / 2.0
    else:
      preamp_db = 0.0
  elif preamp_db is None:
    preamp_db = 0.0

  vol = float(args[0]) if args else 0.5
  main(eq_path=eq_path, preamp_db=preamp_db, volume=vol)
