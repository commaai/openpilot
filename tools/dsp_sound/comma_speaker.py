#!/usr/bin/env python3
"""Comma Speaker — turn a comma 4 (mici) into a foolproof network speaker.

Run with no args:
    ./comma_speaker.py

What it does:
  1. Discovers your comma device (cached IP → LAN scan → comma proxy lookup → prompt)
  2. Installs and starts the simplified play_server.py on the device via SSH proxy
  3. Creates a virtual PipeWire sink "Comma Speaker" on your laptop
  4. Sets it as the system default → all audio (browser, Spotify, etc.) plays on the comma
  5. On Ctrl+C: restores your previous default sink and stops the device server

Test without a device:
    ./comma_speaker.py --test

Other flags:
    --device "name"  pick a specific comma by alias / dongle id
    --ip 1.2.3.4     skip discovery, use this LAN IP
    --no-default     don't hijack the default sink (route apps manually)
    --ping           play a 440 Hz beep on the device, prompt y/n
    --stop           kill any server still running on the device, exit

Linux only (PipeWire). macOS/Windows would need BlackHole + ffmpeg or VB-Cable.
"""
import argparse
import atexit
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# add openpilot root to path so CommaApi import works
_HERE = Path(__file__).resolve().parent
_BASEDIR = _HERE.parent.parent
sys.path.insert(0, str(_BASEDIR))

PORT = 7777
SSH_PROXY = 'ssh.comma.ai'
SSH_KEY = _BASEDIR / 'system' / 'hardware' / 'tici' / 'id_rsa'
LOCAL_SERVER = _HERE / 'play_server.py'

CACHE_DIR = Path.home() / '.cache' / 'comma_speaker'
CACHE_FILE = CACHE_DIR / 'device.json'
LOCK_FILE = '/tmp/comma_speaker.pid'

REMOTE_DIR = '/data/dsp_sound'
REMOTE_SERVER = f'{REMOTE_DIR}/play_server.py'
REMOTE_LOG = '/tmp/play_server.log'

SINK_NAME = 'comma_speaker'
SINK_DESC = 'Comma_Speaker'  # underscores avoid issues with pactl prop parsing

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096

# ---------- pretty printing ----------

def _info(msg): print(f"  {msg}", flush=True)
def _ok(msg):   print(f"  ok  {msg}", flush=True)
def _warn(msg): print(f"  !!  {msg}", file=sys.stderr, flush=True)
def _err(msg):  print(f"  ERR {msg}", file=sys.stderr, flush=True)
def _step(msg): print(f"\n>> {msg}", flush=True)


# ---------- single-instance lock ----------

def acquire_lock():
  fp = open(LOCK_FILE, 'w')
  try:
    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError:
    _err(f"another comma_speaker is already running ({LOCK_FILE})")
    sys.exit(1)
  fp.write(str(os.getpid()))
  fp.flush()
  return fp


# ---------- cache ----------

def load_cached():
  try:
    return json.loads(CACHE_FILE.read_text())
  except (FileNotFoundError, json.JSONDecodeError):
    return {}

def save_cached(**kw):
  CACHE_DIR.mkdir(parents=True, exist_ok=True)
  cur = load_cached()
  cur.update(kw)
  CACHE_FILE.write_text(json.dumps(cur, indent=2))


# ---------- network probe / LAN scan ----------

def tcp_probe(ip, port=PORT, timeout=0.5):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.settimeout(timeout)
  try:
    s.connect((ip, port))
    return True
  except OSError:
    return False
  finally:
    s.close()


def _local_subnets():
  """Yield candidate /24 subnet prefixes for scanning."""
  subnets = set()
  try:
    out = subprocess.run(['ip', '-4', '-j', 'route'],
                         capture_output=True, text=True, timeout=5).stdout
    for r in json.loads(out):
      dst = r.get('dst', '')
      pref = r.get('prefsrc')
      if not pref or pref.startswith('127.'):
        continue
      m = re.match(r'(\d+\.\d+\.\d+)\.\d+/(\d+)', dst)
      if m and int(m.group(2)) >= 16:
        subnets.add(m.group(1))
      # also add a /24 around our prefsrc
      m2 = re.match(r'(\d+\.\d+\.\d+)\.\d+', pref)
      if m2:
        subnets.add(m2.group(1))
  except Exception as e:
    _warn(f"could not enumerate local subnets: {e}")
  return list(subnets)


def discover_lan(timeout_per_host=0.4):
  """Scan /24s around our IPs for hosts answering on port 7777."""
  subnets = _local_subnets()
  if not subnets:
    return []
  hits = []
  with ThreadPoolExecutor(max_workers=64) as pool:
    futures = {}
    for prefix in subnets:
      for last in range(1, 255):
        ip = f"{prefix}.{last}"
        futures[pool.submit(tcp_probe, ip, PORT, timeout_per_host)] = ip
    for fut, ip in futures.items():
      try:
        if fut.result():
          hits.append(ip)
      except Exception:
        pass
  return hits


# ---------- comma API + SSH proxy ----------

def _comma_api_devices():
  """Return list of {dongle_id, alias} dicts. Lazy import to avoid hard dep."""
  try:
    from openpilot.tools.lib.auth_config import get_token
    from openpilot.tools.lib.api import CommaApi
  except ImportError as e:
    _warn(f"cannot import openpilot tools.lib.api ({e}) — proxy device lookup unavailable")
    return None
  try:
    return CommaApi(get_token()).get('v1/me/devices')
  except Exception as e:
    _warn(f"comma API call failed: {e}")
    return None


def resolve_device(name_or_dongle):
  """Return (dongle_id, alias) for the given name/id, interactively if ambiguous."""
  if name_or_dongle and re.fullmatch(r'[0-9a-fA-F]{16}', name_or_dongle):
    return name_or_dongle, name_or_dongle
  devices = _comma_api_devices()
  if not devices:
    return None, None
  by_id = {d['dongle_id']: d.get('alias') or d['dongle_id'] for d in devices}
  if name_or_dongle:
    needle = name_or_dongle.replace(' ', '').lower()
    matches = {k: v for k, v in by_id.items()
               if needle in (v or '').replace(' ', '').lower()}
    if len(matches) == 1:
      did = next(iter(matches))
      return did, by_id[did]
    elif len(matches) == 0:
      _err(f"no device matched '{name_or_dongle}'")
      return None, None
    else:
      _err(f"multiple devices matched '{name_or_dongle}':")
      for k, v in matches.items():
        _err(f"  {v}  ({k})")
      return None, None
  if len(by_id) == 1:
    did = next(iter(by_id))
    return did, by_id[did]
  print("multiple devices in your comma account:")
  ids = list(by_id.items())
  for i, (k, v) in enumerate(ids, 1):
    print(f"  {i}. {v}  ({k})")
  while True:
    try:
      sel = input("pick one (number): ").strip()
      idx = int(sel) - 1
      if 0 <= idx < len(ids):
        return ids[idx]
    except (KeyboardInterrupt, EOFError):
      sys.exit(1)
    except ValueError:
      pass


def _proxy_cmd_args(dongle_id, *cmd):
  """Build an ssh ProxyCommand args list to run `cmd` on the device."""
  proxy = (f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no "
           f"-o UserKnownHostsFile=/dev/null -W %h:%p comma@{SSH_PROXY}")
  base = [
    'ssh',
    '-i', str(SSH_KEY),
    '-o', f"ProxyCommand={proxy}",
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'UserKnownHostsFile=/dev/null',
    '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=15',
    f"comma@comma-{dongle_id}",
  ]
  if cmd:
    base.append('--')
    base.extend(cmd)
  return base


def ssh_proxy(dongle_id, *cmd, timeout=30, capture=True):
  args = _proxy_cmd_args(dongle_id, *cmd)
  return subprocess.run(args, capture_output=capture, text=True, timeout=timeout)


def scp_proxy(dongle_id, local_path, remote_path, timeout=30):
  proxy = (f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no "
           f"-o UserKnownHostsFile=/dev/null -W %h:%p comma@{SSH_PROXY}")
  args = [
    'scp',
    '-i', str(SSH_KEY),
    '-o', f"ProxyCommand={proxy}",
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'UserKnownHostsFile=/dev/null',
    '-o', 'ConnectTimeout=15',
    str(local_path),
    f"comma@comma-{dongle_id}:{remote_path}",
  ]
  return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def get_lan_ip_via_proxy(dongle_id):
  """Ask the device via SSH proxy what its LAN IPs are; return first non-loopback."""
  r = ssh_proxy(dongle_id, "ip -4 -j addr show", timeout=20)
  if r.returncode != 0:
    _warn(f"ssh ip-addr-show failed: {r.stderr.strip()}")
    return None
  try:
    ifs = json.loads(r.stdout)
  except Exception:
    return None
  candidates = []
  for ifc in ifs:
    name = ifc.get('ifname', '')
    if name in ('lo',) or name.startswith('docker') or name.startswith('br-'):
      continue
    for a in ifc.get('addr_info', []):
      if a.get('family') == 'inet' and not a['local'].startswith('127.'):
        # prefer wlan over usb/eth (lower latency over wifi for typical setup)
        prio = 0 if name.startswith('wlan') else 1
        candidates.append((prio, name, a['local']))
  candidates.sort()
  return candidates[0][2] if candidates else None


# ---------- remote server lifecycle ----------

def install_server(dongle_id):
  _info(f"installing server to {REMOTE_SERVER}")
  r = ssh_proxy(dongle_id, f"mkdir -p {REMOTE_DIR}", timeout=15)
  if r.returncode != 0:
    raise RuntimeError(f"mkdir failed: {r.stderr.strip()}")
  r = scp_proxy(dongle_id, LOCAL_SERVER, REMOTE_SERVER)
  if r.returncode != 0:
    raise RuntimeError(f"scp failed: {r.stderr.strip()}")
  r = ssh_proxy(dongle_id, f"chmod +x {REMOTE_SERVER}", timeout=10)
  if r.returncode != 0:
    raise RuntimeError(f"chmod failed: {r.stderr.strip()}")
  _ok("server installed")


def start_remote_server(dongle_id):
  _info("starting server on device")
  cmd = (
    f"pkill -f {REMOTE_SERVER} >/dev/null 2>&1; "
    f"sleep 0.4; "
    f"nohup python3 {REMOTE_SERVER} >{REMOTE_LOG} 2>&1 </dev/null & "
    f"echo $!"
  )
  r = ssh_proxy(dongle_id, cmd, timeout=20)
  if r.returncode != 0:
    raise RuntimeError(f"start failed: {r.stderr.strip()}")
  pid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
  _ok(f"server pid {pid}")
  return pid


def stop_remote_server(dongle_id, pid=None):
  parts = []
  if pid and pid != "?":
    parts.append(f"kill {pid} 2>/dev/null")
  parts.append(f"pkill -f {REMOTE_SERVER} 2>/dev/null")
  parts.append("true")
  cmd = "; ".join(parts)
  try:
    ssh_proxy(dongle_id, cmd, timeout=15)
  except Exception as e:
    _warn(f"could not stop remote server cleanly: {e}")


def wait_for_port(ip, port, timeout=15):
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    if tcp_probe(ip, port, timeout=0.5):
      return True
    time.sleep(0.3)
  return False


# ---------- local PipeWire ----------

def get_default_sink():
  try:
    r = subprocess.run(['pw-metadata', '0', 'default.audio.sink'],
                       capture_output=True, text=True, timeout=3)
    m = re.search(r'"name"\s*:\s*"([^"]+)"', r.stdout)
    return m.group(1) if m else None
  except Exception:
    return None


def set_default_sink(name):
  subprocess.run(['pw-metadata', '0', 'default.audio.sink', f'{{"name":"{name}"}}'],
                 check=False, capture_output=True)


def clear_default_sink():
  subprocess.run(['pw-metadata', '-d', '0', 'default.audio.sink'],
                 check=False, capture_output=True)


def load_null_sink():
  """pactl load-module module-null-sink — creates SINK_NAME + SINK_NAME.monitor source.
  Returns the module id (used for unload on cleanup)."""
  r = subprocess.run([
    'pactl', 'load-module', 'module-null-sink',
    f'sink_name={SINK_NAME}',
    f'sink_properties=device.description={SINK_DESC}',
  ], capture_output=True, text=True, timeout=10)
  if r.returncode != 0:
    raise RuntimeError(f"pactl load-module failed: {r.stderr.strip()}")
  module_id = r.stdout.strip()
  # wait for sink to appear in pactl
  for _ in range(30):
    time.sleep(0.1)
    sinks = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                           capture_output=True, text=True, timeout=5).stdout
    if SINK_NAME in sinks:
      return module_id
  raise RuntimeError(f"null sink '{SINK_NAME}' never appeared")


def unload_null_sink(module_id):
  if not module_id:
    return
  subprocess.run(['pactl', 'unload-module', str(module_id)],
                 check=False, capture_output=True, timeout=5)


def start_capture():
  """parec captures from <SINK_NAME>.monitor. Stereo, raw float32 (no WAV header).
  Streamer downmixes to mono."""
  cmd = ['parec',
         f'--device={SINK_NAME}.monitor',
         '--format=float32le',
         f'--rate={SAMPLE_RATE}',
         '--channels=2',
         '--raw']
  return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ---------- streamer ----------

CAPTURE_CHANNELS = 2                                      # pw-record reads stereo
STEREO_BLOCK_BYTES = SAMPLE_BUFFER * 4 * CAPTURE_CHANNELS  # one block of stereo f32


def streamer(rec_proc_holder, ip, port, stop_evt, stats):
  """Read stereo f32 from pw-record, downmix to mono, send mono over TCP.
  Reconnects on failure."""
  backoff = 0.2
  while not stop_evt.is_set():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
      sock.connect((ip, port))
    except OSError as e:
      stats['reconn'] += 1
      _warn(f"connect to {ip}:{port} failed ({e}); retrying in {backoff:.1f}s")
      sock.close()
      stop_evt.wait(backoff)
      backoff = min(5.0, backoff * 2)
      continue
    backoff = 0.2
    stats['connected'] = True
    rec_proc = rec_proc_holder['proc']
    try:
      while not stop_evt.is_set():
        data = rec_proc.stdout.read(STEREO_BLOCK_BYTES)
        if not data:
          # capture ended (loopback died, etc.) — let caller restart it
          break
        # downmix stereo → mono: average L+R
        stereo = np.frombuffer(data, dtype=np.float32)
        if len(stereo) % CAPTURE_CHANNELS:
          stereo = stereo[:len(stereo) - (len(stereo) % CAPTURE_CHANNELS)]
        mono = stereo.reshape(-1, CAPTURE_CHANNELS).mean(axis=1).astype(np.float32)
        mono_bytes = mono.tobytes()
        try:
          sock.sendall(mono_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
          _warn(f"server dropped ({e}); reconnecting")
          break
        stats['tx_bytes'] += len(mono_bytes)
        if len(mono) > 0:
          stats['rms'] = float(np.sqrt(np.mean(mono * mono)))
          stats['peak'] = float(np.max(np.abs(mono)))
    finally:
      stats['connected'] = False
      try: sock.close()
      except Exception: pass


# ---------- pw-loopback / pw-record watchdog ----------

class CaptureManager:
  """Owns the null-sink module and the parec capture process. Restarts parec if it dies."""
  def __init__(self):
    self.lock = threading.Lock()
    self.module_id = None
    self.rec_proc = None
    self.rec_proc_holder = {'proc': None}
    self.stop_evt = threading.Event()
    self.previous_default = None
    self.make_default = True

  def start(self, make_default=True):
    self.make_default = make_default
    self.module_id = load_null_sink()
    if make_default:
      self.previous_default = get_default_sink()
      set_default_sink(SINK_NAME)
      _ok(f"set default sink → '{SINK_DESC}' (was: {self.previous_default or 'none'})")
    else:
      _info(f"sink '{SINK_DESC}' available — route apps via pavucontrol/Helvum")
    self.rec_proc = start_capture()
    self.rec_proc_holder['proc'] = self.rec_proc

  def watchdog_tick(self):
    with self.lock:
      if self.stop_evt.is_set():
        return
      if self.rec_proc and self.rec_proc.poll() is not None:
        _warn("parec died, restarting capture")
        try:
          self.rec_proc = start_capture()
          self.rec_proc_holder['proc'] = self.rec_proc
        except Exception as e:
          _err(f"could not restart capture: {e}")

  def shutdown(self):
    with self.lock:
      self.stop_evt.set()
      if self.rec_proc and self.rec_proc.poll() is None:
        try:
          self.rec_proc.terminate()
          try: self.rec_proc.wait(timeout=2)
          except subprocess.TimeoutExpired: self.rec_proc.kill()
        except Exception as e:
          _warn(f"could not stop parec: {e}")
      if self.make_default:
        if self.previous_default:
          set_default_sink(self.previous_default)
        else:
          clear_default_sink()
      unload_null_sink(self.module_id)


# ---------- status display ----------

def status_loop(stats, stop_evt, capture_mgr):
  t0 = time.monotonic()
  while not stop_evt.is_set():
    capture_mgr.watchdog_tick()
    elapsed = time.monotonic() - t0
    state = 'connected' if stats.get('connected') else 'disconnected'
    msg = (f"\r  [{state:>12s}]  rms={stats.get('rms', 0):.4f}  "
           f"peak={stats.get('peak', 0):.4f}  "
           f"tx={stats.get('tx_bytes', 0)/1024/1024:.1f}MB  "
           f"reconn={stats.get('reconn', 0):>2d}  "
           f"t={elapsed:6.1f}s  ")
    sys.stdout.write(msg)
    sys.stdout.flush()
    stop_evt.wait(1.0)
  print()


# ---------- modes ----------

def cmd_test():
  """Spawn play_server with the WAV-write env hook on localhost; round-trip a sine sweep."""
  _step("end-to-end test (no device required)")
  test_wav = '/tmp/comma_speaker_test.wav'
  if os.path.exists(test_wav):
    os.unlink(test_wav)

  # use a different port so we don't collide with anything real
  test_port = 7778
  env = dict(os.environ, COMMA_SPEAKER_TEST=test_wav)
  # patch PORT in the subprocess via a tiny wrapper module override:
  # the rewritten play_server.py uses PORT=7777 — for the test, run a one-shot
  # python that imports and overrides it
  bootstrap = (
    f"import sys; sys.path.insert(0, '{_HERE}'); "
    f"import play_server; play_server.PORT = {test_port}; play_server.main()"
  )
  _info(f"spawning play_server in TEST MODE on 127.0.0.1:{test_port}")
  proc = subprocess.Popen([sys.executable, '-c', bootstrap],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  # wait for server to bind by retrying the actual connect
  # (we can't tcp_probe — the test server's accept() only fires once)
  sock = None
  for _ in range(50):
    time.sleep(0.1)
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.connect(('127.0.0.1', test_port))
      break
    except OSError:
      sock.close()
      sock = None
  if sock is None:
    proc.kill()
    _err("test server never came up")
    print(proc.stdout.read().decode(errors='ignore'))
    print(proc.stderr.read().decode(errors='ignore'), file=sys.stderr)
    sys.exit(1)
  _ok("test server listening")

  # build a known signal: 5 s log sine sweep 50 Hz → 18 kHz, then 1 s silence
  duration = 5.0
  silence = 1.0
  n = int(duration * SAMPLE_RATE)
  t = np.arange(n) / SAMPLE_RATE
  k = (18000 / 50.0) ** (1 / duration)
  phase = 2 * np.pi * 50 * (k ** t - 1) / np.log(k)
  sweep = (0.5 * np.sin(phase)).astype(np.float32)
  silence_arr = np.zeros(int(silence * SAMPLE_RATE), dtype=np.float32)
  signal_data = np.concatenate([sweep, silence_arr])

  _info(f"sending {len(signal_data)} samples ({len(signal_data)/SAMPLE_RATE:.1f}s)")
  mono_block_bytes = SAMPLE_BUFFER * 4  # f32 mono per block (server-side block size)
  bytes_sent = 0
  view = memoryview(signal_data.tobytes())
  while bytes_sent < len(view):
    n_send = min(mono_block_bytes, len(view) - bytes_sent)
    sock.sendall(view[bytes_sent:bytes_sent + n_send])
    bytes_sent += n_send
  sock.close()

  # let the server flush the WAV and exit
  try:
    proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
    proc.terminate()
    try: proc.wait(timeout=2)
    except subprocess.TimeoutExpired: proc.kill()

  if not os.path.exists(test_wav):
    _err("test WAV was never written")
    print(proc.stderr.read().decode(errors='ignore'), file=sys.stderr)
    sys.exit(1)

  # read back and compare
  import wave
  with wave.open(test_wav, 'rb') as wf:
    assert wf.getnchannels() == 1
    assert wf.getsampwidth() == 2
    assert wf.getframerate() == SAMPLE_RATE
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)
  received_int16 = np.frombuffer(raw, dtype=np.int16)
  expected_int16 = np.clip(signal_data * 32767.0, -32768, 32767).astype(np.int16)

  if len(received_int16) != len(expected_int16):
    _err(f"sample count mismatch: sent {len(expected_int16)}, got {len(received_int16)}")
    sys.exit(1)
  if not np.array_equal(received_int16, expected_int16):
    diffs = np.where(received_int16 != expected_int16)[0]
    _err(f"data mismatch: first diff at sample {diffs[0]} of {len(expected_int16)}")
    _err(f"  expected {expected_int16[diffs[0]]} got {received_int16[diffs[0]]}")
    _err(f"  total diff samples: {len(diffs)}")
    sys.exit(1)
  _ok(f"passthrough OK — {len(received_int16)} samples bit-identical "
      f"({len(received_int16)/SAMPLE_RATE:.1f}s, {len(received_int16)*2/1024:.1f} KB)")


def cmd_ping(dongle_id, ip, pid):
  """Play a 1-second 440 Hz beep on the device, prompt y/n."""
  _step("playing 440 Hz beep on device")
  n = SAMPLE_RATE
  t = np.arange(n) / SAMPLE_RATE
  beep = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
  # 100 ms ramp in/out to avoid clicks
  ramp = int(0.1 * SAMPLE_RATE)
  beep[:ramp] *= np.linspace(0, 1, ramp)
  beep[-ramp:] *= np.linspace(1, 0, ramp)

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect((ip, PORT))
  sock.sendall(beep.tobytes())
  # send 0.5 s of silence so it doesn't loop or click
  sock.sendall(np.zeros(SAMPLE_RATE // 2, dtype=np.float32).tobytes())
  sock.close()
  time.sleep(2.0)
  try:
    ans = input("  did you hear the beep on the comma device? [y/N] ").strip().lower()
  except (EOFError, KeyboardInterrupt):
    ans = ''
  if ans == 'y':
    _ok("ping confirmed")
    save_cached(ping_ok=True)
    return True
  _warn("no confirmation — check device speaker / network")
  return False


def cmd_stop(dongle_id):
  _step(f"stopping any server on comma-{dongle_id}")
  stop_remote_server(dongle_id)
  _ok("done")


def cmd_run(dongle_id, alias, ip, make_default):
  _step(f"comma_speaker → {alias or 'host'} ({dongle_id or 'no-dongle'}) at {ip}")

  # 1. install + start remote server (skip if already reachable — local test, or
  #    a server already running from a prior session)
  pid = None
  already_up = tcp_probe(ip, PORT, timeout=0.5)
  if already_up:
    _info(f"server already reachable at {ip}:{PORT} — skipping SSH install")
  elif dongle_id:
    install_server(dongle_id)
    pid = start_remote_server(dongle_id)
    atexit.register(lambda: stop_remote_server(dongle_id, pid))
    _info(f"waiting for {ip}:{PORT}")
    if not wait_for_port(ip, PORT, timeout=15):
      _err(f"server not reachable at {ip}:{PORT} — is the device on the same WiFi?")
      _err("       try: ./comma_speaker.py --ip <other-ip>")
      sys.exit(1)
  else:
    _err(f"no server at {ip}:{PORT} and no dongle_id to SSH-install one")
    _err("       start play_server.py manually, or pass --device <name>")
    sys.exit(1)
  _ok("server reachable")

  # 3. setup local audio capture
  _step("setting up local audio sink")
  capture_mgr = CaptureManager()
  atexit.register(capture_mgr.shutdown)
  capture_mgr.start(make_default=make_default)
  _ok(f"virtual sink '{SINK_DESC}' running")

  # 4. start streaming
  _step("streaming")
  stop_evt = threading.Event()
  stats = {'tx_bytes': 0, 'rms': 0.0, 'peak': 0.0, 'reconn': 0, 'connected': False}

  def handle_signal(*_):
    print()
    _info("shutting down")
    stop_evt.set()
  signal.signal(signal.SIGINT, handle_signal)
  signal.signal(signal.SIGTERM, handle_signal)

  st_thread = threading.Thread(target=streamer,
                               args=(capture_mgr.rec_proc_holder, ip, PORT, stop_evt, stats),
                               daemon=True)
  st_thread.start()

  status_loop(stats, stop_evt, capture_mgr)
  st_thread.join(timeout=2)
  capture_mgr.shutdown()
  if dongle_id and pid:
    stop_remote_server(dongle_id, pid)
  _ok("clean exit")


# ---------- main ----------

def parse_args():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--device', help="comma device name (alias) or 16-char dongle id")
  p.add_argument('--ip', help="device LAN IP (skips discovery)")
  p.add_argument('--no-default', action='store_true',
                 help="don't set Comma Speaker as system default sink")
  p.add_argument('--test', action='store_true',
                 help="end-to-end roundtrip test (no device needed)")
  p.add_argument('--ping', action='store_true',
                 help="play a 440 Hz beep on the device, prompt y/n")
  p.add_argument('--stop', action='store_true',
                 help="kill any running server on the device, then exit")
  return p.parse_args()


def check_prereqs(need_pipewire=True):
  missing = []
  if need_pipewire:
    for tool in ('pactl', 'parec', 'pw-metadata'):
      if shutil.which(tool) is None:
        missing.append(tool)
  if missing:
    _err("missing required tools: " + ', '.join(missing))
    _err("       install pipewire-pulse and pipewire-utils")
    sys.exit(1)


def main():
  args = parse_args()

  # test mode is fully self-contained, no device or PipeWire needed
  if args.test:
    cmd_test()
    return

  lock = acquire_lock()  # noqa: F841 — kept open for the lifetime of the process

  cached = load_cached()
  dongle_id, alias, ip = None, None, args.ip

  # fast path: --ip given AND already reachable → skip dongle_id resolution entirely
  # (covers localhost test and "server still running from a prior session")
  if ip and tcp_probe(ip, PORT, timeout=0.5):
    _info(f"server reachable at {ip}:{PORT} — skipping device resolution")
  else:
    # need a dongle_id to SSH-install + start the server
    if args.device:
      dongle_id, alias = resolve_device(args.device)
    elif cached.get('dongle_id'):
      dongle_id = cached['dongle_id']
      alias = cached.get('alias') or dongle_id
      _info(f"using cached device: {alias} ({dongle_id})")
    else:
      dongle_id, alias = resolve_device(None)
    if not dongle_id:
      _err("no device selected")
      sys.exit(1)

    if not ip and cached.get('ip') and cached.get('dongle_id') == dongle_id:
      if tcp_probe(cached['ip']):
        ip = cached['ip']
        _info(f"reusing cached IP: {ip}")
    if not ip:
      _info("scanning local subnet for the device on port 7777 ...")
      hits = discover_lan()
      if len(hits) == 1:
        ip = hits[0]
        _ok(f"found server at {ip}")
      elif len(hits) > 1:
        _info(f"found multiple candidates: {hits} — picking first; pass --ip to override")
        ip = hits[0]
    if not ip:
      _info("asking comma proxy for device LAN IP ...")
      ip = get_lan_ip_via_proxy(dongle_id)
      if ip:
        _ok(f"device reports IP: {ip}")
    if not ip:
      try:
        ip = input("  could not auto-discover. enter device LAN IP: ").strip()
      except (EOFError, KeyboardInterrupt):
        sys.exit(1)
    if not ip:
      _err("no IP")
      sys.exit(1)

    save_cached(dongle_id=dongle_id, alias=alias, ip=ip)

  if args.stop:
    cmd_stop(dongle_id)
    return

  if args.ping:
    install_server(dongle_id)
    pid = start_remote_server(dongle_id)
    atexit.register(lambda: stop_remote_server(dongle_id, pid))
    if not wait_for_port(ip, PORT, timeout=15):
      _err(f"server not reachable at {ip}:{PORT}")
      sys.exit(1)
    cmd_ping(dongle_id, ip, pid)
    return

  check_prereqs(need_pipewire=True)
  cmd_run(dongle_id, alias, ip, make_default=not args.no_default)


if __name__ == '__main__':
  main()
