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
import platform
import queue
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


_OPENPILOT_ROOT = Path(__file__).resolve().parent.parent.parent


def _require_openpilot_venv():
  """numpy + sounddevice are pinned in openpilot's pyproject.toml. Refuse to run
  outside the openpilot venv so users get one canonical setup path."""
  try:
    import numpy  # noqa: F401
    return
  except ImportError:
    pass
  msg = (
    "  ERR numpy is missing — this script must run inside the openpilot venv.\n"
    f"      cd {_OPENPILOT_ROOT}\n"
    "      uv sync       # one-time, populates .venv with all openpilot deps\n"
    "      op venv       # activate, then re-run this script\n"
  )
  sys.exit(msg)


_require_openpilot_venv()
import numpy as np

IS_LINUX = sys.platform.startswith('linux')
IS_MAC   = sys.platform == 'darwin'

# add openpilot root to path so CommaApi import works
_HERE = Path(__file__).resolve().parent
_BASEDIR = _HERE.parent.parent
sys.path.insert(0, str(_BASEDIR))

PORT = 7777
SSH_PROXY = 'ssh.comma.ai'
SSH_KEY_SRC = _BASEDIR / 'system' / 'hardware' / 'tici' / 'id_rsa'
LOCAL_SERVER = _HERE / 'play_server.py'


def _safe_ssh_key():
  """ssh on macOS (and recent OpenSSH on Linux) refuses keys with permissive
  modes. The bundled key is committed as 0664. Copy it to a 0600 tempfile and
  use that path. Cached for the lifetime of the process."""
  global _SSH_KEY_CACHED
  try:
    return _SSH_KEY_CACHED
  except NameError:
    pass
  import tempfile
  fd, path = tempfile.mkstemp(prefix='comma_speaker_key_', suffix='.pem')
  with os.fdopen(fd, 'wb') as f:
    f.write(SSH_KEY_SRC.read_bytes())
  os.chmod(path, 0o600)
  atexit.register(lambda: os.path.exists(path) and os.unlink(path))
  globals()['_SSH_KEY_CACHED'] = path
  return path

CACHE_DIR = Path.home() / '.cache' / 'comma_speaker'
CACHE_FILE = CACHE_DIR / 'device.json'
LOCK_FILE = '/tmp/comma_speaker.pid'

REMOTE_DIR = '/data/dsp_sound'
REMOTE_SERVER = f'{REMOTE_DIR}/play_server.py'
REMOTE_LOG = '/tmp/play_server.log'
# the openpilot venv ships numpy + sounddevice; system python3 has neither
REMOTE_PY = '/usr/local/venv/bin/python'

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
  """Cross-platform /24 subnet enumeration: open a UDP socket to a public IP and
  read back our outgoing source address. No packet is actually sent. Works
  identically on Linux and macOS."""
  subnets = set()
  for probe_ip in ('8.8.8.8', '1.1.1.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      s.connect((probe_ip, 80))
      ip = s.getsockname()[0]
      if not ip.startswith('127.'):
        m = re.match(r'(\d+\.\d+\.\d+)\.\d+', ip)
        if m:
          subnets.add(m.group(1))
    except OSError:
      pass
    finally:
      s.close()
  # also try resolving our own hostname (catches additional interfaces)
  try:
    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
      if not ip.startswith('127.'):
        m = re.match(r'(\d+\.\d+\.\d+)\.\d+', ip)
        if m:
          subnets.add(m.group(1))
  except (socket.gaierror, OSError):
    pass
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


def _ssh_common_opts(key):
  return [
    '-i', key,
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'UserKnownHostsFile=/dev/null',
    '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=15',
  ]


def _ssh_target_args(dongle_id=None, ip=None):
  """Return (ssh_prefix_args, host) for either direct or proxy connection.
  Pass exactly one of dongle_id, ip."""
  key = _safe_ssh_key()
  opts = _ssh_common_opts(key)
  if ip:
    return opts, f"comma@{ip}"
  proxy = (f"ssh -i {key} -o StrictHostKeyChecking=no "
           f"-o UserKnownHostsFile=/dev/null -W %h:%p comma@{SSH_PROXY}")
  return opts + ['-o', f"ProxyCommand={proxy}"], f"comma@comma-{dongle_id}"


def ssh_run(target, *cmd, timeout=30, capture=True):
  """target: dict with 'dongle_id' or 'ip'."""
  opts, host = _ssh_target_args(**target)
  args = ['ssh'] + opts + [host]
  if cmd:
    args.append('--')
    args.extend(cmd)
  return subprocess.run(args, capture_output=capture, text=True, timeout=timeout)


def scp_run(target, local_path, remote_path, timeout=30):
  opts, host = _ssh_target_args(**target)
  args = ['scp'] + opts + [str(local_path), f"{host}:{remote_path}"]
  return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def get_lan_ip_via_proxy(dongle_id):
  """Ask the device via SSH proxy what its LAN IPs are; return first non-loopback."""
  r = ssh_run({'dongle_id': dongle_id}, "ip -4 -j addr show", timeout=20)
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

def install_server(target):
  _info(f"installing server to {REMOTE_SERVER}")
  r = ssh_run(target, f"mkdir -p {REMOTE_DIR}", timeout=15)
  if r.returncode != 0:
    raise RuntimeError(f"mkdir failed: {r.stderr.strip()}")
  r = scp_run(target, LOCAL_SERVER, REMOTE_SERVER)
  if r.returncode != 0:
    raise RuntimeError(f"scp failed: {r.stderr.strip()}")
  r = ssh_run(target, f"chmod +x {REMOTE_SERVER}", timeout=10)
  if r.returncode != 0:
    raise RuntimeError(f"chmod failed: {r.stderr.strip()}")
  _ok("server installed")


# anchor the pattern at the python interpreter path so it only matches the
# server process, never the shell that's running pkill (whose argv contains
# REMOTE_SERVER as text and would otherwise get killed → SSH dies → rc=255)
_REMOTE_PROC_PAT = f'^{re.escape(REMOTE_PY)} .*{re.escape(REMOTE_SERVER.split("/")[-1])}$'


def start_remote_server(target):
  _info("starting server on device")
  # detach via subshell + setsid so the backgrounded python doesn't keep ssh's
  # stdio fds, then verify with pgrep
  cmd = (
    f"pkill -f '{_REMOTE_PROC_PAT}' 2>/dev/null; "
    f"sleep 0.4; "
    f"( setsid {REMOTE_PY} {REMOTE_SERVER} >{REMOTE_LOG} 2>&1 </dev/null & ) ; "
    f"sleep 0.8; "
    f"pgrep -f '{_REMOTE_PROC_PAT}' | head -1"
  )
  r = ssh_run(target, cmd, timeout=20)
  pid = (r.stdout or '').strip().splitlines()
  pid = pid[-1] if pid else ''
  if not pid.isdigit():
    raise RuntimeError(
      f"start failed (rc={r.returncode}) stdout={r.stdout!r} stderr={r.stderr!r}")
  _ok(f"server pid {pid}")
  return pid


def stop_remote_server(target, pid=None):
  parts = []
  if pid and pid != "?":
    parts.append(f"kill {pid} 2>/dev/null")
  parts.append(f"pkill -f '{_REMOTE_PROC_PAT}' 2>/dev/null")
  parts.append("true")
  cmd = "; ".join(parts)
  try:
    ssh_run(target, cmd, timeout=15)
  except Exception as e:
    _warn(f"could not stop remote server cleanly: {e}")


def wait_for_port(ip, port, timeout=15):
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    if tcp_probe(ip, port, timeout=0.5):
      return True
    time.sleep(0.3)
  return False


# ---------- audio backends (cross-platform) ----------
#
# Each backend exposes the same interface to the streamer:
#   check_prereqs()      raise with install instructions if anything is missing
#   start(make_default)  set up virtual sink + capture, optionally hijack default
#   stop()               tear down capture + sink, restore default
#   read(timeout)        return next stereo f32 block (numpy array of shape (N,2)) or None
#   alive()              True if capture is still healthy
#   restart()            best-effort restart of capture only

CAPTURE_CHANNELS = 2  # always stereo; streamer downmixes
QUEUE_DEPTH = 32      # ~2.7s of f32 stereo at 48kHz/4096


class _BaseBackend:
  def __init__(self):
    self.q = queue.Queue(maxsize=QUEUE_DEPTH)
    self.previous_default = None
    self.make_default = True
    self.stats = {'q_drops': 0}

  def _enqueue(self, block_stereo):
    """Common enqueue logic with drop-oldest on overflow."""
    try:
      self.q.put_nowait(block_stereo)
    except queue.Full:
      try: self.q.get_nowait()
      except queue.Empty: pass
      try: self.q.put_nowait(block_stereo)
      except queue.Full: pass
      self.stats['q_drops'] += 1

  def read(self, timeout=1.0):
    try:
      return self.q.get(timeout=timeout)
    except queue.Empty:
      return None


# ---- auto-install of platform audio dependencies ----

def _auto_install_linux(missing):
  """Install missing PipeWire/Pulse tools via the system package manager.
  Returns True iff all required tools are present afterwards."""
  if shutil.which('apt-get'):
    pkgs = ['pipewire-pulse', 'pulseaudio-utils', 'pipewire-bin']
    cmd = ['sudo', 'apt-get', 'install', '-y'] + pkgs
  elif shutil.which('dnf'):
    pkgs = ['pipewire-pulseaudio', 'pulseaudio-utils', 'pipewire-utils']
    cmd = ['sudo', 'dnf', 'install', '-y'] + pkgs
  elif shutil.which('pacman'):
    pkgs = ['pipewire-pulse', 'pipewire']
    cmd = ['sudo', 'pacman', '-S', '--noconfirm'] + pkgs
  else:
    _err("no supported package manager (apt/dnf/pacman) found")
    return False
  _info(f"missing: {', '.join(missing)} — auto-installing: {' '.join(pkgs)}")
  _info("(may prompt for sudo password)")
  try:
    subprocess.run(cmd, timeout=300)
  except Exception as e:
    _warn(f"auto-install failed: {e}")
    return False
  return True


def _auto_install_mac(need_blackhole=False, need_switchaudio=False):
  """Install missing macOS audio system tools via brew. Returns True on success."""
  ok = True
  if need_blackhole or need_switchaudio:
    if not shutil.which('brew'):
      _err("Homebrew not installed — install from https://brew.sh, then re-run.")
      return False
    if need_blackhole:
      _info("installing: brew install --cask blackhole-2ch")
      r = subprocess.run(['brew', 'install', '--cask', 'blackhole-2ch'])
      ok = ok and r.returncode == 0
    if need_switchaudio:
      _info("installing: brew install switchaudio-osx")
      r = subprocess.run(['brew', 'install', 'switchaudio-osx'])
      ok = ok and r.returncode == 0
    if need_blackhole and ok:
      _info("restarting CoreAudio so BlackHole device shows up (will prompt for sudo)")
      subprocess.run(['sudo', 'killall', 'coreaudiod'])
  return ok


# ---- Linux: pactl null-sink + pw-metadata default + parec subprocess ----

class LinuxBackend(_BaseBackend):
  def __init__(self):
    super().__init__()
    self.module_id = None
    self.rec_proc = None
    self.reader_thread = None
    self.stop_evt = threading.Event()

  def check_prereqs(self):
    needed = ('pactl', 'parec', 'pw-metadata')
    missing = [t for t in needed if shutil.which(t) is None]
    if missing:
      _auto_install_linux(missing)
      missing = [t for t in needed if shutil.which(t) is None]
    if missing:
      raise RuntimeError(
        "missing tools after install attempt: " + ', '.join(missing) +
        " — on Debian/Ubuntu try: sudo apt install pipewire-pulse pulseaudio-utils pipewire-bin")

  @staticmethod
  def get_default_sink():
    try:
      r = subprocess.run(['pw-metadata', '0', 'default.audio.sink'],
                         capture_output=True, text=True, timeout=3)
      m = re.search(r'"name"\s*:\s*"([^"]+)"', r.stdout)
      return m.group(1) if m else None
    except Exception:
      return None

  @staticmethod
  def set_default_sink(name):
    subprocess.run(['pw-metadata', '0', 'default.audio.sink', f'{{"name":"{name}"}}'],
                   check=False, capture_output=True)

  @staticmethod
  def clear_default_sink():
    subprocess.run(['pw-metadata', '-d', '0', 'default.audio.sink'],
                   check=False, capture_output=True)

  def _load_null_sink(self):
    r = subprocess.run([
      'pactl', 'load-module', 'module-null-sink',
      f'sink_name={SINK_NAME}',
      f'sink_properties=device.description={SINK_DESC}',
    ], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
      raise RuntimeError(f"pactl load-module failed: {r.stderr.strip()}")
    module_id = r.stdout.strip()
    for _ in range(30):
      time.sleep(0.1)
      sinks = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                             capture_output=True, text=True, timeout=5).stdout
      if SINK_NAME in sinks:
        return module_id
    raise RuntimeError(f"null sink '{SINK_NAME}' never appeared")

  def _start_parec(self):
    cmd = ['parec',
           f'--device={SINK_NAME}.monitor',
           '--format=float32le',
           f'--rate={SAMPLE_RATE}',
           f'--channels={CAPTURE_CHANNELS}',
           '--raw']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

  def _reader_loop(self):
    block_bytes = SAMPLE_BUFFER * 4 * CAPTURE_CHANNELS
    buf = b''
    while not self.stop_evt.is_set():
      proc = self.rec_proc
      if proc is None or proc.poll() is not None:
        time.sleep(0.05)
        continue
      try:
        data = proc.stdout.read(block_bytes)
      except Exception:
        time.sleep(0.05)
        continue
      if not data:
        time.sleep(0.05)
        continue
      buf += data
      while len(buf) >= block_bytes:
        chunk = buf[:block_bytes]
        buf = buf[block_bytes:]
        stereo = np.frombuffer(chunk, dtype=np.float32).reshape(-1, CAPTURE_CHANNELS).copy()
        self._enqueue(stereo)

  def start(self, make_default=True):
    self.make_default = make_default
    self.module_id = self._load_null_sink()
    if make_default:
      self.previous_default = self.get_default_sink()
      self.set_default_sink(SINK_NAME)
      _ok(f"set default sink → '{SINK_DESC}' (was: {self.previous_default or 'none'})")
    else:
      _info(f"sink '{SINK_DESC}' available — route apps via pavucontrol/Helvum")
    self.rec_proc = self._start_parec()
    self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
    self.reader_thread.start()

  def alive(self):
    return self.rec_proc is not None and self.rec_proc.poll() is None

  def restart(self):
    _warn("parec died, restarting capture")
    if self.rec_proc:
      try: self.rec_proc.kill()
      except Exception: pass
    try:
      self.rec_proc = self._start_parec()
    except Exception as e:
      _err(f"could not restart capture: {e}")

  def stop(self):
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
        self.set_default_sink(self.previous_default)
      else:
        self.clear_default_sink()
    if self.module_id:
      subprocess.run(['pactl', 'unload-module', str(self.module_id)],
                     check=False, capture_output=True, timeout=5)


# ---- macOS: BlackHole 2ch as default output + sounddevice.InputStream ----

class MacBackend(_BaseBackend):
  """Captures from a BlackHole 2ch virtual audio device on macOS.

  BlackHole (https://existential.audio/blackhole) is the standard free virtual
  audio driver on macOS. Install with:  brew install blackhole-2ch
  Optional for default-sink switching: brew install switchaudio-osx
  """
  BLACKHOLE_NAME = 'BlackHole 2ch'

  def __init__(self):
    super().__init__()
    self.stream = None
    self.device_idx = None

  def check_prereqs(self):
    # sounddevice is pinned in openpilot's pyproject — _require_openpilot_venv()
    # at module load enforces the venv, so import here can't fail
    import sounddevice as sd  # noqa: F401
    need_switch = shutil.which('SwitchAudioSource') is None
    need_bh = self._find_blackhole() is None
    if need_switch or need_bh:
      _auto_install_mac(need_blackhole=need_bh, need_switchaudio=need_switch)
    if self._find_blackhole() is None:
      raise RuntimeError(
        f"'{self.BLACKHOLE_NAME}' not found after install. "
        "Try restarting CoreAudio: sudo killall coreaudiod, then re-run.")
    if shutil.which('SwitchAudioSource') is None:
      _warn("SwitchAudioSource still missing — '--no-default' will be implied")
    self.device_idx = self._find_blackhole()

  def _find_blackhole(self):
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
      if self.BLACKHOLE_NAME.lower() in d['name'].lower() and d['max_input_channels'] >= 2:
        return i
    return None

  @staticmethod
  def get_default_output():
    if shutil.which('SwitchAudioSource') is None:
      return None
    try:
      r = subprocess.run(['SwitchAudioSource', '-c', '-t', 'output'],
                         capture_output=True, text=True, timeout=3)
      return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
      return None

  @staticmethod
  def set_default_output(name):
    if shutil.which('SwitchAudioSource') is None:
      return False
    try:
      r = subprocess.run(['SwitchAudioSource', '-s', name, '-t', 'output'],
                         capture_output=True, text=True, timeout=5)
      return r.returncode == 0
    except Exception:
      return False

  def start(self, make_default=True):
    import sounddevice as sd
    self.make_default = make_default and shutil.which('SwitchAudioSource') is not None
    if self.device_idx is None:
      self.device_idx = self._find_blackhole()
    if self.device_idx is None:
      raise RuntimeError(f"'{self.BLACKHOLE_NAME}' device disappeared")

    if self.make_default:
      self.previous_default = self.get_default_output()
      if self.set_default_output(self.BLACKHOLE_NAME):
        _ok(f"set default output → '{self.BLACKHOLE_NAME}' (was: {self.previous_default or 'none'})")
      else:
        _warn(f"could not set default output; route apps to '{self.BLACKHOLE_NAME}' manually")
        self.make_default = False
    else:
      _info(f"capture device: '{self.BLACKHOLE_NAME}' — route apps to it in System Settings → Sound → Output")

    def cb(indata, frames, time_info, status):
      if status:
        # status flags include input overflow; mostly cosmetic
        pass
      # indata is a (frames, channels) numpy array of float32
      self._enqueue(indata.copy())

    self.stream = sd.InputStream(device=self.device_idx,
                                 channels=CAPTURE_CHANNELS,
                                 samplerate=SAMPLE_RATE,
                                 dtype='float32',
                                 blocksize=SAMPLE_BUFFER,
                                 callback=cb)
    self.stream.start()

  def alive(self):
    return self.stream is not None and self.stream.active

  def restart(self):
    _warn("InputStream went inactive, restarting")
    try:
      if self.stream:
        self.stream.stop()
        self.stream.close()
    except Exception: pass
    self.start(self.make_default)

  def stop(self):
    if self.stream:
      try: self.stream.stop()
      except Exception: pass
      try: self.stream.close()
      except Exception: pass
      self.stream = None
    if self.make_default and self.previous_default:
      self.set_default_output(self.previous_default)


def make_backend():
  if IS_LINUX: return LinuxBackend()
  if IS_MAC:   return MacBackend()
  raise RuntimeError(f"unsupported platform: {sys.platform} (only linux + darwin so far)")


# ---------- streamer ----------

def streamer(backend, ip, port, stop_evt, stats):
  """Pull stereo f32 numpy blocks from the backend queue, downmix to mono, send mono over TCP."""
  backoff = 0.2
  while not stop_evt.is_set():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
      sock.connect((ip, port))
    except OSError as e:
      stats['reconn'] += 1
      _warn(f"connect to {ip}:{port} failed ({e}); retrying in {backoff:.1f}s")
      sock.close()
      stop_evt.wait(backoff)
      backoff = min(5.0, backoff * 2)
      continue
    sock.settimeout(None)
    backoff = 0.2
    stats['connected'] = True
    try:
      while not stop_evt.is_set():
        block = backend.read(timeout=1.0)
        if block is None:
          continue  # backend has nothing yet — keep socket alive, wait more
        # block shape: (frames, CAPTURE_CHANNELS)
        mono = block.mean(axis=1).astype(np.float32)
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
        stats['q_drops'] = backend.stats.get('q_drops', 0)
    finally:
      stats['connected'] = False
      try: sock.close()
      except Exception: pass


# ---------- status display ----------

def status_loop(stats, stop_evt, backend):
  t0 = time.monotonic()
  while not stop_evt.is_set():
    if backend is not None and not backend.alive():
      backend.restart()
    elapsed = time.monotonic() - t0
    state = 'connected' if stats.get('connected') else 'disconnected'
    msg = (f"\r  [{state:>12s}]  rms={stats.get('rms', 0):.4f}  "
           f"peak={stats.get('peak', 0):.4f}  "
           f"tx={stats.get('tx_bytes', 0)/1024/1024:.1f}MB  "
           f"reconn={stats.get('reconn', 0):>2d}  "
           f"qdrop={stats.get('q_drops', 0):>2d}  "
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


def cmd_stop(target):
  _step(f"stopping any server on {_target_label(target)}")
  stop_remote_server(target)
  _ok("done")


def _target_label(target):
  return f"comma-{target['dongle_id']}" if target.get('dongle_id') else target.get('ip', '?')


def cmd_run(target, alias, ip, make_default):
  _step(f"comma_speaker → {alias or 'host'} ({_target_label(target) if target else 'no-target'}) at {ip}")

  # 1. install + start remote server (skip if already reachable — local test, or
  #    a server already running from a prior session)
  pid = None
  already_up = tcp_probe(ip, PORT, timeout=0.5)
  if already_up:
    _info(f"server already reachable at {ip}:{PORT} — skipping SSH install")
  elif target:
    install_server(target)
    pid = start_remote_server(target)
    atexit.register(lambda: stop_remote_server(target, pid))
    _info(f"waiting for {ip}:{PORT}")
    if not wait_for_port(ip, PORT, timeout=15):
      _err(f"server not reachable at {ip}:{PORT} — is the device on the same WiFi?")
      _err("       try: ./comma_speaker.py --ip <other-ip>")
      sys.exit(1)
  else:
    _err(f"no server at {ip}:{PORT} and no SSH target to install one")
    _err("       start play_server.py manually, or pass --device <name>")
    sys.exit(1)
  _ok("server reachable")

  # 3. setup local audio capture (platform-specific)
  _step("setting up local audio capture")
  backend = make_backend()
  backend.check_prereqs()
  atexit.register(backend.stop)
  backend.start(make_default=make_default)
  _ok(f"backend '{type(backend).__name__}' running")

  # 4. start streaming
  _step("streaming")
  stop_evt = threading.Event()
  stats = {'tx_bytes': 0, 'rms': 0.0, 'peak': 0.0, 'reconn': 0,
           'connected': False, 'q_drops': 0}

  def handle_signal(*_):
    print()
    _info("shutting down")
    stop_evt.set()
  signal.signal(signal.SIGINT, handle_signal)
  signal.signal(signal.SIGTERM, handle_signal)

  st_thread = threading.Thread(target=streamer,
                               args=(backend, ip, PORT, stop_evt, stats),
                               daemon=True)
  st_thread.start()

  status_loop(stats, stop_evt, backend)
  st_thread.join(timeout=2)
  backend.stop()
  if target and pid:
    stop_remote_server(target, pid)
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


def check_prereqs():
  """Best-effort up-front check; backend.check_prereqs() also runs at start()."""
  try:
    backend = make_backend()
    backend.check_prereqs()
  except Exception as e:
    _err(str(e))
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
  target = None  # dict with 'dongle_id' or 'ip' — used for SSH/SCP

  # --ip given: skip device resolution entirely. SSH directly to the IP for
  # install/start. The server may already be running, in which case SSH is
  # only used if the user passes --stop / --ping.
  if ip:
    target = {'ip': ip}
    if tcp_probe(ip, PORT, timeout=0.5):
      _info(f"server reachable at {ip}:{PORT}")
    else:
      _info(f"will SSH directly to {ip} to install/start server")
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
    target = {'dongle_id': dongle_id}

    if cached.get('ip') and cached.get('dongle_id') == dongle_id:
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
    cmd_stop(target)
    return

  if args.ping:
    install_server(target)
    pid = start_remote_server(target)
    atexit.register(lambda: stop_remote_server(target, pid))
    if not wait_for_port(ip, PORT, timeout=15):
      _err(f"server not reachable at {ip}:{PORT}")
      sys.exit(1)
    cmd_ping(dongle_id, ip, pid)
    return

  check_prereqs()
  cmd_run(target, alias, ip, make_default=not args.no_default)


if __name__ == '__main__':
  main()
