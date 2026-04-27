#!/usr/bin/env python3
"""Register a virtual PipeWire sink and stream audio to play_server.py.

Creates a virtual sink using pw-loopback:
  - 'comma_sink' — apps route their output here
  - 'comma_capture' — virtual source that pw-record captures from

The loopback routes audio from sink → source internally (no monitors needed).
"""
import atexit
import json
import signal
import socket
import subprocess
import sys
import time

SAMPLE_RATE = 48000
CHUNK = 4096 * 4

SINK_NAME = "comma_sink"
SINK_DESC = "Comma"
CAPTURE_NAME = "comma_capture"


def find_node(name_pattern):
  """Find a PipeWire node by node.name. Returns (id, name) or None."""
  try:
    result = subprocess.run(['pw-dump'], capture_output=True, text=True, timeout=5)
    objs = json.loads(result.stdout)
  except Exception as e:
    print(f"pw-dump failed: {e}", file=sys.stderr)
    return None
  for obj in objs:
    if obj.get('type') != 'PipeWire:Interface:Node':
      continue
    props = obj.get('info', {}).get('props', {})
    if props.get('node.name') == name_pattern:
      return (obj['id'], props.get('node.name'))
  return None


def list_audio_nodes():
  try:
    result = subprocess.run(['pw-dump'], capture_output=True, text=True, timeout=5)
    objs = json.loads(result.stdout)
    print("Audio nodes:", file=sys.stderr)
    for obj in objs:
      if obj.get('type') != 'PipeWire:Interface:Node':
        continue
      props = obj.get('info', {}).get('props', {})
      mc = props.get('media.class', '')
      if 'Audio' not in mc:
        continue
      print(f"  id={obj['id']:4}  {mc:28}  {props.get('node.name', '?'):30}  ({props.get('node.description', '')})",
            file=sys.stderr)
  except Exception:
    pass


def start_loopback():
  """Create virtual sink + source via pw-loopback. Stderr visible so you see errors."""
  proc = subprocess.Popen([
    'pw-loopback',
    f'--capture-props=media.class=Audio/Sink node.name={SINK_NAME} node.description={SINK_DESC}',
    f'--playback-props=media.class=Audio/Source/Virtual node.name={CAPTURE_NAME}',
  ], stderr=sys.stderr)
  atexit.register(_cleanup, proc)
  # wait for nodes to appear
  for _ in range(30):
    time.sleep(0.1)
    if proc.poll() is not None:
      print(f"ERROR: pw-loopback exited with code {proc.returncode}", file=sys.stderr)
      sys.exit(1)
    hit = find_node(CAPTURE_NAME)
    if hit is not None:
      return proc, hit
  print(f"ERROR: virtual source '{CAPTURE_NAME}' never appeared", file=sys.stderr)
  list_audio_nodes()
  sys.exit(1)


def _cleanup(proc):
  if proc.poll() is None:
    try:
      if proc.stdin: proc.stdin.close()
    except: pass
    proc.send_signal(signal.SIGTERM)
    try: proc.wait(timeout=2)
    except subprocess.TimeoutExpired: proc.kill()


def _skip_wav_header(stdout):
  """Skip the WAV/RIFF header that pw-record writes, leaving the stream at raw PCM data."""
  hdr = stdout.read(12)
  if len(hdr) < 12 or hdr[:4] != b'RIFF' or hdr[8:12] != b'WAVE':
    return hdr  # not WAV — return unconsumed bytes
  while True:
    chunk_hdr = stdout.read(8)
    if len(chunk_hdr) < 8:
      return b''
    chunk_id = chunk_hdr[:4]
    chunk_size = int.from_bytes(chunk_hdr[4:8], 'little')
    if chunk_id == b'data':
      return b''  # stream is now positioned at raw PCM samples
    stdout.read(chunk_size)  # skip non-data chunks (fmt, etc.)


def get_default_sink():
  import re
  try:
    result = subprocess.run(['pw-metadata', '0', 'default.audio.sink'],
                            capture_output=True, text=True, timeout=2)
    m = re.search(r'"name"\s*:\s*"([^"]+)"', result.stdout)
    if m:
      return m.group(1)
  except Exception:
    pass
  return None

def set_default_sink(name):
  subprocess.run(['pw-metadata', '0', 'default.audio.sink', f'{{"name":"{name}"}}'],
                 check=False, capture_output=True)

def clear_default_sink():
  subprocess.run(['pw-metadata', '-d', '0', 'default.audio.sink'],
                 check=False, capture_output=True)


def start_record(target, capture_sink=False):
  cmd = [
    'pw-record',
    '--target', str(target),
    '--format=f32', '--rate', str(SAMPLE_RATE), '--channels=1',
  ]
  if capture_sink:
    # needed to capture a sink's monitor (vs trying to play into the sink)
    cmd += ['--properties=stream.capture.sink=true']
  cmd.append('-')
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr)
  atexit.register(_cleanup, proc)
  return proc


def main(host, port, target, mode, make_default):
  if mode == 'monitor':
    # find the default sink's node ID and target it explicitly
    default_name = get_default_sink()
    if not default_name:
      print("ERROR: could not determine default sink", file=sys.stderr)
      list_audio_nodes()
      sys.exit(1)
    hit = find_node(default_name)
    if hit is None:
      print(f"ERROR: default sink '{default_name}' not found as a node", file=sys.stderr)
      list_audio_nodes()
      sys.exit(1)
    target = hit[0]
    print(f"Capturing default sink '{default_name}' (id={target})")
  elif mode == 'sink':
    _, (node_id, node_name) = start_loopback()
    print(f"Created virtual sink: id={node_id} name={node_name}")
    if make_default:
      prev = get_default_sink()
      set_default_sink(SINK_NAME)
      atexit.register(lambda: set_default_sink(prev) if prev else clear_default_sink())
      print(f"Set 'Comma' as default sink (was: {prev})")
    else:
      print("Route apps to 'Comma' via Helvum/qpwgraph/pavucontrol")
    target = node_id
  else:  # explicit target
    print(f"Using target: {target}")

  # monitor mode targets a sink → need stream.capture.sink=true
  rec = start_record(target, capture_sink=(mode == 'monitor'))
  time.sleep(0.3)
  if rec.poll() is not None:
    print("ERROR: pw-record exited immediately", file=sys.stderr)
    sys.exit(1)

  leftover = _skip_wav_header(rec.stdout)

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect((host, port))
  print(f"Streaming to {host}:{port} (Ctrl-C to stop)")

  try:
    import numpy as np
    total_bytes = 0
    last_print = time.monotonic()
    if leftover:
      sock.sendall(leftover)
      total_bytes += len(leftover)
    while True:
      data = rec.stdout.read(CHUNK)
      if not data:
        print("\npw-record stream ended", file=sys.stderr)
        break
      sock.sendall(data)
      total_bytes += len(data)
      # print RMS level every second so we can tell if we're capturing silence
      now = time.monotonic()
      if now - last_print >= 1.0:
        samples = np.frombuffer(data, dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        kbs = total_bytes / 1024 / (now - last_print + (now - last_print))  # rough
        sys.stdout.write(f"\r  rms={rms:.4f}  peak={peak:.4f}  total={total_bytes//1024} KB  ")
        sys.stdout.flush()
        last_print = now
  except KeyboardInterrupt:
    pass
  except (BrokenPipeError, ConnectionResetError):
    print("\nserver disconnected", file=sys.stderr)
  finally:
    sock.close()


if __name__ == "__main__":
  host = '127.0.0.1'
  port = 7777
  target = None
  mode = 'monitor'  # default: capture what's playing on default sink
  make_default = True
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == '--target' and i + 1 < len(sys.argv):
      i += 1
      target = sys.argv[i]
      mode = 'target'
    elif sys.argv[i] == '--sink':
      mode = 'sink'
    elif sys.argv[i] == '--no-default':
      make_default = False
    elif sys.argv[i] == '--host' and i + 1 < len(sys.argv):
      i += 1
      host = sys.argv[i]
    elif sys.argv[i] == '--port' and i + 1 < len(sys.argv):
      i += 1
      port = int(sys.argv[i])
    elif sys.argv[i] in ('-h', '--help'):
      print(f"Usage: {sys.argv[0]} [--host H] [--port P] [--sink | --target NAME]")
      print(f"  default: capture default audio output (everything you hear)")
      print(f"  --sink: create a virtual 'Comma' sink and capture from it")
      print(f"          --no-default: don't make it the default sink")
      print(f"  --target NAME: capture from a specific PipeWire node by name")
      sys.exit(0)
    i += 1

  main(host, port, target, mode, make_default)
