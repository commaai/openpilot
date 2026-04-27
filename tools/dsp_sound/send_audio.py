#!/usr/bin/env python3
"""Send any audio file to play_server.py. Uses ffmpeg to decode → float32 mono 48kHz."""
import socket
import subprocess
import sys
import time

SAMPLE_RATE = 48000
CHUNK = 4096 * 4  # SAMPLE_BUFFER samples * 4 bytes/float32

def main(path, host='127.0.0.1', port=7777):
  # decode with ffmpeg: any format → raw float32 LE, mono, 48kHz
  proc = subprocess.Popen([
    'ffmpeg', '-hide_banner', '-loglevel', 'error',
    '-i', path,
    '-f', 'f32le', '-ac', '1', '-ar', str(SAMPLE_RATE),
    '-',
  ], stdout=subprocess.PIPE)

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    sock.connect((host, port))
    print(f"Streaming {path} to {host}:{port}")
    bytes_per_sec = SAMPLE_RATE * 4
    start = time.monotonic()
    total = 0
    while True:
      data = proc.stdout.read(CHUNK)
      if not data:
        break
      sock.sendall(data)
      total += len(data)
      # gentle rate limit so we don't buffer the whole file
      target_time = total / bytes_per_sec - 1.0  # stay ~1s ahead
      now = time.monotonic() - start
      if now < target_time:
        time.sleep(target_time - now)
    print("Done")
  finally:
    sock.close()
    proc.wait()

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <audio_file> [host] [port]")
    sys.exit(1)
  host = sys.argv[2] if len(sys.argv) > 2 else '127.0.0.1'
  port = int(sys.argv[3]) if len(sys.argv) > 3 else 7777
  main(sys.argv[1], host, port)
