from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from serial import Serial


@dataclass
class TraceRecord:
  ts_monotonic: float
  direction: str
  line: str


def infer_direction(line: str) -> str:
  # Modem AT commands are echoed as "AT+...", responses are usually "OK"/"+..."/"ERROR".
  if line.startswith("AT"):
    return "tx"
  return "rx"


def write_record(fp: TextIO, rec: TraceRecord) -> None:
  fp.write(json.dumps({
    "ts_monotonic": rec.ts_monotonic,
    "direction": rec.direction,
    "line": rec.line,
  }) + "\n")
  fp.flush()


def run_mm_probe(period_s: float) -> None:
  cmd = ["mmcli", "-m", "any", "--output-json"]
  while True:
    subprocess.run(cmd, check=False, capture_output=True, text=True)
    time.sleep(period_s)


def trace_at_port(port: str, baudrate: int, timeout_s: float, duration_s: float, out_path: Path, run_probe: bool, probe_period_s: float) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  probe_proc: subprocess.Popen[str] | None = None
  if run_probe:
    # Trigger ModemManager AT traffic while tracing.
    probe_script = "\n".join([
      "import subprocess,time",
      f"period={probe_period_s}",
      "cmd=['mmcli','-m','any','--output-json']",
      "while True:",
      "  subprocess.run(cmd, check=False, capture_output=True, text=True)",
      "  time.sleep(period)",
    ])
    probe_proc = subprocess.Popen(
      ["python3", "-c", probe_script],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      text=True,
    )

  try:
    with Serial(port, baudrate=baudrate, timeout=timeout_s, write_timeout=timeout_s, exclusive=True) as ser, out_path.open("w", encoding="utf-8") as fp:
      ser.reset_input_buffer()
      deadline = time.monotonic() + duration_s
      print(f"[at-port-interceptor] tracing {port} for {duration_s:.1f}s -> {out_path}")
      while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
          continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
          continue
        rec = TraceRecord(
          ts_monotonic=time.monotonic(),
          direction=infer_direction(line),
          line=line,
        )
        write_record(fp, rec)
  finally:
    if probe_proc is not None:
      probe_proc.terminate()
      try:
        probe_proc.wait(timeout=2.0)
      except subprocess.TimeoutExpired:
        probe_proc.kill()


def summarize_trace(path: Path) -> list[str]:
  seen: set[str] = set()
  ordered: list[str] = []
  with path.open("r", encoding="utf-8") as fp:
    for line in fp:
      rec = json.loads(line)
      s = str(rec.get("line", ""))
      if s.startswith("AT") and s not in seen:
        seen.add(s)
        ordered.append(s)
  return ordered


def main() -> None:
  p = argparse.ArgumentParser(description="Trace AT port traffic while ModemManager is running.")
  p.add_argument("--port", default="/dev/ttyUSB2", help="AT serial port (default: /dev/ttyUSB2)")
  p.add_argument("--baudrate", type=int, default=115200)
  p.add_argument("--timeout", type=float, default=0.25)
  p.add_argument("--duration", type=float, default=60.0, help="Trace duration in seconds")
  p.add_argument("--out", default="/tmp/mm_at_trace.jsonl", help="Output jsonl path")
  p.add_argument("--run-mm-probe", action="store_true", help="Run periodic mmcli probe to trigger traffic")
  p.add_argument("--probe-period", type=float, default=1.0, help="Seconds between mmcli probes")
  args = p.parse_args()

  out_path = Path(args.out).expanduser().resolve()
  trace_at_port(
    port=args.port,
    baudrate=args.baudrate,
    timeout_s=args.timeout,
    duration_s=args.duration,
    out_path=out_path,
    run_probe=args.run_mm_probe,
    probe_period_s=args.probe_period,
  )

  commands = summarize_trace(out_path)
  print("[at-port-interceptor] unique AT commands observed:")
  for cmd in commands:
    print(f"  {cmd}")
  print("[at-port-interceptor] done")


if __name__ == "__main__":
  main()
