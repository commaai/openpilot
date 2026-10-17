#!/usr/bin/env python3
import argparse
import contextlib
import math
import os
import queue
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from collections import deque
from multiprocessing import Process, Queue

from openpilot.cereal import messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.tools.sim.simulator import start_simulator
from openpilot.tools.sim.world import MAX_HEADING_ERROR, MAX_LATERAL_ERROR

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROUTE = "8494c69d3c710e81|000001d4--2648a9a404/5"
DEFAULT_ROUTE_START = 15.  # the logged driver tailgates before this, and openpilot can't open up the gap
MIN_MODEL_HZ = 15

KEYS = {'1': "cruise_up", '2': "cruise_down", '3': "cruise_cancel", 'w': "gas", 's': "brake", 'a': "steer_left", 'd': "steer_right",
        'z': "blinker_left", 'x': "blinker_right", 'i': "ignition", 'r': "restart"}
HELP = ["2 engage / set-   1 resume / set+   3 cancel   w s gas / brake   a d steer",
        "z x blinkers   r restart   i ignition   q quit"]

GREEN, YELLOW, RED, DIM, BOLD, RESET = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m", "\x1b[1m", "\x1b[0m"


def launch_openpilot(log) -> subprocess.Popen:
  return subprocess.Popen([os.path.join(SIM_DIR, "launch_openpilot.sh")], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_openpilot(manager: subprocess.Popen):
  manager.terminate()
  try:
    manager.wait(15)
  except subprocess.TimeoutExpired:
    os.killpg(manager.pid, 9)
    manager.wait()


class Rate:
  """Messages per second on a service, over the last couple of seconds"""
  def __init__(self, service: str):
    self.sock = messaging.sub_sock(service, conflate=False)
    self.counts: deque[tuple[float, int]] = deque()

  def update(self) -> float:
    now = time.monotonic()
    self.counts.append((now, len(messaging.drain_sock_raw(self.sock))))
    while now - self.counts[0][0] > 2:
      self.counts.popleft()
    dt = now - self.counts[0][0]
    return sum(c for _, c in list(self.counts)[1:]) / dt if dt > 0.5 else math.nan


class Sim:
  def __init__(self, args):
    self.args = args
    self.cmd_q: Queue = Queue()
    self.status_q: Queue = Queue()
    self.sm = messaging.SubMaster(['selfdriveState', 'carState', 'onroadEvents', 'managerState'])
    self.rates = {"camera": Rate('narrowRoadCameraState'), "model": Rate('modelV2')}
    self.metric = Params().get_bool("IsMetric")
    self.log_path = os.path.join(Paths.comma_home(), "sim.log")
    self.log = open(self.log_path, "w")

    self.manager: subprocess.Popen | None = None
    self.simulator: Process | None = None
    self.status: dict | None = None
    self.error: str | None = None
    self.engaged = False
    self.messages: deque[str] = deque(maxlen=6)

  def note(self, msg: str):
    self.messages.append(f"{DIM}{time.strftime('%H:%M:%S')}{RESET} {msg}")

  def start(self):
    # reuse an openpilot that's already running, otherwise start one. it resets CalibrationParams, so the simulator must come after it's up
    for _ in range(15):
      self.sm.update(100)
    if self.sm.seen['managerState']:
      self.note("using the openpilot that's already running")
    else:
      self.manager = launch_openpilot(self.log)
      self.note("starting openpilot")

  def update(self):
    self.sm.update(0)
    if self.simulator is None and self.sm.seen['managerState']:
      self.simulator = start_simulator(self.args.route, self.args.start_sec, self.cmd_q, self.status_q, self.log_path)
      self.note(f"loading {self.args.route}")

    while True:
      try:
        kind, info = self.status_q.get_nowait()
      except queue.Empty:
        break
      if kind == "status":
        if self.status is None:
          self.note("openpilot engages as soon as it's ready")
        self.status = info
      elif kind == "episode_end":
        self.status = info
        self.note(f"{RED}episode over: {self.end_reason(info)}{RESET}, press r to restart")
      elif kind == "error":
        self.error = info.strip().splitlines()[-1]
        self.note(f"{RED}simulator crashed: {self.error}{RESET}")

    if self.manager is not None and self.manager.poll() is not None and self.error is None:
      self.error = f"openpilot exited with {self.manager.returncode}"
      self.note(f"{RED}{self.error}{RESET}")

    active = self.sm['selfdriveState'].active
    if active != self.engaged:
      self.engaged = active
      if active:
        self.note(f"{GREEN}engaged{RESET}")
      else:
        events = [str(e.name) for e in self.sm['onroadEvents']]
        self.note("disengaged" + (f": {', '.join(events)}" if events else ""))

  def end_reason(self, status: dict) -> str:
    return {"route_end": "reached the end of the segment",
            "collision": "hit the lead car",
            "out_of_envelope": f"{status['offset']:+.1f} m / {status['heading']:+.1f}° off the logged path " +
                               f"(limit {MAX_LATERAL_ERROR:.0f} m / {math.degrees(MAX_HEADING_ERROR):.0f}°)"}[status["ended"]]

  def speed(self, v: float) -> str:
    return f"{v * 3.6:.0f} km/h" if self.metric else f"{v * 2.237:.0f} mph"

  def openpilot_line(self) -> str:
    ss = self.sm['selfdriveState']
    if self.error is not None:
      return f"{RED}not running: {self.error}{RESET} (see {self.log_path})"
    if self.simulator is None:
      return "starting up"
    if self.status is None:
      return "loading the route"
    if ss.active:
      return f"{GREEN}{BOLD}engaged{RESET}"
    if self.status["ended"] is not None:
      return f"{YELLOW}episode over{RESET}, press r to restart"
    if ss.engageable:
      return f"{YELLOW}ready{RESET}, press 2 to engage"
    no_entry = [str(e.name) for e in self.sm['onroadEvents'] if e.noEntry]
    return f"{YELLOW}not ready{RESET}" + (f": {', '.join(no_entry)}" if no_entry else "")

  def render(self) -> list[str]:
    st, ss, cs = self.status, self.sm['selfdriveState'], self.sm['carState']
    lines = [f"{BOLD}openpilot sim{RESET}  {self.args.route}  from {self.args.start_sec:.0f} s", "",
             f"  openpilot  {self.openpilot_line()}"]
    alert = " · ".join(t for t in (ss.alertText1, ss.alertText2) if t)
    lines.append(f"  alert      {alert}" if alert else "")

    if st is not None:
      set_speed = f", set {self.speed(cs.vCruise / 3.6)}" if 0 < cs.vCruise < 255 else ""
      lines.append(f"  speed      {self.speed(st['speed'])}{set_speed}   steering {st['steer']:+.1f}°")
      if st["closed_loop"]:
        lines.append(f"  path       {st['offset']:+.2f} m, {st['heading']:+.1f}° from the logged path" +
                     f"   {DIM}(ends at {MAX_LATERAL_ERROR:.0f} m / {math.degrees(MAX_HEADING_ERROR):.0f}°){RESET}")
      else:
        lines.append(f"  path       {DIM}replaying the log until openpilot engages{RESET}")
      lines.append(f"  lead       {st['lead_gap']:.0f} m" if not math.isnan(st['lead_gap']) else "  lead       none")
      done = min(st['t'] / max(st['duration'], 1e-3), 1.)
      bar = "█" * round(done * 30) + "░" * (30 - round(done * 30))
      lines.append(f"  episode    {bar} {st['t']:.0f} / {st['duration']:.0f} s   engaged {st['engaged_time']:.0f} s, " +
                   f"worst offset {st['max_lateral_error']:.2f} m")

      camera, model = self.rates["camera"].update(), self.rates["model"].update()
      rates = f"  rates      camera {camera:.0f} Hz, model {model:.0f} Hz"
      if model < MIN_MODEL_HZ:
        rates += f"   {RED}the model is too slow for openpilot to drive{RESET}"
      lines.append(rates)

    lines += ["", *(f"  {DIM}{h}{RESET}" for h in HELP), "", *(f"  {m}" for m in self.messages)]
    return lines

  def close(self):
    if self.simulator is not None:
      self.simulator.terminate()
      self.simulator.join()
    if self.manager is not None:
      stop_openpilot(self.manager)
    self.log.close()


def main():
  parser = argparse.ArgumentParser(description="Drive openpilot through a logged route")
  parser.add_argument('--route', default=DEFAULT_ROUTE, help="segment to drive")
  parser.add_argument('--start-sec', type=float, help=f"seconds into the segment to start from (default: {DEFAULT_ROUTE_START:.0f} s on the default route)")
  args = parser.parse_args()
  if args.start_sec is None:
    args.start_sec = DEFAULT_ROUTE_START if args.route == DEFAULT_ROUTE else 0.

  # closing the terminal shuts down openpilot too
  for sig in (signal.SIGHUP, signal.SIGTERM):
    signal.signal(sig, signal.default_int_handler)

  sim = Sim(args)
  tui = sys.stdin.isatty() and sys.stdout.isatty()
  if tui:
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    sys.stdout.write("\x1b[?1049h\x1b[?25l")  # alternate screen, hide cursor
  try:
    sim.start()
    while True:
      sim.update()
      if tui:
        width = shutil.get_terminal_size().columns
        sys.stdout.write("\x1b[H" + "".join(clip(line, width) + "\x1b[K\n" for line in sim.render()) + "\x1b[J")
        sys.stdout.flush()
        if select.select([sys.stdin], [], [], 0.1)[0]:
          key = sys.stdin.read(1)
          if key == 'q':
            break
          if key in KEYS:
            sim.cmd_q.put(KEYS[key])
      else:
        time.sleep(0.1)
  except KeyboardInterrupt:
    pass
  finally:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # don't get interrupted cleaning up
    with contextlib.suppress(OSError):  # the terminal may be gone
      if tui:
        sys.stdout.write("\x1b[?25h\x1b[?1049l")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
      print("shutting down", flush=True)
    sim.close()
    with contextlib.suppress(OSError):
      print("\n".join(sim.messages) + RESET)


def clip(line: str, width: int) -> str:
  """Cuts a line to width printable characters, skipping over escape codes"""
  out, n, i = [], 0, 0
  while i < len(line):
    if line[i] == "\x1b":
      j = line.index("m", i) + 1
      out.append(line[i:j])
      i = j
      continue
    if n == width:
      break
    out.append(line[i])
    n += 1
    i += 1
  return "".join(out) + RESET


if __name__ == "__main__":
  main()
