#!/usr/bin/env python3
import argparse
import contextlib
import math
import os
import queue
import re
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

KEYS = {"up": "cruise_up", "down": "cruise_down", "left": "blinker_left", "right": "blinker_right",
        'w': "gas", 's': "brake", 'a': "steer_left", 'd': "steer_right", 'g': "recenter", 'r': "restart", 'i': "ignition"}
ARROWS = {'A': "up", 'B': "down", 'C': "right", 'D': "left"}

RESET, BOLD, DIM = "\x1b[0m", "\x1b[1m", "\x1b[2m"
GREEN, YELLOW, RED, BLUE, CYAN, GREY = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[34m", "\x1b[36m", "\x1b[90m"
ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def badge(text: str, bg: int) -> str:
  return f"\x1b[1;30;{bg}m {text} {RESET}"

def key(k: str, color: str = "") -> str:
  return f"{color}\x1b[7m {k} {RESET}"

def colored(text: str, color: str) -> str:
  return f"{color}{text}{RESET}"


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
    self.messages.append(f"{GREY}{time.strftime('%H:%M:%S')}{RESET}  {msg}")

  def start(self):
    # reuse an openpilot that's already running, otherwise start one. it resets CalibrationParams, so the simulator must come after it's up
    for _ in range(15):
      self.sm.update(100)
    if self.sm.seen['managerState']:
      self.note("using the openpilot that's already running")
    else:
      self.manager = launch_openpilot(self.log)
      self.note("starting openpilot")

  def press(self, k: str):
    if k in (' ', '\r', '\n'):
      self.cmd_q.put("cruise_down" if str(self.sm['selfdriveState'].state) == 'disabled' else "cruise_cancel")
    elif k in KEYS:
      self.cmd_q.put(KEYS[k])
      if k == 'r':
        self.note("restarted the episode")
      elif k == 'g':
        self.note("back on the logged path")

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
        self.note(colored(f"episode over: {self.end_reason(info)}", RED))
      elif kind == "error":
        self.error = info.strip().splitlines()[-1]
        self.note(colored(f"simulator crashed: {self.error}", RED))

    if self.manager is not None and self.manager.poll() is not None and self.error is None:
      self.error = f"openpilot exited with {self.manager.returncode}"
      self.note(colored(self.error, RED))

    active = self.sm['selfdriveState'].active
    if active != self.engaged:
      self.engaged = active
      if active:
        self.note(colored("engaged", GREEN))
      else:
        events = [str(e.name) for e in self.sm['onroadEvents']]
        self.note(colored("disengaged", YELLOW) + (f"  {GREY}{', '.join(events)}{RESET}" if events else ""))

  def end_reason(self, status: dict) -> str:
    return {"route_end": "reached the end of the segment",
            "collision": "hit the lead car",
            "out_of_envelope": f"{abs(status['offset']):.1f} m / {abs(status['heading']):.1f}° off the logged path"}[status["ended"]]

  def speed(self, v: float) -> str:
    return f"{v * 3.6:.0f} km/h" if self.metric else f"{v * 2.237:.0f} mph"

  def state_line(self) -> str:
    ss = self.sm['selfdriveState']
    if self.error is not None:
      return f"{badge('ERROR', 41)}  {colored(self.error, RED)}  {GREY}see {self.log_path}{RESET}"
    if self.status is None:
      return f"{badge('STARTING' if self.simulator is None else 'LOADING', 44)}  {GREY}getting openpilot and the route ready{RESET}"
    if self.status["ended"] is not None:
      hint = "r to restart" if self.status["ended"] == "route_end" else "g to get back on the logged path, r to restart"
      return f"{badge('EPISODE OVER', 100)}  {self.end_reason(self.status)}  {GREY}{hint}{RESET}"
    if str(ss.state) == 'enabled':
      return badge("ENGAGED", 42)
    if str(ss.state) == 'preEnabled':
      return badge("ENGAGING", 42)
    if str(ss.state) == 'overriding':
      return f"{badge('OVERRIDE', 47)}  {GREY}you're driving, openpilot takes back over when you let go{RESET}"
    if str(ss.state) == 'softDisabling':
      return badge("DISENGAGING", 41)
    if ss.engageable:
      return f"{badge('READY', 43)}  {GREY}space to engage{RESET}"
    no_entry = [str(e.name) for e in self.sm['onroadEvents'] if e.noEntry]
    return f"{badge('NOT READY', 41)}  {GREY}{', '.join(no_entry)}{RESET}"

  def alert_line(self) -> str:
    ss = self.sm['selfdriveState']
    if not ss.alertText1:
      return ""
    color = {"userPrompt": YELLOW, "critical": RED}.get(str(ss.alertStatus), "")
    return f"{color}{BOLD}{ss.alertText1}{RESET}" + (f"  {color}{ss.alertText2}{RESET}" if ss.alertText2 else "")

  def lane_line(self, st: dict) -> str:
    """The car's offset from the logged path, on a gauge that spans the episode's limits"""
    n = 41
    if not st["closed_loop"]:
      return f"{GREY}{'─' * (n // 2)}●{'─' * (n // 2)}  following the log until openpilot engages{RESET}"
    offset = st["offset"]
    frac = min(abs(offset) / MAX_LATERAL_ERROR, 1.)
    color = GREEN if frac < 0.25 else YELLOW if frac < 0.6 else RED
    pos = round((max(-1., min(offset / MAX_LATERAL_ERROR, 1.)) + 1) / 2 * (n - 1))
    gauge = ''.join(f"{color}●{RESET}" if i == pos else f"{GREY}│{RESET}" if i == n // 2 else f"{GREY}─{RESET}" for i in range(n))
    return (f"{RED}├{RESET}{gauge}{RED}┤{RESET}  {color}{abs(offset):.2f} m {'right' if offset > 0 else 'left'}{RESET}  " +
            f"{abs(st['heading']):.1f}° {'right' if st['heading'] > 0 else 'left'}   {GREY}episode ends at {MAX_LATERAL_ERROR:.0f} m / " +
            f"{math.degrees(MAX_HEADING_ERROR):.0f}°{RESET}")

  def lead_line(self, st: dict) -> str:
    gap = st["lead_gap"]
    if math.isnan(gap):
      return colored("no lead", GREY)
    headway = gap / max(st["speed"], 0.1)
    color = RED if headway < 1 else YELLOW if headway < 2 else GREEN
    return f"{color}{gap:.0f} m{RESET}  {GREY}{headway:.1f} s ahead{RESET}"

  def inputs_line(self) -> str:
    cs = self.sm['carState']
    def lit(on: bool, text: str, color: str) -> str:
      return f"{color}{BOLD}{text}{RESET}" if on else colored(text, GREY)
    return "  ".join([lit(cs.leftBlinker, "◀ blinker", YELLOW), lit(cs.rightBlinker, "blinker ▶", YELLOW), lit(cs.gasPressed, "gas", GREEN),
                      lit(cs.brakePressed, "brake", RED), lit(cs.steeringPressed, "steering", CYAN)])

  def episode_line(self, st: dict) -> str:
    n = 30
    filled = round(min(st["t"] / max(st["duration"], 1e-3), 1.) * n)
    return (f"{GREEN}{'█' * filled}{GREY}{'░' * (n - filled)}{RESET}  {st['t']:.0f} / {st['duration']:.0f} s   " +
            f"{GREY}engaged {st['engaged_time']:.0f} s, worst offset {st['max_lateral_error']:.2f} m{RESET}")

  def rates_line(self) -> str:
    camera, model = self.rates["camera"].update(), self.rates["model"].update()
    def rate(hz: float) -> str:
      color = GREEN if hz >= 19 else YELLOW if hz >= MIN_MODEL_HZ else RED
      return colored(f"{hz:.0f} Hz", color)
    line = f"camera {rate(camera)}   model {rate(model)}"
    if model < MIN_MODEL_HZ:
      line += f"   {RED}the model is too slow for openpilot to drive{RESET}"
    return line

  def help_items(self) -> list[str]:
    disabled = str(self.sm['selfdriveState'].state) == 'disabled'
    engage = f"{key('space', GREEN if disabled and self.sm['selfdriveState'].engageable else '')} {'engage' if disabled else 'disengage'}"
    return [engage, f"{key('↑')}{key('↓')} set speed", f"{key('←')}{key('→')} blinkers", f"{key('w')}{key('s')} gas / brake",
            f"{key('a')}{key('d')} steer", f"{key('g')} back on the log", f"{key('r')} restart", f"{key('i')} ignition", f"{key('q')} quit"]

  def render(self, width: int) -> list[str]:
    lines = [f"{BOLD}openpilot sim{RESET}  {GREY}{self.args.route} from {self.args.start_sec:.0f} s{RESET}", "",
             f"  {self.state_line()}", f"  {self.alert_line()}", ""]
    if (st := self.status) is not None:
      cs = self.sm['carState']
      set_speed = f"   {GREY}set{RESET} {CYAN}{BOLD}{self.speed(cs.vCruise / 3.6)}{RESET}" if 0 < cs.vCruise < 255 else ""
      lines += [f"  {GREY}speed{RESET}    {BOLD}{self.speed(st['speed'])}{RESET}{set_speed}   {GREY}steering {st['steer']:+.1f}°{RESET}",
                f"  {GREY}lane{RESET}     {self.lane_line(st)}",
                f"  {GREY}lead{RESET}     {self.lead_line(st)}",
                f"  {GREY}inputs{RESET}   {self.inputs_line()}",
                f"  {GREY}episode{RESET}  {self.episode_line(st)}",
                f"  {GREY}rates{RESET}    {self.rates_line()}", ""]

    # key help, wrapped to the terminal
    row: list[str] = []
    for item in self.help_items():
      if row and visible_len("   ".join(row + [item])) + 2 > width:
        lines.append("  " + "   ".join(row))
        row = []
      row.append(item)
    lines += ["  " + "   ".join(row), "", *(f"  {m}" for m in self.messages)]
    return lines

  def close(self):
    if self.simulator is not None:
      self.simulator.terminate()
      self.simulator.join()
    if self.manager is not None:
      stop_openpilot(self.manager)
    self.log.close()


def read_keys(fd: int) -> list[str]:
  data = os.read(fd, 64).decode(errors="ignore")
  keys = []
  while data:
    if data.startswith("\x1b[") and len(data) > 2:
      keys.append(ARROWS.get(data[2], ""))
      data = data[3:]
    else:
      keys.append(data[0])
      data = data[1:]
  return keys


def visible_len(line: str) -> int:
  return len(ESCAPE.sub("", line))


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
        sys.stdout.write("\x1b[H" + "".join(clip(line, width) + "\x1b[K\n" for line in sim.render(width)) + "\x1b[J")
        sys.stdout.flush()
        if select.select([sys.stdin], [], [], 0.1)[0]:
          keys = read_keys(sys.stdin.fileno())
          if 'q' in keys:
            break
          for k in keys:
            sim.press(k)
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


if __name__ == "__main__":
  main()
