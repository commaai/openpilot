#!/usr/bin/env python3
"""Fuzz test for jotpluggler — random UI interactions looking for crashes and freezes.
Each test gets its own Xvfb + process so all 20 run in parallel via xdist."""
from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

BINARY = Path(__file__).resolve().parent / "jotpluggler"
WIDTH, HEIGHT = 1280, 720
DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496"
FUZZ_ACTIONS = 300
SCREENSHOT_EVERY = 30
FREEZE_TIMEOUT_S = 8.0
ROUNDS_PER_SCENARIO = 10

KEYS = [
  "Escape", "Return", "Tab", "space", "Delete", "BackSpace",
  "Up", "Down", "Left", "Right",
  "ctrl+z", "ctrl+y", "ctrl+s", "ctrl+a", "ctrl+c", "ctrl+v",
  "Home", "End", "Page_Up", "Page_Down", "F1", "F5", "F11", "plus", "minus",
]


@dataclass(frozen=True)
class Scenario:
  name: str
  cabana: bool = False


SCENARIOS = [
  Scenario("plot"),
  Scenario("cabana", cabana=True),
]


class Session:
  def __init__(self, width: int = WIDTH, height: int = HEIGHT):
    self.xvfb = self.jotp = None
    self.display = self.winid = ""
    self.env: dict[str, str] = {}
    self.tmpdir = Path(tempfile.mkdtemp(prefix="jotp_fuzz_"))
    self._fhs = []
    self._w, self._h = width, height

  def start(self, *, route: str, cabana: bool = False, layout: str | None = None, sync_load: bool = False):
    self.xvfb = subprocess.Popen(
      ["Xvfb", "-displayfd", "1", "-screen", "0", f"{self._w}x{self._h}x24", "-dpi", "96", "-nolisten", "tcp"],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    self.display = f":{self.xvfb.stdout.readline().strip()}"
    assert self.display != ":"

    self.env = os.environ.copy()
    self.env["DISPLAY"] = self.display
    if cabana:
      self.env["JOTP_START_CABANA"] = "1"

    cmd = [str(BINARY), "--show", "--width", str(self._w), "--height", str(self._h)]
    if layout:
      cmd.extend(["--layout", layout])
    if sync_load:
      cmd.append("--sync-load")
    cmd.append(route)

    stdout_fh = open(self.tmpdir / "stdout.log", "w")
    stderr_fh = open(self.tmpdir / "stderr.log", "w")
    self._fhs = [stdout_fh, stderr_fh]
    self.jotp = subprocess.Popen(cmd, env=self.env, stdout=stdout_fh, stderr=stderr_fh)

    proc = subprocess.run(
      ["xdotool", "search", "--sync", "--name", "jotpluggler"],
      env=self.env, capture_output=True, text=True, timeout=30,
    )
    wids = proc.stdout.strip().splitlines()
    assert wids, "jotpluggler window never appeared"
    self.winid = wids[0]

    self._xdo("windowmove", self.winid, "0", "0")
    self._xdo("windowsize", "--sync", self.winid, str(self._w), str(self._h))
    self._xdo("windowfocus", "--sync", self.winid)
    time.sleep(2.0)

  def stop(self):
    for p in (self.jotp, self.xvfb):
      if p and p.poll() is None:
        p.terminate()
        try:
          p.wait(timeout=5)
        except subprocess.TimeoutExpired:
          p.kill()
    for fh in self._fhs:
      fh.close()
    if self.xvfb:
      for s in (self.xvfb.stdout, self.xvfb.stderr):
        if s:
          s.close()

  @property
  def alive(self):
    return self.jotp is not None and self.jotp.poll() is None

  def crash_reason(self) -> str:
    p = self.tmpdir / "stderr.log"
    text = p.read_text(errors="replace") if p.exists() else ""
    for pat in [r"Assertion.*failed", r"SIGSEGV|SIGABRT|SIGFPE|SIGBUS", r"AddressSanitizer"]:
      m = re.search(pat, text)
      if m:
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        return text[start:end if end != -1 else len(text)].strip()
    return f"exit code {self.jotp.poll() if self.jotp else '?'}"

  def stderr_tail(self, n: int = 20) -> str:
    p = self.tmpdir / "stderr.log"
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""

  def _xdo(self, *args: str, timeout: float = 5.0):
    subprocess.run(["xdotool", *args], env=self.env, check=True, timeout=timeout)

  def click(self, x, y, button=1):
    self._xdo("mousemove", str(x), str(y))
    time.sleep(0.02)
    self._xdo("click", str(button))
    time.sleep(0.05)
    return f"click ({x}, {y}) btn={button}"

  def doubleclick(self, x, y):
    self._xdo("mousemove", str(x), str(y))
    time.sleep(0.02)
    self._xdo("click", "--repeat", "2", "--delay", "80", "1")
    time.sleep(0.05)
    return f"dblclick ({x}, {y})"

  def drag(self, x1, y1, x2, y2):
    self._xdo("mousemove", str(x1), str(y1))
    time.sleep(0.02)
    self._xdo("mousedown", "1")
    for i in range(1, 7):
      a = i / 6
      self._xdo("mousemove", str(round(x1 + (x2 - x1) * a)), str(round(y1 + (y2 - y1) * a)))
      time.sleep(0.02)
    self._xdo("mouseup", "1")
    time.sleep(0.05)
    return f"drag ({x1},{y1})->({x2},{y2})"

  def scroll(self, x, y, clicks):
    self._xdo("mousemove", str(x), str(y))
    self._xdo("click", "--repeat", str(abs(clicks)), "4" if clicks > 0 else "5")
    time.sleep(0.05)
    return f"scroll ({x},{y}) n={clicks}"

  def key(self, keys):
    self._xdo("key", keys)
    time.sleep(0.05)
    return f"key {keys}"

  def type_text(self, text):
    self._xdo("type", "--clearmodifiers", text)
    time.sleep(0.05)
    return f"type {text!r}"

  def mousemove(self, x, y):
    self._xdo("mousemove", str(x), str(y))
    return f"move ({x},{y})"

  def screenshot(self, path: Path) -> float:
    t0 = time.monotonic()
    xwd = Path(tempfile.mktemp(suffix=".xwd"))
    try:
      subprocess.run(["xwd", "-silent", "-id", self.winid, "-out", str(xwd)],
                     env=self.env, check=True, timeout=FREEZE_TIMEOUT_S)
      subprocess.run(["convert", str(xwd), str(path)],
                     env=self.env, check=True, timeout=FREEZE_TIMEOUT_S)
    finally:
      xwd.unlink(missing_ok=True)
    return time.monotonic() - t0


def _rand_action(session: Session, rng: random.Random) -> str:
  xy = lambda: (rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1))
  kind = rng.choices(
    ["click", "dblclick", "drag", "scroll", "key", "type", "move"],
    weights=[35, 8, 15, 15, 15, 5, 7], k=1,
  )[0]
  if kind == "click":
    return session.click(*xy(), rng.choice([1, 1, 1, 1, 3]))
  if kind == "dblclick":
    return session.doubleclick(*xy())
  if kind == "drag":
    x1, y1 = xy()
    x2 = max(0, min(WIDTH - 1, x1 + rng.randint(-400, 400)))
    y2 = max(0, min(HEIGHT - 1, y1 + rng.randint(-400, 400)))
    return session.drag(x1, y1, x2, y2)
  if kind == "scroll":
    return session.scroll(*xy(), rng.choice([-5, -3, -1, 1, 3, 5]))
  if kind == "key":
    return session.key(rng.choice(KEYS))
  if kind == "type":
    return session.type_text("".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=rng.randint(1, 8))))
  return session.mousemove(*xy())


def run_fuzz(scenario: Scenario, round_id: int):
  rng = random.Random(round_id * 31 + hash(scenario.name))
  route = DEMO_ROUTE if rng.random() < 0.10 else f"{DEMO_ROUTE}/0/q"
  label = f"{scenario.name}-r{round_id:02d}-{'full' if '/' not in route[len(DEMO_ROUTE):] else 'quick'}"

  session = Session()
  try:
    session.start(route=route, cabana=scenario.cabana)
    print(f"\n[{label}] pid={session.jotp.pid} display={session.display} route={'full' if route == DEMO_ROUTE else 'quick'}")

    shots_dir = session.tmpdir / "screenshots"
    shots_dir.mkdir()
    log: list[str] = []
    crashes, freezes = [], []

    for i in range(1, FUZZ_ACTIONS + 1):
      # periodic screenshot = freeze check
      if i % SCREENSHOT_EVERY == 0:
        try:
          elapsed = session.screenshot(shots_dir / f"{i:04d}.png")
          log.append(f"[{i:03d}] screenshot ({elapsed:.2f}s)")
          if elapsed > FREEZE_TIMEOUT_S * 0.8:
            freezes.append(f"[{i:03d}] slow screenshot ({elapsed:.2f}s)")
        except subprocess.TimeoutExpired:
          freezes.append(f"[{i:03d}] screenshot timed out ({FREEZE_TIMEOUT_S}s)")
        except subprocess.CalledProcessError:
          pass

      # random action
      try:
        desc = _rand_action(session, rng)
      except subprocess.TimeoutExpired:
        freezes.append(f"[{i:03d}] xdotool timed out")
        continue
      except subprocess.CalledProcessError:
        continue
      log.append(f"[{i:03d}] {desc}")
      if i <= 5 or i % 50 == 0:
        print(f"  [{i:03d}/{FUZZ_ACTIONS}] {desc}")

      # crash check
      if not session.alive:
        reason = session.crash_reason()
        crashes.append(f"[{i:03d}] {reason}")
        print(f"  !! CRASH at action {i}: {reason}")
        print(f"  !! last: {desc}")
        print(f"  !! stderr:\n{session.stderr_tail()}")
        break

    # write log
    log_path = session.tmpdir / "fuzz.log"
    log_path.write_text("\n".join(log) + "\n")
    print(f"  [{label}] done: {len(log)} actions, {len(crashes)} crashes, {len(freezes)} freezes — {log_path}")

    parts = []
    if crashes:
      parts.append(f"CRASH: {crashes[0]}")
    if freezes:
      parts.append(f"FREEZE: {freezes[0]}")
    if parts:
      pytest.fail(f"[{label}] {' | '.join(parts)} (log: {log_path})")
  finally:
    session.stop()


def _make_fuzz_params():
  return [pytest.param(s, r, id=f"{s.name}-r{r:02d}")
          for s in SCENARIOS for r in range(ROUNDS_PER_SCENARIO)]


LAYOUTS_DIR = Path(__file__).resolve().parent / "layouts"

LAYOUT_CHECKS: dict[str, str] = {
  "CAN-bus-debug": "panes for CAN bus counters (may use derivative curves that appear flat with limited data)",
  "camera-timings": "multiple tabs with camera/model timing curves",
  "can-states": "CAN RX/TX/lost/bus-off counters across buses",
  "controls_mismatch_debug": "control state, lag, safety checks, or pedal state curves",
  "gps": "GPS-related panes (some may be empty if route lacks GPS data)",
  "gps_vs_llk": "custom python distance/speed curves (may show error dialog if GPS data is missing — that is acceptable)",
  "locationd_debug": "IMU sensor panes (some may be empty if route lacks sensor data — that is acceptable)",
  "longitudinal": "4 panes: accel actual vs planned, PID terms, speed, longActive",
  "max-torque-debug": "lateral accel or steering torque curves with custom python series",
  "system_lag_debug": "CPU usage, CPU temps, or lag curves",
  "thermal_debug": "thermal monitoring panes (some may be empty on non-TICI data — that is acceptable)",
  "torque-controller": "lateral control curves across multiple tabs",
  "tuning": "lateral/longitudinal tuning curves (may show custom series error dialog — that is acceptable)",
  "ublox-debug": "GPS/u-blox panes (some may be empty if route lacks GPS data — that is acceptable)",
  "new-layout": "this is an intentionally empty template layout — an empty plot pane is expected and correct",
}


def codex_review(image_paths: list[Path] | Path, prompt: str, timeout: float = 60.0) -> tuple[bool, str]:
  """Ask codex to review screenshot(s). Returns (passed, explanation)."""
  if isinstance(image_paths, Path):
    image_paths = [image_paths]
  cmd = ["codex", "exec", "-c", 'model_reasoning_effort="high"']
  for p in image_paths:
    cmd.extend(["-i", str(p)])
  cmd.append("-")  # read prompt from stdin
  result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
  answer = result.stdout.strip()
  if not answer:
    # codex may print the model answer in stderr after the banner
    for line in reversed(result.stderr.strip().splitlines()):
      stripped = line.strip()
      if stripped and stripped not in ("codex", "") and not stripped.startswith(("tokens", "OpenAI", "---", "workdir", "model", "provider", "approval", "sandbox", "reasoning", "session")):
        answer = stripped
        break
  passed = "PASS" in answer.upper().split("\n")[0] if answer else False
  if not answer:
    answer = f"(no response, rc={result.returncode}, stderr_tail={result.stderr[-300:]})"
  return passed, answer


def _crop_image(src: Path, dst: Path, box: str):
  """Crop an image using ImageMagick. box is WxH+X+Y."""
  subprocess.run(["convert", str(src), "-crop", box, str(dst)], check=True, timeout=10)


def _random_crops(src: Path, out_dir: Path, n: int, w: int, h: int, seed: int) -> list[Path]:
  """Generate n random crops from src for detail inspection."""
  rng = random.Random(seed)
  crop_w, crop_h = w // 2, h // 2
  paths = []
  for i in range(n):
    x = rng.randint(0, w - crop_w)
    y = rng.randint(0, h - crop_h)
    p = out_dir / f"crop_{i}.png"
    _crop_image(src, p, f"{crop_w}x{crop_h}+{x}+{y}")
    paths.append(p)
  return [p for p in paths if p.exists()]


LAYOUT_WIDTH, LAYOUT_HEIGHT = 1920, 1080
NUM_CROPS = 3


def run_layout_screenshot(layout: str):
  session = Session(width=LAYOUT_WIDTH, height=LAYOUT_HEIGHT)
  try:
    session.start(route=f"{DEMO_ROUTE}/0/q", layout=layout, sync_load=True)
    assert session.alive, f"jotpluggler crashed on startup: {session.crash_reason()}"

    shot = session.tmpdir / "screenshot.png"
    session.screenshot(shot)
    assert shot.stat().st_size > 1000, f"screenshot too small ({shot.stat().st_size}B)"

    crops = _random_crops(shot, session.tmpdir, NUM_CROPS, LAYOUT_WIDTH, LAYOUT_HEIGHT, seed=hash(layout))

    specific = LAYOUT_CHECKS.get(layout, "")
    crop_desc = " ".join(f"Image {i+2} is a random zoomed crop for detail inspection." for i in range(len(crops)))
    prompt_parts = [
      f"You are reviewing the '{layout}' layout in jotpluggler, a vehicle telemetry plotting tool.",
      f"Image 1 is the full screenshot. {crop_desc}",
      "This uses a short demo route so some panes may lack data — that's OK.",
      "Review and check the following. PASS only if the UI is structurally sound:",
      "- the app rendered without crashing (not a blank/black window)",
      "- plot panes are laid out correctly (not overlapping each other)",
      "- text labels and legends are readable (not garbled or clipped by other elements)",
      "- no UI elements overlapping other UI elements (e.g. buttons covering text, legends colliding with controls). in particular, check that close/dismiss buttons (X) have clear spacing from nearby text like legends — if they touch or crowd, that is a FAIL",
      "- no rendering artifacts or broken UI elements",
      "- the sidebar is visible on the left",
      "- error dialogs about missing data or custom series are acceptable — not a failure",
      "- some panes being empty due to limited demo data is acceptable — not a failure",
    ]
    if specific:
      prompt_parts.append(f"- layout-specific: {specific}")
    prompt_parts.append(
      "Reply with exactly PASS or FAIL on the first line, then a one-sentence explanation."
    )
    prompt = "\n".join(prompt_parts)

    passed, explanation = codex_review([shot, *crops], prompt)
    print(f"  [layout_{layout}] codex: {'PASS' if passed else 'FAIL'}")
    print(f"    {explanation}")
    assert passed, f"codex review failed for layout '{layout}': {explanation}"
  finally:
    session.stop()


def _layout_names():
  return sorted(p.stem for p in LAYOUTS_DIR.glob("*.json"))


@pytest.mark.skipif(not BINARY.exists(), reason="jotpluggler binary not built")
class TestJotplugglerFuzz:
  @pytest.mark.parametrize("scenario,round_id", _make_fuzz_params())
  def test_fuzz(self, scenario: Scenario, round_id: int):
    run_fuzz(scenario, round_id)


@pytest.mark.skipif(not BINARY.exists(), reason="jotpluggler binary not built")
class TestJotplugglerLayouts:
  @pytest.mark.parametrize("layout", _layout_names())
  def test_layout_screenshot(self, layout: str):
    run_layout_screenshot(layout)
