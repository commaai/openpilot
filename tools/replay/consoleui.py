#!/usr/bin/env python3
import curses
import signal
import threading
import time
from enum import Enum, auto
from typing import Optional

import cereal.messaging as messaging

from openpilot.tools.replay.replay import Replay
from openpilot.tools.replay.seg_mgr import ReplayFlags
from openpilot.tools.replay.timeline import FindFlag, TimelineType

BORDER_SIZE = 3

KEYBOARD_SHORTCUTS = [
  [
    ("s", "+10s"),
    ("S", "-10s"),
    ("m", "+60s"),
    ("M", "-60s"),
    ("space", "Pause/Resume"),
    ("e", "Next Engagement"),
    ("d", "Next Disengagement"),
    ("t", "Next User Tag"),
  ],
  [
    ("i", "Next Info"),
    ("w", "Next Warning"),
    ("c", "Next Critical"),
    ("enter", "Enter seek request"),
    ("+/-", "Playback speed"),
    ("q", "Exit"),
  ],
]


class Color(Enum):
  DEFAULT = 0
  DEBUG = 1
  YELLOW = 2
  GREEN = 3
  RED = 4
  CYAN = 5
  BRIGHT_WHITE = 6
  ENGAGED = 7
  DISENGAGED = 8


class Status(Enum):
  PLAYING = auto()
  PAUSED = auto()


class Win(Enum):
  TITLE = 0
  STATS = 1
  LOG = 2
  LOG_BORDER = 3
  DOWNLOAD_BAR = 4
  TIMELINE = 5
  TIMELINE_DESC = 6
  HELP = 7
  CAR_STATE = 8


SPEED_ARRAY = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


class ConsoleUI:
  def __init__(self, replay: Replay):
    self.replay = replay
    self.status = Status.PLAYING
    self.windows: dict[Win, Optional[curses.window]] = {w: None for w in Win}
    self.max_width = 0
    self.max_height = 0

    self._lock = threading.Lock()
    self._logs: list[tuple[int, str]] = []
    self._progress_cur = 0
    self._progress_total = 0
    self._download_success = False
    self._exit = False

    self._sm = messaging.SubMaster(["carState", "liveParameters"])

    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, lambda s, f: setattr(self, '_exit', True))

  def _init_curses(self, stdscr) -> None:
    self._stdscr = stdscr
    curses.curs_set(0)
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    # Initialize colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(Color.DEBUG.value, 246, -1)
    curses.init_pair(Color.YELLOW.value, curses.COLOR_YELLOW, -1)
    curses.init_pair(Color.GREEN.value, curses.COLOR_GREEN, -1)
    curses.init_pair(Color.RED.value, curses.COLOR_RED, -1)
    curses.init_pair(Color.CYAN.value, curses.COLOR_CYAN, -1)
    curses.init_pair(Color.BRIGHT_WHITE.value, curses.COLOR_WHITE, -1)
    curses.init_pair(Color.ENGAGED.value, curses.COLOR_GREEN, curses.COLOR_GREEN)
    curses.init_pair(Color.DISENGAGED.value, curses.COLOR_BLUE, curses.COLOR_BLUE)

    self._init_windows()

  def _init_windows(self) -> None:
    self.max_height, self.max_width = self._stdscr.getmaxyx()

    # Title bar
    self.windows[Win.TITLE] = curses.newwin(1, self.max_width, 0, 0)
    self.windows[Win.TITLE].bkgd(' ', curses.A_REVERSE)
    self.windows[Win.TITLE].addstr(0, 3, "openpilot replay (Python)")

    # Stats
    self.windows[Win.STATS] = curses.newwin(2, self.max_width - 2 * BORDER_SIZE, 2, BORDER_SIZE)

    # Timeline
    self.windows[Win.TIMELINE] = curses.newwin(4, self.max_width - 2 * BORDER_SIZE, 5, BORDER_SIZE)

    # Timeline description
    self.windows[Win.TIMELINE_DESC] = curses.newwin(1, 100, 10, BORDER_SIZE)

    # Car state
    self.windows[Win.CAR_STATE] = curses.newwin(3, 100, 12, BORDER_SIZE)

    # Download bar
    self.windows[Win.DOWNLOAD_BAR] = curses.newwin(1, 100, 16, BORDER_SIZE)

    # Log window
    log_height = self.max_height - 27
    if log_height > 4:
      self.windows[Win.LOG_BORDER] = curses.newwin(log_height, self.max_width - 2 * (BORDER_SIZE - 1), 17, BORDER_SIZE - 1)
      self.windows[Win.LOG_BORDER].box()
      self.windows[Win.LOG] = curses.newwin(log_height - 2, self.max_width - 2 * BORDER_SIZE, 18, BORDER_SIZE)
      self.windows[Win.LOG].scrollok(True)

    # Help window
    if self.max_height >= 23:
      self.windows[Win.HELP] = curses.newwin(5, self.max_width - 2 * BORDER_SIZE, self.max_height - 6, BORDER_SIZE)
    elif self.max_height >= 17:
      self.windows[Win.HELP] = curses.newwin(1, self.max_width - 2 * BORDER_SIZE, self.max_height - 1, BORDER_SIZE)
      self.windows[Win.HELP].addstr(0, 0, "Expand screen vertically to list available commands")

    self._stdscr.refresh()
    self._display_timeline_desc()
    if self.max_height >= 23:
      self._display_help()
    self._update_summary()
    self._update_timeline()

    for win in self.windows.values():
      if win:
        win.noutrefresh()
    curses.doupdate()

  def _add_str(self, win, text: str, color: Color = Color.DEFAULT, bold: bool = False) -> None:
    attrs = 0
    if color != Color.DEFAULT:
      attrs |= curses.color_pair(color.value)
    if bold:
      attrs |= curses.A_BOLD
    try:
      win.addstr(text, attrs)
    except curses.error:
      pass  # Ignore write errors at edge of window

  def _display_help(self) -> None:
    win = self.windows[Win.HELP]
    if not win:
      return

    for i, row in enumerate(KEYBOARD_SHORTCUTS):
      win.move(i * 2, 0)
      for key, desc in row:
        win.attron(curses.A_REVERSE)
        win.addstr(f" {key} ")
        win.attroff(curses.A_REVERSE)
        win.addstr(f" {desc} ")
    win.refresh()

  def _display_timeline_desc(self) -> None:
    win = self.windows[Win.TIMELINE_DESC]
    if not win:
      return

    indicators = [
      (Color.ENGAGED, " Engaged ", False),
      (Color.DISENGAGED, " Disengaged ", False),
      (Color.GREEN, " Info ", True),
      (Color.YELLOW, " Warning ", True),
      (Color.RED, " Critical ", True),
      (Color.CYAN, " User Tag ", True),
    ]

    for color, name, bold in indicators:
      self._add_str(win, "__", color, bold)
      self._add_str(win, name)
    win.refresh()

  def _update_summary(self) -> None:
    win = self.windows[Win.STATS]
    if not win:
      return

    route = self.replay.route
    if route:
      segments = route.segments
      win.addstr(0, 0, f"Route: {self.replay._seg_mgr._route_name}, {len(segments)} segments")
      win.addstr(1, 0, f"Car Fingerprint: {self.replay.car_fingerprint}")
    win.refresh()

  def _update_status(self) -> None:
    win = self.windows[Win.CAR_STATE]
    if not win:
      return

    win.erase()

    self._sm.update(0)

    # Status
    status_text = "playing" if self.status == Status.PLAYING else "paused..."
    status_color = Color.GREEN if self.status == Status.PLAYING else Color.YELLOW
    win.addstr(0, 0, "STATUS:    ")
    self._add_str(win, status_text, status_color)
    win.addstr("      ")

    # Time
    cur_ts = self.replay.current_seconds
    segment = int(cur_ts / 60)
    win.addstr(0, 25, "TIME:  ")
    self._add_str(win, f"{cur_ts:.1f}s", Color.BRIGHT_WHITE, True)
    win.addstr(f" - segment {segment}")

    # Speed
    win.addstr(1, 0, "SPEED: ")
    try:
      v_ego = self._sm["carState"].vEgo
      self._add_str(win, f"{v_ego:.2f}", Color.BRIGHT_WHITE)
    except Exception:
      self._add_str(win, "N/A", Color.YELLOW)
    win.addstr(" m/s")

    # Playback speed
    win.addstr(1, 25, "PLAYBACK: ")
    self._add_str(win, f"{self.replay.speed:.1f}x", Color.BRIGHT_WHITE, True)

    win.refresh()

  def _update_timeline(self) -> None:
    win = self.windows[Win.TIMELINE]
    if not win:
      return

    width = self.max_width - 2 * BORDER_SIZE
    win.erase()

    # Draw disengaged background
    win.attron(curses.color_pair(Color.DISENGAGED.value))
    for row in [1, 2]:
      win.move(row, 0)
      win.addstr(" " * (width - 1))
    win.attroff(curses.color_pair(Color.DISENGAGED.value))

    total_sec = self.replay.max_seconds - self.replay.min_seconds
    if total_sec <= 0:
      win.refresh()
      return

    # Draw timeline entries
    entries = self.replay.get_timeline()
    for entry in entries:
      start_pos = int((entry.start_time - self.replay.min_seconds) / total_sec * width)
      end_pos = int((entry.end_time - self.replay.min_seconds) / total_sec * width)
      start_pos = max(0, min(start_pos, width - 1))
      end_pos = max(0, min(end_pos, width - 1))

      if entry.type == TimelineType.ENGAGED:
        for row in [1, 2]:
          win.chgat(row, start_pos, end_pos - start_pos + 1, curses.color_pair(Color.ENGAGED.value))
      elif entry.type == TimelineType.USER_BOOKMARK:
        win.chgat(3, start_pos, end_pos - start_pos + 1, curses.color_pair(Color.CYAN.value))
      else:
        color = Color.GREEN
        if entry.type == TimelineType.ALERT_WARNING:
          color = Color.YELLOW
        elif entry.type == TimelineType.ALERT_CRITICAL:
          color = Color.RED
        try:
          win.chgat(3, start_pos, end_pos - start_pos + 1, curses.color_pair(color.value))
        except curses.error:
          pass

    # Draw current position
    cur_pos = int((self.replay.current_seconds - self.replay.min_seconds) / total_sec * width)
    cur_pos = max(0, min(cur_pos, width - 2))
    try:
      win.attron(curses.color_pair(Color.BRIGHT_WHITE.value))
      win.addch(0, cur_pos, curses.ACS_VLINE)
      win.addch(3, cur_pos, curses.ACS_VLINE)
      win.attroff(curses.color_pair(Color.BRIGHT_WHITE.value))
    except curses.error:
      pass

    win.refresh()

  def _update_progress_bar(self) -> None:
    win = self.windows[Win.DOWNLOAD_BAR]
    if not win:
      return

    win.erase()
    with self._lock:
      if self._download_success and self._progress_cur < self._progress_total:
        width = 35
        progress = self._progress_cur / self._progress_total
        pos = int(width * progress)
        bar = "=" * pos + ">" + " " * (width - pos)
        win.addstr(0, 0, f"Downloading [{bar}] {int(progress * 100)}%")
    win.refresh()

  def _log_message(self, msg: str, color: Color = Color.DEFAULT) -> None:
    win = self.windows[Win.LOG]
    if win:
      self._add_str(win, msg + "\n", color)
      win.refresh()

  def _pause_replay(self, pause: bool) -> None:
    self.replay.pause(pause)
    self.status = Status.PAUSED if pause else Status.PLAYING

  def _handle_key(self, key: int) -> bool:
    if key == ord('q') or key == ord('Q'):
      return False

    if key == ord('\n'):
      # Pause and get seek input
      self._pause_replay(True)
      self._update_status()
      curses.curs_set(1)
      self._stdscr.nodelay(False)

      self._log_message("Waiting for input...", Color.YELLOW)
      y = self.max_height - 9
      self._stdscr.move(y, BORDER_SIZE)
      self._add_str(self._stdscr, "Enter seek request (seconds): ", Color.BRIGHT_WHITE, True)
      self._stdscr.refresh()

      curses.echo()
      try:
        input_str = self._stdscr.getstr(y, BORDER_SIZE + 30, 10).decode('utf-8')
        choice = int(input_str)
        self._pause_replay(False)
        self.replay.seek_to(choice, relative=False)
      except (ValueError, curses.error):
        pass
      curses.noecho()

      self._stdscr.move(y, 0)
      self._stdscr.clrtoeol()
      self._stdscr.nodelay(True)
      curses.curs_set(0)
      self._stdscr.refresh()

    elif key == ord('+') or key == ord('='):
      speed = self.replay.speed
      for s in SPEED_ARRAY:
        if s > speed:
          self._log_message(f"playback speed: {s:.1f}x", Color.YELLOW)
          self.replay.speed = s
          break

    elif key == ord('-') or key == ord('_'):
      speed = self.replay.speed
      for s in reversed(SPEED_ARRAY):
        if s < speed:
          self._log_message(f"playback speed: {s:.1f}x", Color.YELLOW)
          self.replay.speed = s
          break

    elif key == ord('e'):
      self.replay.seek_to_flag(FindFlag.NEXT_ENGAGEMENT)
    elif key == ord('d'):
      self.replay.seek_to_flag(FindFlag.NEXT_DISENGAGEMENT)
    elif key == ord('t'):
      self.replay.seek_to_flag(FindFlag.NEXT_USER_BOOKMARK)
    elif key == ord('i'):
      self.replay.seek_to_flag(FindFlag.NEXT_INFO)
    elif key == ord('w'):
      self.replay.seek_to_flag(FindFlag.NEXT_WARNING)
    elif key == ord('c'):
      self.replay.seek_to_flag(FindFlag.NEXT_CRITICAL)
    elif key == ord('m'):
      self.replay.seek_to(60, relative=True)
    elif key == ord('M'):
      self.replay.seek_to(-60, relative=True)
    elif key == ord('s'):
      self.replay.seek_to(10, relative=True)
    elif key == ord('S'):
      self.replay.seek_to(-10, relative=True)
    elif key == ord(' '):
      self._pause_replay(not self.replay.is_paused)

    return True

  def _main_loop(self, stdscr) -> int:
    self._init_curses(stdscr)

    frame = 0
    while not self._exit:
      key = stdscr.getch()
      if not self._handle_key(key):
        break

      if frame % 25 == 0:
        # Check for terminal resize
        new_h, new_w = stdscr.getmaxyx()
        if new_h != self.max_height or new_w != self.max_width:
          for win in self.windows.values():
            if win:
              try:
                del win
              except Exception:
                pass
          stdscr.clear()
          stdscr.refresh()
          self._init_windows()

        self._update_summary()

      self._update_timeline()
      self._update_status()
      self._update_progress_bar()

      # Process logs
      with self._lock:
        for msg_type, msg in self._logs:
          color = Color.DEFAULT
          if msg_type == 0:  # Debug
            color = Color.DEBUG
          elif msg_type == 1:  # Warning
            color = Color.YELLOW
          elif msg_type == 2:  # Critical
            color = Color.RED
          self._log_message(msg, color)
        self._logs.clear()

      frame += 1
      time.sleep(0.05)  # ~20 Hz

    return 0

  def exec(self) -> int:
    return curses.wrapper(self._main_loop)
