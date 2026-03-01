import curses
import signal
import threading
import time

from cereal import messaging

from openpilot.system.version import get_version
from openpilot.tools.replay.replay_pyx import (
  FindFlag, TimelineType, ReplyMsgType, formatted_data_size,
)

BORDER_SIZE = 3
SPEED_ARRAY = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]

KEYBOARD_SHORTCUTS = [
  [
    ("s", "+10s"),
    ("shift+s", "-10s"),
    ("m", "+60s"),
    ("shift+m", "-60s"),
    ("space", "Pause/Resume"),
    ("e", "Next Engagement"),
    ("d", "Next Disengagement"),
    ("t", "Next User Tag"),
    ("i", "Next Info"),
    ("w", "Next Warning"),
    ("c", "Next Critical"),
  ],
  [
    ("enter", "Enter seek request"),
    ("+/-", "Playback speed"),
    ("q", "Exit"),
  ],
]

# Color pair IDs (matching C++ enum)
class Color:
  Default = 0
  Debug = 1
  Yellow = 2
  Green = 3
  Red = 4
  Cyan = 5
  BrightWhite = 6
  Engaged = 7
  Disengaged = 8

# Window IDs
class Win:
  Title = 0
  Stats = 1
  Timeline = 2
  TimelineDesc = 3
  CarState = 4
  DownloadBar = 5
  LogBorder = 6
  Log = 7
  Help = 8
  Max = 9


def add_str(win, text, color=Color.Default, bold=False):
  if win is None:
    return
  attr = 0
  if color != Color.Default:
    attr |= curses.color_pair(color)
  if bold:
    attr |= curses.A_BOLD
  try:
    win.addstr(text, attr)
  except curses.error:
    pass


class ConsoleUI:
  def __init__(self, replay, stdscr):
    self.replay = replay
    self.stdscr = stdscr
    self.sm = messaging.SubMaster(["carState", "liveParameters"])
    self.paused = False
    self.max_height = 0
    self.max_width = 0
    self.wins = [None] * Win.Max

    # Thread-safe log and progress state
    self._lock = threading.Lock()
    self._logs = []
    self._progress_cur = 0
    self._progress_total = 0
    self._download_success = False

    # Set up curses
    curses.curs_set(0)
    curses.cbreak()
    curses.noecho()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    curses.start_color()
    # https://www.ditig.com/256-colors-cheat-sheet
    curses.init_pair(Color.Debug, 246, curses.COLOR_BLACK)        # #949494
    curses.init_pair(Color.Yellow, 184, curses.COLOR_BLACK)
    curses.init_pair(Color.Red, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(Color.Cyan, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(Color.BrightWhite, 15, curses.COLOR_BLACK)
    curses.init_pair(Color.Disengaged, curses.COLOR_BLUE, curses.COLOR_BLUE)
    curses.init_pair(Color.Engaged, 28, 28)
    curses.init_pair(Color.Green, 34, curses.COLOR_BLACK)

    self._init_windows()

    # Install callbacks
    replay.install_message_handler(self._on_message)
    replay.install_download_progress_handler(self._on_progress)

  def _on_message(self, msg_type, msg):
    with self._lock:
      self._logs.append((msg_type, msg))

  def _on_progress(self, cur, total, success):
    with self._lock:
      self._progress_cur = cur
      self._progress_total = total
      self._download_success = success

  def _init_windows(self):
    self.max_height, self.max_width = self.stdscr.getmaxyx()
    self.wins = [None] * Win.Max

    self.wins[Win.Title] = curses.newwin(1, self.max_width, 0, 0)
    self.wins[Win.Stats] = curses.newwin(2, self.max_width - 2 * BORDER_SIZE, 2, BORDER_SIZE)
    self.wins[Win.Timeline] = curses.newwin(4, self.max_width - 2 * BORDER_SIZE, 5, BORDER_SIZE)
    self.wins[Win.TimelineDesc] = curses.newwin(1, 100, 10, BORDER_SIZE)
    self.wins[Win.CarState] = curses.newwin(3, 100, 12, BORDER_SIZE)
    self.wins[Win.DownloadBar] = curses.newwin(1, 100, 16, BORDER_SIZE)

    log_height = self.max_height - 27
    if log_height > 4:
      self.wins[Win.LogBorder] = curses.newwin(log_height, self.max_width - 2 * (BORDER_SIZE - 1), 17, BORDER_SIZE - 1)
      self.wins[Win.LogBorder].box()
      self.wins[Win.Log] = curses.newwin(log_height - 2, self.max_width - 2 * BORDER_SIZE, 18, BORDER_SIZE)
      self.wins[Win.Log].scrollok(True)

    if self.max_height >= 23:
      self.wins[Win.Help] = curses.newwin(5, self.max_width - 2 * BORDER_SIZE, self.max_height - 6, BORDER_SIZE)
    elif self.max_height >= 17:
      self.wins[Win.Help] = curses.newwin(1, self.max_width - 2 * BORDER_SIZE, self.max_height - 1, BORDER_SIZE)
      try:
        self.wins[Win.Help].addstr(0, 0, "Expand screen vertically to list available commands")
      except curses.error:
        pass

    # Title bar
    self.wins[Win.Title].bkgd(' ', curses.A_REVERSE)
    try:
      self.wins[Win.Title].addstr(0, 3, f"openpilot replay {get_version()}")
    except curses.error:
      pass

    # Initial draw
    self.stdscr.refresh()
    self._display_timeline_desc()
    if self.max_height >= 23:
      self._display_help()
    self._update_summary()
    self._update_timeline()
    for win in self.wins:
      if win is not None:
        win.noutrefresh()
    curses.doupdate()

  def _update_size(self):
    if curses.is_term_resized(self.max_height, self.max_width):
      self.wins = [None] * Win.Max
      curses.endwin()
      self.stdscr.clear()
      self.stdscr.refresh()
      self._init_windows()

  def _update_status(self):
    win = self.wins[Win.CarState]
    if win is None:
      return

    self.sm.update(0)

    def write_item(y, x, key, value, unit, bold=False, color=Color.BrightWhite):
      try:
        win.move(y, x)
      except curses.error:
        return
      add_str(win, key)
      add_str(win, value, color, bold)
      add_str(win, unit)

    if self.paused:
      status_str, status_color = "paused...", Color.Yellow
    else:
      status_str, status_color = "playing", Color.Green

    write_item(0, 0, "STATUS:    ", status_str, "      ", False, status_color)

    cur_ts = self.replay.route_date_time() + int(self.replay.current_seconds())
    time_string = time.ctime(cur_ts)
    current_segment = " - " + str(int(self.replay.current_seconds() / 60))
    write_item(0, 25, "TIME:  ", time_string, current_segment, True)

    lp = self.sm["liveParameters"]
    cs = self.sm["carState"]
    write_item(1, 0, "STIFFNESS: ", f"{lp.stiffnessFactor * 100:.2f} %", "  ")
    write_item(1, 25, "SPEED: ", f"{cs.vEgo:.2f}", " m/s")
    write_item(2, 0, "STEER RATIO: ", f"{lp.steerRatio:.2f}", "")
    angle_offsets = f"{lp.angleOffsetAverageDeg:.2f}|{lp.angleOffsetDeg:.2f}"
    write_item(2, 25, "ANGLE OFFSET(AVG|INSTANT): ", angle_offsets, " deg")

    win.noutrefresh()

  def _display_help(self):
    win = self.wins[Win.Help]
    if win is None:
      return
    for i, shortcuts in enumerate(KEYBOARD_SHORTCUTS):
      try:
        win.move(i * 2, 0)
      except curses.error:
        continue
      for key, desc in shortcuts:
        try:
          win.addstr(f" {key} ", curses.A_REVERSE)
          win.addstr(f" {desc} ")
        except curses.error:
          break
    win.noutrefresh()

  def _display_timeline_desc(self):
    win = self.wins[Win.TimelineDesc]
    if win is None:
      return
    indicators = [
      (Color.Engaged, " Engaged ", False),
      (Color.Disengaged, " Disengaged ", False),
      (Color.Green, " Info ", True),
      (Color.Yellow, " Warning ", True),
      (Color.Red, " Critical ", True),
      (Color.Cyan, " User Tag ", True),
    ]
    for color, name, bold in indicators:
      add_str(win, "__", color, bold)
      add_str(win, name)
    win.noutrefresh()

  def _log_message(self, msg_type, msg):
    win = self.wins[Win.Log]
    if win is None:
      return
    color = Color.Default
    if msg_type == int(ReplyMsgType.Debug):
      color = Color.Debug
    elif msg_type == int(ReplyMsgType.Warning):
      color = Color.Yellow
    elif msg_type == int(ReplyMsgType.Critical):
      color = Color.Red
    add_str(win, msg + "\n", color)
    win.noutrefresh()

  def _update_progress_bar(self):
    win = self.wins[Win.DownloadBar]
    if win is None:
      return
    win.erase()
    if self._download_success and self._progress_cur < self._progress_total:
      width = 35
      progress = self._progress_cur / self._progress_total
      pos = int(width * progress)
      bar = "=" * pos + ">" + " " * (width - pos)
      text = f"Downloading [{bar}]  {int(progress * 100)}% {formatted_data_size(self._progress_total)}"
      try:
        win.addstr(0, 0, text)
      except curses.error:
        pass
    win.noutrefresh()

  def _update_summary(self):
    win = self.wins[Win.Stats]
    if win is None:
      return
    try:
      win.addstr(0, 0, f"Route: {self.replay.route_name()}, {self.replay.segment_count()} segments")
      win.addstr(1, 0, f"Car Fingerprint: {self.replay.car_fingerprint()}")
    except curses.error:
      pass
    win.noutrefresh()

  def _update_timeline(self):
    win = self.wins[Win.Timeline]
    if win is None:
      return
    width = win.getmaxyx()[1]
    win.erase()

    # Draw disengaged background
    try:
      win.attron(curses.color_pair(Color.Disengaged))
      for row in (1, 2):
        win.move(row, 0)
        win.addstr(" " * (width - 1))
      win.attroff(curses.color_pair(Color.Disengaged))
    except curses.error:
      pass

    total_sec = self.replay.max_seconds() - self.replay.min_seconds()
    if total_sec <= 0:
      win.noutrefresh()
      return

    min_sec = self.replay.min_seconds()

    for start_time, end_time, entry_type in self.replay.get_timeline():
      start_pos = int(((start_time - min_sec) / total_sec) * width)
      end_pos = int(((end_time - min_sec) / total_sec) * width)
      start_pos = max(0, min(start_pos, width - 1))
      end_pos = max(0, min(end_pos, width - 1))
      n = end_pos - start_pos + 1
      if n <= 0:
        continue

      try:
        if entry_type == int(TimelineType.Engaged):
          win.chgat(1, start_pos, n, curses.color_pair(Color.Engaged))
          win.chgat(2, start_pos, n, curses.color_pair(Color.Engaged))
        elif entry_type == int(TimelineType.UserBookmark):
          win.chgat(3, start_pos, n, curses.color_pair(Color.Cyan))
        else:
          if entry_type == int(TimelineType.AlertInfo):
            color_id = Color.Green
          elif entry_type == int(TimelineType.AlertWarning):
            color_id = Color.Yellow
          else:
            color_id = Color.Red
          win.chgat(3, start_pos, n, curses.color_pair(color_id))
      except curses.error:
        pass

    # Current position indicator
    cur_pos = int(((self.replay.current_seconds() - min_sec) / total_sec) * width)
    cur_pos = max(0, min(cur_pos, width - 1))
    try:
      win.attron(curses.color_pair(Color.BrightWhite))
      win.addch(0, cur_pos, curses.ACS_VLINE)
      win.addch(3, cur_pos, curses.ACS_VLINE)
      win.attroff(curses.color_pair(Color.BrightWhite))
    except curses.error:
      pass

    win.noutrefresh()

  def _pause_replay(self, pause):
    self.replay.pause(pause)
    self.paused = pause

  def _handle_key(self, c):
    if c == ord('\n'):
      # Pause and enter blocking seek mode
      self._pause_replay(True)
      self._update_status()
      curses.doupdate()
      curses.curs_set(1)
      self.stdscr.nodelay(False)

      y = self.max_height - 9
      try:
        self.stdscr.move(y, BORDER_SIZE)
        add_str(self.stdscr, "Enter seek request: ", Color.BrightWhite, True)
        self.stdscr.refresh()
      except curses.error:
        pass

      curses.echo()
      try:
        input_str = self.stdscr.getstr(y, BORDER_SIZE + 20, 10)
        choice = int(input_str)
      except (ValueError, curses.error):
        choice = 0
      curses.noecho()

      self._pause_replay(False)
      self.replay.seek_to(choice, False)

      try:
        self.stdscr.move(y, 0)
        self.stdscr.clrtoeol()
      except curses.error:
        pass
      self.stdscr.nodelay(True)
      curses.curs_set(0)
      self.stdscr.refresh()

    elif c in (ord('+'), ord('=')):
      cur_speed = self.replay.get_speed()
      for s in SPEED_ARRAY:
        if s > cur_speed:
          self.replay.set_speed(s)
          break

    elif c in (ord('_'), ord('-')):
      cur_speed = self.replay.get_speed()
      prev = None
      for s in SPEED_ARRAY:
        if s >= cur_speed:
          break
        prev = s
      if prev is not None:
        self.replay.set_speed(prev)

    elif c == ord('e'):
      self.replay.seek_to_flag(FindFlag.nextEngagement)
    elif c == ord('d'):
      self.replay.seek_to_flag(FindFlag.nextDisEngagement)
    elif c == ord('t'):
      self.replay.seek_to_flag(FindFlag.nextUserBookmark)
    elif c == ord('i'):
      self.replay.seek_to_flag(FindFlag.nextInfo)
    elif c == ord('w'):
      self.replay.seek_to_flag(FindFlag.nextWarning)
    elif c == ord('c'):
      self.replay.seek_to_flag(FindFlag.nextCritical)
    elif c == ord('m'):
      self.replay.seek_to(60, True)
    elif c == ord('M'):
      self.replay.seek_to(-60, True)
    elif c == ord('s'):
      self.replay.seek_to(10, True)
    elif c == ord('S'):
      self.replay.seek_to(-10, True)
    elif c == ord(' '):
      self._pause_replay(not self.replay.is_paused())

  def exec(self):
    do_exit = threading.Event()
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda *_: do_exit.set())
    signal.signal(signal.SIGTERM, lambda *_: do_exit.set())

    try:
      frame = 0
      while not do_exit.is_set():
        c = self.stdscr.getch()
        if c in (ord('q'), ord('Q')):
          break
        if c != -1:
          self._handle_key(c)

        if frame % 25 == 0:
          self._update_size()
          self._update_summary()

        self._update_timeline()
        self._update_status()

        with self._lock:
          self._update_progress_bar()
          for msg_type, msg in self._logs:
            self._log_message(msg_type, msg)
          self._logs.clear()

        curses.doupdate()
        frame += 1
        time.sleep(0.05)
    finally:
      self.replay.install_download_progress_handler(None)
      self.replay.install_message_handler(None)
      signal.signal(signal.SIGINT, prev_sigint)
      signal.signal(signal.SIGTERM, prev_sigterm)

    return 0
