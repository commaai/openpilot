import curses
import signal
import threading
import time

from cereal import messaging

from openpilot.common.realtime import Ratekeeper
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
    ("enter", "Seek to seconds"),
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
    self.max_height = 0
    self.max_width = 0
    self.wins = [None] * Win.Max

    # Thread-safe log and progress state
    self._lock = threading.Lock()
    self._logs = []
    self._progress_cur = 0
    self._progress_total = 0
    self._download_success = False

    # Cached timeline data (refreshed every ~1.25s)
    self._timeline_cache = []
    self._cached_min_sec = 0.0
    self._cached_max_sec = 0.0

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

    self._version = get_version()

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

    w = self.max_width
    inner_w = max(1, w - 2 * BORDER_SIZE)
    stat_w = min(100, inner_w)

    self.wins[Win.Title] = curses.newwin(1, w, 0, 0)
    if self.max_height > 3 and inner_w > 0:
      self.wins[Win.Stats] = curses.newwin(2, inner_w, 2, BORDER_SIZE)
    if self.max_height > 8 and inner_w > 0:
      self.wins[Win.Timeline] = curses.newwin(4, inner_w, 5, BORDER_SIZE)
    if self.max_height > 10 and stat_w > 0:
      self.wins[Win.TimelineDesc] = curses.newwin(1, stat_w, 10, BORDER_SIZE)
    if self.max_height > 14 and stat_w > 0:
      self.wins[Win.CarState] = curses.newwin(3, stat_w, 12, BORDER_SIZE)
    if self.max_height > 16 and stat_w > 0:
      self.wins[Win.DownloadBar] = curses.newwin(1, stat_w, 16, BORDER_SIZE)

    log_height = self.max_height - 27
    if log_height > 4 and inner_w > 0:
      self.wins[Win.LogBorder] = curses.newwin(log_height, w - 2 * (BORDER_SIZE - 1), 17, BORDER_SIZE - 1)
      self.wins[Win.LogBorder].box()
      self.wins[Win.Log] = curses.newwin(log_height - 2, inner_w, 18, BORDER_SIZE)
      self.wins[Win.Log].scrollok(True)

    if self.max_height >= 23 and inner_w > 0:
      self.wins[Win.Help] = curses.newwin(5, inner_w, self.max_height - 6, BORDER_SIZE)
    elif self.max_height >= 17 and inner_w > 0:
      self.wins[Win.Help] = curses.newwin(1, inner_w, self.max_height - 1, BORDER_SIZE)
      try:
        self.wins[Win.Help].addstr(0, 0, "Expand screen vertically to list available commands")
      except curses.error:
        pass

    # Title bar
    self.wins[Win.Title].bkgd(' ', curses.A_REVERSE)
    try:
      title = f"openpilot replay {self._version}  |  {self.replay.route_name()}"
      self.wins[Win.Title].addstr(0, 3, title[:self.max_width - 4])
    except curses.error:
      pass

    # Initial draw
    self.stdscr.refresh()
    self._display_timeline_desc()
    if self.max_height >= 23:
      self._display_help()
    self._update_summary()
    self._update_timeline(self.replay.current_seconds())
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

  def _update_status(self, cur_sec):
    win = self.wins[Win.CarState]
    if win is None:
      return

    win.erase()
    self.sm.update(0)

    def write_item(y, x, key, value, unit, bold=False, color=Color.BrightWhite):
      try:
        win.move(y, x)
      except curses.error:
        return
      add_str(win, key)
      add_str(win, value, color, bold)
      add_str(win, unit)

    speed = self.replay.get_speed()
    if self.replay.is_paused():
      status_str, status_color = "paused...", Color.Yellow
    else:
      status_str, status_color = "playing", Color.Green

    speed_str = f"  {speed:.1f}x" if speed != 1.0 else ""
    write_item(0, 0, "STATUS:    ", status_str + speed_str, "      ", False, status_color)

    cur_ts = self.replay.route_date_time() + int(cur_sec)
    time_string = time.ctime(cur_ts)
    current_segment = " - " + str(int(cur_sec / 60))
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
    if msg_type == ReplyMsgType.Debug:
      color = Color.Debug
    elif msg_type == ReplyMsgType.Warning:
      color = Color.Yellow
    elif msg_type == ReplyMsgType.Critical:
      color = Color.Red
    add_str(win, msg + "\n", color)
    win.noutrefresh()

  def _update_progress_bar(self, cur, total, success):
    win = self.wins[Win.DownloadBar]
    if win is None:
      return
    win.erase()
    if success and cur < total:
      width = 35
      progress = cur / total
      pos = int(width * progress)
      bar = "=" * pos + ">" + " " * max(0, width - pos - 1)
      text = f"Downloading [{bar}]  {int(progress * 100)}% {formatted_data_size(total)}"
      try:
        win.addstr(0, 0, text)
      except curses.error:
        pass
    win.noutrefresh()

  def _update_summary(self):
    win = self.wins[Win.Stats]
    if win is None:
      return
    win.erase()
    try:
      win.addstr(0, 0, f"Route: {self.replay.route_name()}, {self.replay.segment_count()} segments")
      win.addstr(1, 0, f"Car Fingerprint: {self.replay.car_fingerprint()}")
    except curses.error:
      pass
    win.noutrefresh()

  def _update_timeline(self, cur_sec):
    win = self.wins[Win.Timeline]
    if win is None:
      return
    width = win.getmaxyx()[1]
    win.erase()

    # Draw disengaged background
    try:
      for row in (1, 2):
        win.hline(row, 0, ord(' ') | curses.color_pair(Color.Disengaged), width)
    except curses.error:
      pass

    min_sec = self._cached_min_sec
    total_sec = self._cached_max_sec - min_sec
    if total_sec <= 0:
      win.noutrefresh()
      return

    tl_engaged = TimelineType.Engaged
    tl_bookmark = TimelineType.UserBookmark
    tl_info = TimelineType.AlertInfo
    tl_warning = TimelineType.AlertWarning

    for start_time, end_time, entry_type in self._timeline_cache:
      start_pos = int(((start_time - min_sec) / total_sec) * width)
      end_pos = int(((end_time - min_sec) / total_sec) * width)
      start_pos = max(0, min(start_pos, width - 1))
      end_pos = max(0, min(end_pos, width - 1))
      n = end_pos - start_pos + 1
      if n <= 0:
        continue

      try:
        if entry_type == tl_engaged:
          win.chgat(1, start_pos, n, curses.color_pair(Color.Engaged))
          win.chgat(2, start_pos, n, curses.color_pair(Color.Engaged))
        elif entry_type == tl_bookmark:
          win.chgat(3, start_pos, n, curses.color_pair(Color.Cyan))
        else:
          if entry_type == tl_info:
            color_id = Color.Green
          elif entry_type == tl_warning:
            color_id = Color.Yellow
          else:
            color_id = Color.Red
          win.chgat(3, start_pos, n, curses.color_pair(color_id))
      except curses.error:
        pass

    # Current position indicator
    cur_pos = int(((cur_sec - min_sec) / total_sec) * width)
    cur_pos = max(0, min(cur_pos, width - 1))
    try:
      win.attron(curses.color_pair(Color.BrightWhite))
      win.addch(0, cur_pos, curses.ACS_VLINE)
      win.addch(3, cur_pos, curses.ACS_VLINE)
      win.attroff(curses.color_pair(Color.BrightWhite))
    except curses.error:
      pass

    win.noutrefresh()

  def _log_speed(self, speed):
    self._log_message(ReplyMsgType.Warning, f"playback speed: {speed:.1f}x")

  def _handle_key(self, c):
    if c == ord('\n'):
      if self.max_height < 10:
        return

      # Pause and enter blocking seek mode
      was_paused = self.replay.is_paused()
      self.replay.pause(True)
      self._update_status(self.replay.current_seconds())
      curses.doupdate()
      curses.curs_set(1)
      self.stdscr.nodelay(False)

      y = self.max_height - 9
      try:
        prompt = "Seek to (seconds): "
        try:
          self.stdscr.move(y, BORDER_SIZE)
          add_str(self.stdscr, prompt, Color.BrightWhite, True)
          self.stdscr.refresh()
        except curses.error:
          pass

        curses.echo()
        choice = None
        try:
          input_str = self.stdscr.getstr(y, BORDER_SIZE + len(prompt), 10)
          choice = float(input_str)
        except (ValueError, curses.error):
          pass
        curses.noecho()

        if choice is not None:
          self.replay.seek_to(choice, False)
        if not was_paused:
          self.replay.pause(False)
      finally:
        curses.noecho()
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
          self._log_speed(s)
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
        self._log_speed(prev)

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
      self.replay.pause(not self.replay.is_paused())

  def exec(self):
    do_exit = threading.Event()
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, lambda *_: do_exit.set())
    signal.signal(signal.SIGTERM, lambda *_: do_exit.set())

    try:
      rk = Ratekeeper(20, print_delay_threshold=None)
      while not do_exit.is_set():
        c = self.stdscr.getch()
        if c in (ord('q'), ord('Q')):
          break
        if c != -1 and c != curses.KEY_RESIZE:
          self._handle_key(c)

        self._update_size()

        if rk.frame % 25 == 0:
          self._update_summary()
          self._timeline_cache = self.replay.get_timeline()
          self._cached_min_sec = self.replay.min_seconds()
          self._cached_max_sec = self.replay.max_seconds()

        cur_sec = self.replay.current_seconds()
        self._update_timeline(cur_sec)
        self._update_status(cur_sec)

        with self._lock:
          self._logs, logs_snapshot = [], self._logs
          progress_cur = self._progress_cur
          progress_total = self._progress_total
          download_success = self._download_success

        self._update_progress_bar(progress_cur, progress_total, download_success)
        for msg_type, msg in logs_snapshot:
          self._log_message(msg_type, msg)

        curses.doupdate()
        rk.keep_time()
    finally:
      self.replay.install_download_progress_handler(None)
      self.replay.install_message_handler(None)
      signal.signal(signal.SIGINT, prev_sigint)
      signal.signal(signal.SIGTERM, prev_sigterm)

    return 0
