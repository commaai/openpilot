"""
DashC screen: a Vantrue N5S-style dash cam view for the comma 3X (front camera only on screen).

Place at: openpilot/selfdrive/ui/onroad/dashcam_view.py

Full-screen live front camera with:
  - top left:   blinking REC dot + elapsed drive time (REC only when loggerd + encoderd are running)
  - top center: date/time stamp, like the N5S burns into its video
  - top right:  Google Drive upload state and free storage
  - bottom left:  driver attention icon (dim = fine, yellow/orange/red = warning level)
  - bottom right: small pulsing dot, only while a file is uploading (no text)
All three cameras are still recorded; this only changes what is shown on the screen.
Tap the screen to open the normal sidebar (settings).
"""
import os
import time
from datetime import datetime

import raylib_nb as rl

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.loggerd.config import get_available_bytes
from openpilot.system.loggerd.uploader import listdir_by_creation
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

TOKEN_PATH = "/data/dashcam/google_token.json"
ACTIVE_PATH = "/tmp/dashcam_uploading"
VIDEO_FILES = ("fcamera.hevc", "ecamera.hevc", "dcamera.hevc")
GDRIVE_ATTR_NAME = "user.gdrive"
GDRIVE_ATTR_VALUE = b"1"
GDRIVE_ERROR_ATTR = "user.gdrive_error"

WHITE = rl.Color(255, 255, 255, 255)
SOFT_WHITE = rl.Color(255, 255, 255, 200)
RED = rl.Color(235, 40, 40, 255)
GREEN = rl.Color(90, 220, 130, 255)
AMBER = rl.Color(255, 190, 60, 255)
DIM = rl.Color(255, 255, 255, 70)
YELLOW = rl.Color(255, 210, 60, 255)
ORANGE = rl.Color(255, 140, 30, 255)
BAR_DARK = rl.Color(0, 0, 0, 150)
BAR_CLEAR = rl.Color(0, 0, 0, 0)

BAR_HEIGHT = 150
MARGIN = 50


class DashcamView(CameraView):
  def __init__(self):
    super().__init__("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD)
    self._set_placeholder_color(rl.Color(0, 0, 0, 255))

    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)

    self._drive_start: float | None = None
    self._free_gb = 0.0
    self._last_storage_check = 0.0
    self._last_upload_check = 0.0
    self._upload_state = ("DRIVE", SOFT_WHITE)
    self._signin_rect: rl.Rectangle | None = None

  # ---- state -------------------------------------------------------------
  def _is_recording(self) -> bool:
    try:
      running = {p.name for p in ui_state.sm["managerState"].processes if p.running}
      return "loggerd" in running and "encoderd" in running
    except Exception:
      return False

  def _refresh_stats(self):
    now = time.monotonic()
    if now - self._last_storage_check > 5:
      self._last_storage_check = now
      self._free_gb = get_available_bytes(default=0) / 1e9

  def _gps_display(self):
    """Coordinates from the freshest GPS fix, or None while there is no fix."""
    source = latest_gps_source(ui_state.sm)
    if source is None:
      return None
    gps = ui_state.sm[source]
    lat_dir = "N" if gps.latitude >= 0 else "S"
    lon_dir = "E" if gps.longitude >= 0 else "W"
    return f"{abs(gps.latitude):.4f}°{lat_dir} {abs(gps.longitude):.4f}°{lon_dir}"

  def _upload_status(self):
    """Top-right Drive status: sign-in, active upload, or the newest finished segment's state."""
    if not os.path.exists(TOKEN_PATH):
      return "DRIVE: SIGN IN", AMBER
    if os.path.exists(ACTIVE_PATH):
      return "UPLOADING…", GREEN
    return self._scan_upload_status()

  def _scan_upload_status(self):
    """Read upload xattrs of the newest finished segment; throttled to 5 s."""
    now = time.monotonic()
    if now - self._last_upload_check <= 5:
      return self._upload_state
    self._last_upload_check = now

    root = Paths.log_root()
    for d in reversed(listdir_by_creation(root)):
      path = os.path.join(root, d)
      try:
        names = os.listdir(path)
      except OSError:
        continue
      if any(n.endswith(".lock") for n in names):
        continue  # still being recorded

      states = []
      for name in VIDEO_FILES:
        if name not in names:
          continue
        fn = os.path.join(path, name)
        try:
          done = os.getxattr(fn, GDRIVE_ATTR_NAME) == GDRIVE_ATTR_VALUE
          failed = os.getxattr(fn, GDRIVE_ERROR_ATTR) == GDRIVE_ATTR_VALUE
        except OSError:
          done = failed = False
        states.append((done, failed))

      if not states:
        continue
      if any(failed for _done, failed in states):
        self._upload_state = ("UPLOAD FAILED", RED)
      elif len(states) == len(VIDEO_FILES) and all(done for done, _failed in states):
        self._upload_state = ("DRIVE ✓", GREEN)
      else:
        ds = ui_state.sm["deviceState"]
        allow_metered = ui_state.params.get_bool("DashcamAllowMetered")
        if ds.networkType == 0 or (ds.networkMetered and not allow_metered):
          self._upload_state = ("OFFLINE", AMBER)
        else:
          self._upload_state = ("QUEUED", SOFT_WHITE)
      return self._upload_state

    self._upload_state = ("DRIVE", SOFT_WHITE)
    return self._upload_state

  # ---- drawing -----------------------------------------------------------
  def _text(self, font, text, x, y, size, color):
    rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, color)

  def _text_width(self, font, text, size) -> float:
    return measure_text_cached(font, text, size).x

  def _render(self, rect: rl.Rectangle):
    if not ui_state.display_onroad:
      self._drive_start = None
      return
    if self._drive_start is None:
      self._drive_start = time.monotonic()

    self._refresh_stats()
    super()._render(rect)
    self._draw_top_bar(rect)
    self._draw_bottom_bar(rect)

  def _draw_top_bar(self, rect: rl.Rectangle):
    rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y), int(rect.width), BAR_HEIGHT, BAR_DARK, BAR_CLEAR)
    y = rect.y + 40

    # REC dot + elapsed time
    x = rect.x + MARGIN
    if self._is_recording():
      if int(time.monotonic() * 1.5) % 2 == 0:
        rl.draw_rectangle_rounded(rl.Rectangle(x, y + 8, 36, 36), 1.0, 12, RED)
      self._text(self._font_bold, "REC", x + 55, y, 56, WHITE)
      elapsed = int(time.monotonic() - (self._drive_start or time.monotonic()))
      self._text(self._font_medium, f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}",
                 x + 190, y + 6, 46, SOFT_WHITE)
    else:
      self._text(self._font_bold, "NOT RECORDING", x, y, 56, AMBER)

    # date / time stamp
    stamp = datetime.now().strftime("%Y/%m/%d  %H:%M:%S")
    w = self._text_width(self._font_bold, stamp, 56)
    self._text(self._font_bold, stamp, rect.x + (rect.width - w) / 2, y, 56, WHITE)

    # Drive status + storage
    drive_text, drive_color = self._upload_status()
    storage_text = f"{self._free_gb:.0f} GB FREE"
    dw = self._text_width(self._font_medium, drive_text, 42)
    sw = self._text_width(self._font_medium, storage_text, 42)
    right = rect.x + rect.width - MARGIN
    self._text(self._font_medium, drive_text, right - dw, y - 6, 42, drive_color)
    self._text(self._font_medium, storage_text, right - sw, y + 44, 42, SOFT_WHITE)
    self._signin_rect = None if drive_text != "DRIVE: SIGN IN" else rl.Rectangle(right - dw, y - 6, dw, 50)

  def open_signin_if_tapped(self) -> bool:
    """Allow sign-in without SSH by tapping the Drive status in the dashcam view."""
    if self._signin_rect is None or os.path.exists(TOKEN_PATH):
      return False
    for ev in gui_app.mouse_events:
      if ev.left_pressed and rl.check_collision_point_rec(rl.Vector2(ev.pos.x, ev.pos.y), self._signin_rect):
        from openpilot.selfdrive.ui.onroad.dashcam_signin import DashcamSignIn
        DashcamSignIn()
        return True
    return False

  def _draw_attention_icon(self, cx: float, cy: float):
    """Silent driver attention indicator: dim when fine, yellow/orange/red as the warning level rises."""
    try:
      dm = ui_state.sm["driverMonitoringState"]
      level = int(dm.alertLevel.raw)
      face = bool(dm.visionPolicyState.faceDetected)
    except Exception:
      level, face = 0, True

    if level >= 3:
      pulse = 0.5 + 0.5 * abs(((time.monotonic() * 3.0) % 2.0) - 1.0)
      color = rl.Color(235, 40, 40, int(150 + 105 * pulse))
    elif level == 2:
      color = ORANGE
    elif level == 1:
      color = YELLOW
    else:
      color = DIM if face else rl.Color(255, 190, 60, 140)  # amber-ish when no face is seen

    # simple head-and-shoulders silhouette
    rl.draw_circle(int(cx), int(cy - 14), 11, color)
    rl.draw_circle(int(cx), int(cy + 24), 22, color)

  def _draw_bottom_bar(self, rect: rl.Rectangle):
    top = rect.y + rect.height - BAR_HEIGHT
    rl.draw_rectangle_gradient_v(int(rect.x), int(top), int(rect.width), BAR_HEIGHT, BAR_CLEAR, BAR_DARK)

    # driver attention indicator (bottom left, icon only)
    self._draw_attention_icon(rect.x + MARGIN + 25, top + BAR_HEIGHT / 2 + 5)

    # GPS coordinates (bottom center), only while there is a live fix
    coords_text = self._gps_display()
    if coords_text is not None:
      cw = self._text_width(self._font_medium, coords_text, 34)
      self._text(self._font_medium, coords_text, rect.x + (rect.width - cw) / 2, top + 60, 34, SOFT_WHITE)

    # upload dot: only visible while a file is actually uploading
    if os.path.exists(ACTIVE_PATH):
      pulse = 0.5 + 0.5 * abs(((time.monotonic() * 0.8) % 2.0) - 1.0)
      dot = rl.Color(90, 220, 130, int(120 + 135 * pulse))
      rl.draw_circle(int(rect.x + rect.width - MARGIN - 14), int(top + BAR_HEIGHT / 2 + 20), 14, dot)
