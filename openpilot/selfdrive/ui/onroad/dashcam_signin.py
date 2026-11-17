"""On-device Google sign-in for DashC. Uses the OAuth device flow, so the comma only
needs to display a short code; approval happens in a browser on the user's phone.

Entering the flow: tap "DRIVE: SIGN IN" in the dashcam view.
Google Cloud setup (once): enable Drive API, create an OAuth client of type
"TVs and Limited Input devices", then type its client ID and secret here.
Only the narrow drive.file scope is requested.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import raylib_nb as rl

from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.button import Button
from openpilot.system.ui.widgets.keyboard import Keyboard

TOKEN_PATH = "/data/dashcam/google_token.json"
DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive.file"


class DeviceFlow:
  """Network worker. Never touches the UI thread except through plain state fields."""

  def __init__(self, client_id: str, client_secret: str):
    self.client_id = client_id
    self.client_secret = client_secret
    self.state = "starting"
    self.url = ""
    self.code = ""
    self.message = ""
    self.cancelled = threading.Event()
    self.thread = threading.Thread(target=self._run, daemon=True)

  def start(self):
    self.thread.start()

  def cancel(self):
    self.cancelled.set()

  @staticmethod
  def _post(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode())
    try:
      with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)
    except urllib.error.HTTPError as e:
      try:
        return json.load(e)
      except ValueError:
        return {"error": f"http_{e.code}"}

  def _run(self):
    try:
      dev = self._post(DEVICE_CODE_URL, {"client_id": self.client_id, "scope": SCOPE})
      if "device_code" not in dev:
        raise RuntimeError(dev.get("error", "could not start sign-in"))
      self.url = dev.get("verification_url", "")
      self.code = dev.get("user_code", "")
      self.state = "waiting"

      interval = max(1, int(dev.get("interval", 5)))
      deadline = time.monotonic() + int(dev.get("expires_in", 1800))
      while time.monotonic() < deadline and not self.cancelled.is_set():
        time.sleep(interval)
        tok = self._post(TOKEN_URL, {
          "client_id": self.client_id,
          "client_secret": self.client_secret,
          "device_code": dev["device_code"],
          "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        })
        if "refresh_token" in tok:
          os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
          tmp = TOKEN_PATH + ".tmp"
          with open(tmp, "w") as f:
            json.dump({"client_id": self.client_id, "client_secret": self.client_secret,
                       "refresh_token": tok["refresh_token"], "folder_name": "DashC"}, f, indent=2)
          os.replace(tmp, TOKEN_PATH)
          self.state = "done"
          cloudlog.info("dashcam_signin_success")
          return
        err = tok.get("error")
        if err == "slow_down":
          interval += 5
        elif err != "authorization_pending":
          raise RuntimeError(str(err))
      self.state = "error"
      self.message = "Code expired" if not self.cancelled.is_set() else "Cancelled"
    except Exception as e:
      self.state = "error"
      self.message = str(e)[:160]
      cloudlog.warning(f"dashcam_signin_failed: {e}")


class DashcamSignIn(Widget):
  """Modal overlay: two keyboard prompts, then the device code and status."""

  def __init__(self):
    super().__init__()
    self._keyboard = Keyboard(min_text_size=1, password_mode=True, show_password_toggle=True)
    self._flow: DeviceFlow | None = None
    self._client_id = ""
    self._font = gui_app.font(FontWeight.MEDIUM)
    self._bold = gui_app.font(FontWeight.BOLD)
    self._close = Button("Close", self._on_close)
    gui_app.push_widget(self)
    self._ask("Google Cloud Client ID", self._on_client_id)

  def _ask(self, title: str, callback):
    self._keyboard.reset()
    self._keyboard.set_title(title)
    self._keyboard.set_callback(callback)
    gui_app.push_widget(self._keyboard)

  def _on_client_id(self, result):
    if result != DialogResult.CONFIRM:
      self._on_close()
      return
    self._client_id = self._keyboard.text.strip()
    if not self._client_id:
      self._on_close()
      return
    self._ask("Google Cloud Client Secret", self._on_client_secret)

  def _on_client_secret(self, result):
    secret = self._keyboard.text.strip() if result == DialogResult.CONFIRM else ""
    if not secret:
      self._on_close()
      return
    self._flow = DeviceFlow(self._client_id, secret)
    self._flow.start()

  def _on_close(self):
    if self._flow is not None:
      self._flow.cancel()
    gui_app.pop_widget()

  @staticmethod
  def _text(font, value: str, x: float, y: float, size: int, color):
    rl.draw_text_ex(font, value, rl.Vector2(x, y), size, 0, color)

  def _render(self, rect: rl.Rectangle):
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(0, 0, 0, 235))
    flow = self._flow
    if flow is None:
      return

    x, y = rect.x + 120, rect.y + 100
    white = rl.Color(255, 255, 255, 255)
    self._text(self._bold, "Google Drive sign-in", x, y, 64, white)
    if flow.state == "waiting":
      self._text(self._font, f"Open this on your phone:  {flow.url}", x, y + 130, 44, white)
      self._text(self._bold, f"Code: {flow.code}", x, y + 210, 88, rl.Color(90, 220, 130, 255))
      self._text(self._font, "Waiting for approval...", x, y + 330, 44, white)
    elif flow.state == "done":
      self._text(self._font, "Signed in. Recordings will upload on Wi-Fi.", x, y + 150, 48, rl.Color(90, 220, 130, 255))
    elif flow.state == "error":
      self._text(self._font, f"Sign-in failed: {flow.message}", x, y + 150, 44, rl.Color(255, 160, 60, 255))
    else:
      self._text(self._font, "Starting...", x, y + 150, 48, white)

    self._close.set_rect(rl.Rectangle(x, y + 430, 420, 110))
    self._close.render()
