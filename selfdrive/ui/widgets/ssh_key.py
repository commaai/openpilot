import pyray as rl
import requests
import threading
import copy
from enum import Enum

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.list_view import (
  ListItem,
  BUTTON_HEIGHT,
  BUTTON_FONT_SIZE,
  BUTTON_WIDTH,
)
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.widget import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import alert_dialog
from openpilot.system.ui.widgets.keyboard import Keyboard


class SshKeyState(Enum):
  LOADING = "LOADING"
  ADD = "ADD"
  REMOVE = "REMOVE"


class SshKeyItem(ListItem):
  HTTP_TIMEOUT = 15  # seconds
  MAX_WIDTH = 500

  def __init__(self, title: str, description: str):
    super().__init__(title, description=description)

    self._keyboard = Keyboard()
    self._params = Params()
    self._error_message: str = ""
    self._text_font = gui_app.font(FontWeight.MEDIUM)

    self._refresh_state()

  def get_action_width(self) -> int:
    return self.MAX_WIDTH

  def _refresh_state(self):
    self._username = self._params.get("GithubUsername", "")
    self._state = SshKeyState.REMOVE if self._params.get("GithubSshKeys") else SshKeyState.ADD

  def render_action(self, rect: rl.Rectangle) -> bool:
    # Show error dialog if there's an error
    if self._error_message:
      message = copy.copy(self._error_message)
      gui_app.set_modal_overlay(lambda: alert_dialog(message))
      self._username = ""
      self._error_message = ""

    # Draw username if exists
    if self._username:
      text_size = measure_text_cached(self._text_font, self._username, BUTTON_FONT_SIZE)
      rl.draw_text_ex(
        self._text_font,
        self._username,
        (rect.x + rect.width - BUTTON_WIDTH - text_size.x - 30, rect.y + (rect.height - text_size.y) / 2),
        BUTTON_FONT_SIZE,
        1.0,
        rl.WHITE,
      )

    # Draw button
    if gui_button(
      rl.Rectangle(
        rect.x + rect.width - BUTTON_WIDTH, rect.y + (rect.height - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT
      ),
      self._state.value,
      is_enabled=self._state != SshKeyState.LOADING,
      border_radius=BUTTON_HEIGHT // 2,
      font_size=BUTTON_FONT_SIZE,
      button_style=ButtonStyle.LIST_ACTION,
    ):
      self._handle_button_click()
      return True
    return False

  def _handle_button_click(self):
    if self._state == SshKeyState.ADD:
      self._keyboard.clear()
      self._keyboard.set_title("Enter your GitHub username")
      gui_app.set_modal_overlay(self._keyboard, callback=self._on_username_submit)
    elif self._state == SshKeyState.REMOVE:
      self._params.remove("GithubUsername")
      self._params.remove("GithubSshKeys")
      self._refresh_state()

  def _on_username_submit(self, result: DialogResult):
    if result != DialogResult.CONFIRM:
      return

    username = self._keyboard.text.strip()
    if not username:
      return

    self._state = SshKeyState.LOADING
    threading.Thread(target=lambda: self._fetch_ssh_key(username), daemon=True).start()

  def _fetch_ssh_key(self, username: str):
    try:
      url = f"https://github.com/{username}.keys"
      response = requests.get(url, timeout=self.HTTP_TIMEOUT)
      response.raise_for_status()
      keys = response.text.strip()
      if not keys:
        raise requests.exceptions.HTTPError("No SSH keys found")

      # Success - save keys
      self._params.put("GithubUsername", username)
      self._params.put("GithubSshKeys", keys)
      self._state = SshKeyState.REMOVE
      self._username = username

    except requests.exceptions.Timeout:
      self._error_message = "Request timed out"
      self._state = SshKeyState.ADD
    except Exception:
      self._error_message = f"No SSH keys found for user '{username}'"
      self._state = SshKeyState.ADD
