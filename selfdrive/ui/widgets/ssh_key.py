import pyray as rl
import requests
import threading
from collections.abc import Callable
from enum import Enum

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.confirm_dialog import alert_dialog
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.list_view import (
  ItemAction,
  ListItem,
  BUTTON_HEIGHT,
  BUTTON_BORDER_RADIUS,
  BUTTON_FONT_SIZE,
  BUTTON_WIDTH,
)

VALUE_FONT_SIZE = 48


class SshKeyFetcher:
  HTTP_TIMEOUT = 15  # seconds

  def __init__(self, params: Params):
    self._params = params
    self._on_response: Callable[[str | None], None] | None = None
    self._done: bool = False
    self._error: str | None = None

  def fetch(self, username: str, on_response: Callable[[str | None], None]):
    self._error = None
    self._on_response = on_response
    threading.Thread(target=self._fetch_thread, args=(username,), daemon=True).start()

  def update(self):
    if not self._done:
      return
    self._done = False
    if self._error is not None:
      self.clear()
    if self._on_response:
      self._on_response(self._error)

  def clear(self):
    self._params.remove("GithubUsername")
    self._params.remove("GithubSshKeys")

  def _fetch_thread(self, username: str):
    try:
      response = requests.get(f"https://github.com/{username}.keys", timeout=self.HTTP_TIMEOUT)
      response.raise_for_status()
      keys = response.text.strip()
      if not keys:
        raise requests.exceptions.HTTPError("No SSH keys found")

      self._params.put("GithubUsername", username)
      self._params.put("GithubSshKeys", keys)
    except requests.exceptions.Timeout:
      self._error = tr("Request timed out")
    except Exception:
      self._error = tr("No SSH keys found for user '{}'").format(username)
    finally:
      self._done = True


class SshKeyActionState(Enum):
  LOADING = tr_noop("LOADING")
  ADD = tr_noop("ADD")
  REMOVE = tr_noop("REMOVE")


class SshKeyAction(ItemAction):
  MAX_WIDTH = 500

  def __init__(self):
    super().__init__(self.MAX_WIDTH, True)

    self._keyboard = Keyboard(min_text_size=1)
    self._params = Params()
    self._fetcher = SshKeyFetcher(self._params)
    self._text_font = gui_app.font(FontWeight.NORMAL)
    self._button = Button("", click_callback=self._handle_button_click, button_style=ButtonStyle.LIST_ACTION,
                          border_radius=BUTTON_BORDER_RADIUS, font_size=BUTTON_FONT_SIZE)

    self._refresh_state()

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    super().set_touch_valid_callback(touch_callback)
    self._button.set_touch_valid_callback(touch_callback)

  def _refresh_state(self):
    self._username = self._params.get("GithubUsername")
    self._state = SshKeyActionState.REMOVE if self._params.get("GithubSshKeys") else SshKeyActionState.ADD

  def _update_state(self):
    super()._update_state()
    self._fetcher.update()

  def _render(self, rect: rl.Rectangle) -> bool:
    # Draw username if exists
    if self._username:
      text_size = measure_text_cached(self._text_font, self._username, VALUE_FONT_SIZE)
      rl.draw_text_ex(
        self._text_font,
        self._username,
        (rect.x + rect.width - BUTTON_WIDTH - text_size.x - 30, rect.y + (rect.height - text_size.y) / 2),
        VALUE_FONT_SIZE,
        1.0,
        rl.Color(170, 170, 170, 255),
      )

    # Draw button
    button_rect = rl.Rectangle(rect.x + rect.width - BUTTON_WIDTH, rect.y + (rect.height - BUTTON_HEIGHT) / 2, BUTTON_WIDTH, BUTTON_HEIGHT)
    self._button.set_rect(button_rect)
    self._button.set_text(tr(self._state.value))
    self._button.set_enabled(self._state != SshKeyActionState.LOADING)
    self._button.render(button_rect)
    return False

  def _handle_button_click(self):
    if self._state == SshKeyActionState.ADD:
      self._keyboard.reset()
      self._keyboard.set_title(tr("Enter your GitHub username"))
      self._keyboard.set_callback(self._on_username_submit)
      gui_app.push_widget(self._keyboard)
    elif self._state == SshKeyActionState.REMOVE:
      self._fetcher.clear()
      self._refresh_state()

  def _on_username_submit(self, result: DialogResult):
    if result != DialogResult.CONFIRM:
      return

    username = self._keyboard.text.strip()
    if not username:
      return

    self._state = SshKeyActionState.LOADING
    self._fetcher.fetch(username, self._on_fetch_response)

  def _on_fetch_response(self, error: str | None):
    if error is None:
      self._state = SshKeyActionState.REMOVE
      self._username = self._params.get("GithubUsername")
    else:
      self._state = SshKeyActionState.ADD
      self._username = ""
      gui_app.push_widget(alert_dialog(error))


def ssh_key_item(title: str | Callable[[], str], description: str | Callable[[], str]) -> ListItem:
  return ListItem(title=title, description=description, action_item=SshKeyAction())
