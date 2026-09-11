import unittest
from unittest.mock import Mock, patch

import pyray as rl

from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS
from openpilot.selfdrive.ui.lib.prime_state import PrimeType
from openpilot.selfdrive.ui.mici.layouts.offroad_alerts import MiciOffroadAlerts
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.mici.layouts.settings.device import DeviceLayoutMici
from openpilot.selfdrive.ui.mici.layouts.settings.settings import SettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state, device
from openpilot.system.ui.lib.application import gui_app, MousePos, MouseEvent
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.scroller import Scroller


class TestPrimeOffroadAlerts(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(gui_app, "font", return_value=rl.Font()))
    self.textures = self.enterContext(patch.object(gui_app, "texture", side_effect=lambda *args: rl.Texture()))
    self.enterContext(patch.object(UnifiedLabel, "get_content_height", return_value=32))
    self.enterContext(patch("openpilot.selfdrive.ui.mici.layouts.offroad_alerts.threading.Thread"))
    self.enterContext(patch("openpilot.selfdrive.ui.mici.layouts.offroad_alerts.Params"))
    self.prime_type = self.enterContext(patch.object(ui_state.prime_state, "get_type", return_value=PrimeType.UNPAIRED))
    self.alerts = MiciOffroadAlerts()

  def test_pairing_transitions(self):
    for prime_type, expected in ((PrimeType.UNPAIRED, "PairDevice"), (PrimeType.NONE, "UpgradeToPrime"),
                                 (PrimeType.LITE, None), (PrimeType.MAGENTA, None), (PrimeType.NONE, "UpgradeToPrime"),
                                 (PrimeType.UNPAIRED, "PairDevice")):
      with self.subTest(prime_type=prime_type):
        self.prime_type.return_value = prime_type
        self.alerts._update_state()
        self.assertEqual([alert.key for alert in self.alerts.sorted_alerts if alert.visible], [expected] if expected else [])
        self.assertEqual(self.alerts.active_alerts(), int(expected is not None))
        self.assertEqual(self.alerts.max_severity(), -1 if expected else None)

  def test_both_prime_alerts_navigate_to_pairing(self):
    callback = Mock()
    self.alerts.set_pairing_callback(callback)
    for prime_type in (PrimeType.UNPAIRED, PrimeType.NONE):
      with self.subTest(prime_type=prime_type):
        self.prime_type.return_value = prime_type
        self.alerts._update_state()
        alert = next(item for item in self.alerts._prime_alert_items if item.alert_data.visible)
        callback.reset_mock()
        alert._handle_mouse_release(MousePos(0, 0))
        callback.assert_called_once_with()

  def test_param_refresh_preserves_prime_alerts_and_warning_priority(self):
    params = dict.fromkeys(OFFROAD_ALERTS)
    params.update(UpdateAvailable=False, UpdaterNewDescription="")
    params["Offroad_ConnectivityNeeded"] = {"text": "Connect to internet."}
    self.assertEqual(self.alerts._refresh(params), 2)
    self.assertTrue(self.alerts._pairing_alert.visible)
    visible_items = [item for item in self.alerts._scroller.items if item.alert_data.visible]
    self.assertEqual(visible_items[0].alert_data.key, "Offroad_ConnectivityNeeded")
    self.assertEqual(self.alerts.max_severity(), 1)

  def test_alert_copy_and_icons(self):
    pairing, prime = self.alerts._prime_alert_items
    self.assertEqual(pairing._title_text, "Finish setup")
    self.assertEqual(pairing._body_text, "Pair your device with comma connect (connect.comma.ai) and claim your comma prime offer.")
    self.prime_type.return_value = PrimeType.NONE
    self.alerts._update_state()
    self.assertEqual(prime._title_text, "Upgrade to prime")
    for filename in ("green_settings.png", "green_cell.png"):
      self.textures.assert_any_call(f"icons_mici/offroad_alerts/{filename}", 64, 64)

  def test_icon_top_right_position(self):
    for item in self.alerts._prime_alert_items:
      item.alert_data.visible = True
      item.update_alert_data(item.alert_data)
      item.set_position(10, 20)
      with patch.object(rl, "draw_texture_ex") as draw, patch.object(UnifiedLabel, "render"):
        item._render(item.rect)
      icon, position, _, scale, _ = draw.call_args.args
      self.assertIs(icon, item._custom_icon)
      self.assertEqual(position.x, 10 + item.ALERT_WIDTH - item.ALERT_PADDING - item.ICON_SIZE)
      self.assertEqual(position.y, 20 + item.ALERT_PADDING)
      self.assertEqual(scale, 1.0)

  def test_finish_setup_opens_and_highlights_pairing_button(self):
    panel = DeviceLayoutMici()
    panel.set_rect(rl.Rectangle(0, 0, 536, 240))
    settings = object.__new__(SettingsLayout)
    settings._device_panel = panel
    self.alerts.set_pairing_callback(lambda: settings._open_device(highlight_pairing=True))

    def check_grow():
      self.assertFalse(panel._scroller.is_auto_scrolling)
      button_center = panel._pairing_button.rect.x + panel._pairing_button.rect.width / 2
      self.assertAlmostEqual(button_center, panel.rect.x + panel.rect.width / 2, delta=1)

    with patch.object(gui_app, "push_widget", side_effect=lambda widget: widget.show_event()) as push, \
         patch.object(rl, "get_time", return_value=1.0), \
         patch.object(panel._scroller, "_get_scroll", side_effect=lambda *_: panel._scroller.scroll_panel.get_offset()), \
         patch.object(panel._pairing_button, "trigger_grow_animation", side_effect=check_grow) as grow:
      self.alerts._prime_alert_items[0]._handle_mouse_release(MousePos(0, 0))
      push.assert_called_once_with(panel)
      grow.assert_not_called()

      for _ in range(300):
        panel._update_state()
        panel._scroller.set_rect(panel.rect)
        panel._scroller._update_state()
        panel._scroller._layout()
      grow.assert_called_once()

      self.assertIsNone(panel._shown_callback)
      panel.hide_event()
      panel.show_event()
      for _ in range(100):
        panel._update_state()
      grow.assert_called_once()

  def test_pair_click_does_not_retrigger_alert_underneath(self):
    main = object.__new__(MiciMainLayout)
    Scroller.__init__(main)
    main._alerts_layout = self.alerts
    main._home_layout = Mock()
    main._car_onroad_layout = Mock()
    main._body_onroad_layout = Mock()
    settings = self._make_settings()
    panel = settings._device_panel
    main._settings_layout = settings
    main._scroller.add_widget(self.alerts)

    with patch.object(device, "add_interactive_timeout_callback"), patch.object(ui_state, "add_on_body_changed_callbacks"):
      main._setup_callbacks()

    events = [MouseEvent(MousePos(100, 100), 0, True, False, True, 1),
              MouseEvent(MousePos(100, 100), 0, False, True, False, 1.1)]
    with patch.object(gui_app, "_nav_stack", []), patch.object(gui_app, "_mouse_events", events), \
         patch("openpilot.system.ui.lib.application.cloudlog.warning") as warning, \
         patch.object(ui_state.prime_state, "is_paired", return_value=False), \
         patch.object(ui_state.params, "get", return_value="0123456789abcdef"), \
         patch("openpilot.selfdrive.ui.mici.layouts.settings.device.system_time_valid", return_value=True), \
         patch("openpilot.selfdrive.ui.mici.layouts.settings.device.PairingDialog") as pairing_dialog:
      gui_app.push_widget(main)
      alert = self.alerts._prime_alert_items[0]
      alert._handle_mouse_release(MousePos(100, 100))
      self.assertIs(gui_app.get_active_widget(), panel)
      self.assertFalse(self.alerts._scroller.enabled)
      self.assertFalse(alert._touch_valid())

      panel._pairing_button.set_position(0, 0)
      alert._process_mouse_events()
      panel._pairing_button._process_mouse_events()
      pairing_dialog.assert_called_once()
      self.assertIs(gui_app.get_active_widget(), pairing_dialog.return_value)
      warning.assert_not_called()

      gui_app.pop_widget()
      gui_app.pop_widget()
      self.assertIs(gui_app.get_active_widget(), settings)
      self.assertFalse(self.alerts._scroller.enabled)
      gui_app.pop_widget()
      self.assertTrue(self.alerts._scroller.enabled)

  def _make_settings(self):
    for name in ("TogglesLayoutMici", "NetworkLayoutMici", "SoftwareLayoutMici", "DeveloperLayoutMici", "FirehoseLayout"):
      self.enterContext(patch(f"openpilot.selfdrive.ui.mici.layouts.settings.settings.{name}"))
    settings = SettingsLayout()
    for panel in (settings, settings._device_panel):
      panel.set_rect(rl.Rectangle(0, 0, 536, 240))
    return settings

  def test_prime_alerts_open_device_directly_with_settings_underneath(self):
    for prime_type in (PrimeType.UNPAIRED, PrimeType.NONE):
      with self.subTest(prime_type=prime_type):
        self.prime_type.return_value = prime_type
        self.alerts._update_state()
        settings = self._make_settings()
        panel = settings._device_panel
        self.alerts.set_pairing_callback(settings.show_pairing)
        for page in (settings, panel):
          scroller = page._scroller
          self.enterContext(patch.object(scroller, "_get_scroll", side_effect=lambda *_, s=scroller: s.scroll_panel.get_offset()))

        def check_pair_highlight(panel=panel):
          self.assertIs(gui_app.get_active_widget(), panel)
          self.assertFalse(panel._scroller.is_auto_scrolling)
          button = panel._pairing_button
          self.assertAlmostEqual(button.rect.x + button.rect.width / 2, panel.rect.x + panel.rect.width / 2, delta=1)

        with patch.object(gui_app, "_nav_stack", [self.alerts]), patch.object(rl, "get_time", return_value=0) as clock, \
             patch.object(settings._device_button, "trigger_grow_animation") as settings_grow, \
             patch.object(panel._pairing_button, "trigger_grow_animation", side_effect=check_pair_highlight) as pair_grow:
          alert = next(item for item in self.alerts._prime_alert_items if item.alert_data.visible)
          alert._handle_mouse_release(MousePos(0, 0))
          self.assertEqual(gui_app._nav_stack, [self.alerts, settings, panel])
          self.assertIs(gui_app.get_active_widget(), panel)
          self.assertEqual(settings._y_pos_filter.x, 0)
          self.assertFalse(settings.is_visible)
          self.assertFalse(settings._scroller.is_auto_scrolling)
          self.assertFalse(panel._scroller.is_auto_scrolling)

          for frame in range(300):
            clock.return_value = frame / 60
            if panel._shown_callback is not None:
              self.assertFalse(settings.is_visible)
            for page in (settings, panel):
              page._update_state()
              page._scroller.set_rect(page.rect)
              page._scroller._update_state()
              page._scroller._layout()

          settings_grow.assert_not_called()
          pair_grow.assert_called_once()
          self.assertTrue(settings.is_visible)
          self.assertIsNone(panel._shown_callback)
          gui_app.pop_widget()
          self.assertIs(gui_app.get_active_widget(), settings)
          settings._update_state()
          self.assertTrue(settings.is_visible)
          self.assertIs(gui_app.get_active_widget(), settings)
          button = settings._device_button
          self.assertAlmostEqual(button.rect.x + button.rect.width / 2, settings.rect.x + settings.rect.width / 2, delta=1)

  def test_dismissing_device_during_entrance_reveals_settings(self):
    settings = self._make_settings()
    panel = settings._device_panel
    with patch.object(gui_app, "_nav_stack", [self.alerts]):
      settings.show_pairing()
      self.assertFalse(settings.is_visible)
      panel.dismiss()
      self.assertTrue(settings.is_visible)
      for _ in range(180):
        panel._update_state()
        if not gui_app.widget_in_stack(panel):
          break
      self.assertIs(gui_app.get_active_widget(), settings)
      settings._update_state()
      self.assertTrue(settings.is_visible)
      self.assertIsNone(panel._shown_callback)
      self.assertFalse(panel._pending_pairing_grow_animation)
      settings._open_device()
      self.assertTrue(settings.is_visible)
      self.assertIsNone(panel._shown_callback)
