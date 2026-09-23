import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.cereal import messaging, log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer


class FakeSM(dict):
  def __init__(self, messages):
    super().__init__(messages)
    self.valid = dict.fromkeys(messages, True)
    self.alive = dict.fromkeys(messages, True)
    self.recv_frame = dict.fromkeys(messages, 1)


class TestLongitudinalIndicator(unittest.TestCase):
  def setUp(self):
    state = messaging.new_message('selfdriveState').selfdriveState
    state.personality = log.LongitudinalPersonality.standard
    self.sm = FakeSM({
      'selfdriveState': state,
      'onroadEvents': [],
      'longitudinalPlan': SimpleNamespace(hasLead=True, longitudinalPlanSource=log.LongitudinalPlan.LongitudinalPlanSource.e2e),
      'carState': SimpleNamespace(vEgo=20),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=0, vRel=0)),
    })
    self.hud = HudRenderer.__new__(HudRenderer)
    for name in ('_lead_car_white_filter', '_lead_car_green_filter', '_lead_car_orange_filter'):
      setattr(self.hud, name, FirstOrderFilter(0, 0.1, 1 / 60, initialized=False))
    self.hud._reset_longitudinal_layout()
    self.hud._longitudinal_icon_opacity = 1.0
    self.hud._txt_lead_car = SimpleNamespace(width=128, height=101)
    self.hud._txt_lead_car_green = self.hud._txt_lead_car_orange = SimpleNamespace(width=124, height=110)
    self.hud._distance_icon_parts = []
    self.rect = rl.Rectangle(0, 0, 536, 240)
    self.addCleanup(patch.stopall)
    self.ui_state = SimpleNamespace(sm=self.sm, started_frame=1, has_longitudinal_control=True)
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.ui_state', self.ui_state).start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex').start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro').start()

  def test_persistent_selection_and_smooth_repeated_toggles(self):
    aggressive, standard, relaxed = (log.LongitudinalPersonality.aggressive,
                                     log.LongitudinalPersonality.standard,
                                     log.LongitudinalPersonality.relaxed)
    self.hud._update_longitudinal_layout(aggressive)
    self.assertEqual([f.x for f in self.hud._layout_filters], [1, 0, 0])
    self.hud._update_longitudinal_layout(standard)
    self.assertGreater(self.hud._layout_filters[0].x, 0)
    self.assertGreater(self.hud._layout_filters[1].x, 0)
    self.hud._update_longitudinal_layout(relaxed)
    self.assertGreater(self.hud._layout_filters[1].x, 0)
    self.assertGreater(self.hud._layout_filters[2].x, 0)
    # Remains expanded indefinitely, independent of clock or lead data.
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=10000):
      for _ in range(1000):
        self.hud._update_longitudinal_layout(relaxed)
    self.assertAlmostEqual(self.hud._layout_filters[2].x, 1, places=5)
    self.assertAlmostEqual(sum(f.x for f in self.hud._layout_filters), 1)
    self.hud._reset_longitudinal_layout()
    self.hud._update_longitudinal_layout(standard)
    self.assertEqual([f.x for f in self.hud._layout_filters], [0, 1, 0])

  def test_personality_assets_and_alignment(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    expected = {0: [], 1: [(18, 139)], 2: [(22, 129), (18, 144)], 3: [(26, 119), (22, 132), (18, 147)]}
    for count in (1, 2, 3):
      for i, f in enumerate(self.hud._layout_filters, start=1):
        f.x = float(i == count)
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as bars, \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car:
        self.hud._draw_distance_bars(self.rect)
        self.hud._draw_lead_car(self.rect)
      whites = [c for c in bars.call_args_list if c.args[0].startswith('w')]
      greens = [c for c in bars.call_args_list if c.args[0].startswith('g')]
      self.assertEqual([(c.args[1].x, c.args[1].y) for c in whites], expected[count])
      self.assertEqual(len(greens), 0)
      if count:
        self.assertEqual([c.args[4].a for c in whites], [round(255 * 0.9)] * count)
      white, glow = (car.call_args_list[i].args[2] for i in (0, 1))
      self.assertEqual((white.x, white.y, white.width, white.height), self.hud._longitudinal_layout(count)[0])
      self.assertAlmostEqual(glow.x + glow.width * 28 / 124, white.x, places=4)
      self.assertAlmostEqual(glow.y + glow.height * 28 / 110, white.y, places=4)

  def test_personality_transition_moves_each_bar_once_and_scales_car(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    for f, weight in zip(self.hud._layout_filters, (0.5, 0, 0.5), strict=True):
      f.x = weight
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as bars, \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car:
      self.hud._draw_distance_bars(self.rect)
      self.hud._draw_lead_car(self.rect)
    self.assertEqual([c.args[0] for c in bars.call_args_list], ['w0', 'w1', 'w2'])
    self.assertEqual([c.args[1].y for c in bars.call_args_list[:3]], [115, 128, 143])
    self.assertTrue(all(c.args[3] == 1 for c in bars.call_args_list))
    self.assertEqual(bars.call_args_list[2].args[4].a, round(255 * 0.9))
    white = car.call_args_list[0].args[2]
    self.assertEqual((white.x, white.y, white.width, white.height), (21.5, 90.5, 41, 32.5))

  def test_without_longitudinal_control_shows_centered_car_only(self):
    self.ui_state.has_longitudinal_control = False
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    for personality in (log.LongitudinalPersonality.aggressive, log.LongitudinalPersonality.relaxed):
      self.hud._update_longitudinal_layout(personality)
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as bars, \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car:
        self.hud._draw_distance_bars(self.rect)
        self.hud._draw_lead_car(self.rect)
      bars.assert_not_called()
      white, green, orange = [call.args[2] for call in car.call_args_list]
      self.assertEqual((white.x, white.y, white.width, white.height), (16, 102, 52, 41))
      for glow in (green, orange):
        self.assertEqual((glow.width, glow.height), (94, 83))
        self.assertEqual(glow.x + glow.width / 2, white.x + white.width / 2)
        self.assertEqual(glow.y + glow.height / 2, white.y + white.height / 2)

  def test_personality_car_preserves_lead_and_warning_colors(self):
    from opendbc.car.structs import car
    for count in (1, 2, 3):
      for i, fade in enumerate(self.hud._layout_filters, start=1):
        fade.x = float(i == count)
      for has_lead, fcw, expected in ((False, False, [round(255 * 0.35), 0, 0]),
                                       (True, False, [0, 255, 0]), (True, True, [0, 0, 255])):
        self.sm['longitudinalPlan'].hasLead = has_lead
        self.sm['selfdriveState'].alertHudVisual = (car.CarControl.HUDControl.VisualAlert.fcw if fcw
                                                  else car.CarControl.HUDControl.VisualAlert.none)
        for name in ('_lead_car_white_filter', '_lead_car_green_filter', '_lead_car_orange_filter'):
          getattr(self.hud, name).initialized = False
        with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as draw:
          self.hud._draw_lead_car(self.rect)
        self.assertEqual([c.args[5].a for c in draw.call_args_list], expected)

  def test_override_blinks_at_160_bpm_and_rejects_stale_events(self):
    self.sm['onroadEvents'] = [SimpleNamespace(name=log.OnroadEvent.EventName.gasPressedOverride)]
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.time.monotonic', return_value=10):
      self.assertEqual(self.hud._distance_override_opacity(), 1)
    for elapsed, expected in ((0.1, 1), (0.1875, 0.35 / 0.9), (0.374, 0.35 / 0.9),
                              (0.375, 1), (0.5625, 0.35 / 0.9), (3.75, 1)):
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.time.monotonic', return_value=10 + elapsed):
        self.assertEqual(self.hud._distance_override_opacity(), expected)
    for field, invalid in (('valid', False), ('alive', False), ('recv_frame', 0)):
      values = getattr(self.sm, field)
      previous = values['onroadEvents']
      values['onroadEvents'] = invalid
      self.assertEqual(self.hud._distance_override_opacity(), 1)
      self.assertIsNone(self.hud._distance_override_timer)
      values['onroadEvents'] = previous
    self.sm['onroadEvents'] = []
    self.assertEqual(self.hud._distance_override_opacity(), 1)

  def test_override_dims_white_bars_without_changing_car(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    for i, fade in enumerate(self.hud._layout_filters):
      fade.x = float(i == 2)
    with patch.object(self.hud, '_distance_override_opacity', return_value=0.4), \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as bars, \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car_draw:
      self.hud._draw_distance_bars(self.rect)
      self.hud._draw_lead_car(self.rect)
    alphas = {call.args[0]: call.args[4].a for call in bars.call_args_list}
    self.assertEqual(alphas, {'w0': round(255 * 0.9 * 0.4), 'w1': round(255 * 0.9 * 0.4), 'w2': round(255 * 0.9 * 0.4)})
    self.assertEqual([call.args[5].a for call in car_draw.call_args_list], [0, 255, 0])

  def test_car_green_tracks_policy_not_gap(self):
    for distance in (10, 35, 80):
      self.sm['radarState'].leadOne.dRel = distance
      self.hud._draw_lead_car(self.rect)
      self.assertEqual(self.hud._lead_car_green_filter.x, 1)
    self.sm['longitudinalPlan'].longitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource.lead0
    for _ in range(100):
      self.hud._draw_lead_car(self.rect)
    self.assertAlmostEqual(self.hud._lead_car_green_filter.x, 0, places=5)
    self.assertAlmostEqual(self.hud._lead_car_white_filter.x, 0.9, places=5)


if __name__ == '__main__':
  unittest.main()
