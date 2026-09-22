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
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.ui_state', SimpleNamespace(sm=self.sm, started_frame=1)).start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex').start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro').start()
    patch.object(self.hud, '_braking_orange_alpha', return_value=0).start()

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
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
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
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
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

  def test_nearest_bar_highlight_and_timer(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
    for personality, count in ((log.LongitudinalPersonality.aggressive, 1),
                               (log.LongitudinalPersonality.standard, 2),
                               (log.LongitudinalPersonality.relaxed, 3)):
      self.hud._reset_longitudinal_layout()
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=100):
        for _ in range(100):
          self.hud._update_longitudinal_layout(personality)
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as draw:
        self.hud._draw_distance_bars(self.rect)
      greens = [call for call in draw.call_args_list if call.args[0].startswith('g')]
      self.assertEqual(len(greens), 1)
      self.assertEqual(greens[0].args[0], f'g{3 - count}')
      self.assertEqual(greens[0].args[4].a, 255)
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=102.5):
        self.hud._update_longitudinal_layout(personality)
      self.assertGreater(self.hud._distance_highlight_filter.x, 0)
      self.assertLess(self.hud._distance_highlight_filter.x, 1)
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=105):
        for _ in range(100):
          self.hud._update_longitudinal_layout(personality)
        self.assertAlmostEqual(self.hud._distance_highlight_filter.x, 0, places=5)
        self.hud._update_longitudinal_layout(log.LongitudinalPersonality.standard if count != 2 else log.LongitudinalPersonality.relaxed)
      self.assertEqual(self.hud._distance_highlight_time, 105)
      self.assertGreater(self.hud._distance_highlight_filter.x, 0)

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
