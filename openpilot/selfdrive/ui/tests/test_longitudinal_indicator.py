import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.cereal import messaging, log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer, SET_SPEED_PERSISTENCE


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
    self.hud._reset_distance_highlight()
    self.hud._longitudinal_icon_opacity = 1.0
    self.hud._txt_lead_car = SimpleNamespace(width=128, height=101)
    self.hud._txt_lead_car_green = self.hud._txt_lead_car_orange = SimpleNamespace(width=124, height=110)
    self.hud._car_triangle_parts = [('tri', 33, 137), ('tri_green', 21, 125), ('tri_orange', 21, 125)]
    self.hud._distance_icon_parts = []
    self.rect = rl.Rectangle(0, 0, 536, 240)
    self.addCleanup(patch.stopall)
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.ui_state', SimpleNamespace(sm=self.sm, started_frame=1)).start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex').start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro').start()
    patch.object(self.hud, '_braking_orange_alpha', return_value=0).start()

  def test_selection_timeout_and_smooth_repeated_toggles(self):
    aggressive, standard, relaxed = (log.LongitudinalPersonality.aggressive,
                                     log.LongitudinalPersonality.standard,
                                     log.LongitudinalPersonality.relaxed)
    self.hud._update_longitudinal_layout(aggressive, 100)
    self.assertEqual([f.x for f in self.hud._layout_filters], [0, 1, 0, 0])
    self.hud._update_longitudinal_layout(standard, 101)
    self.assertGreater(self.hud._layout_filters[1].x, 0)
    self.assertGreater(self.hud._layout_filters[2].x, 0)
    self.assertLess(self.hud._layout_filters[2].x, 1)
    self.hud._update_longitudinal_layout(relaxed, 101.1)
    self.assertEqual(self.hud._layout_highlight_time, 101.1)
    self.assertGreater(self.hud._layout_filters[2].x, 0)
    self.assertGreater(self.hud._layout_filters[3].x, 0)
    for _ in range(100):
      self.hud._update_longitudinal_layout(relaxed, 101.1 + SET_SPEED_PERSISTENCE - 0.01)
    self.assertAlmostEqual(self.hud._layout_filters[3].x, 1, places=5)
    self.assertEqual(self.hud._layout_filters[0].x, 0)
    self.hud._update_longitudinal_layout(relaxed, 101.1 + SET_SPEED_PERSISTENCE)
    self.assertGreater(self.hud._layout_filters[0].x, 0)
    self.assertLess(self.hud._layout_filters[0].x, 1)
    for _ in range(100):
      self.hud._update_longitudinal_layout(relaxed, 105)
    self.assertAlmostEqual(self.hud._layout_filters[0].x, 1, places=5)
    self.assertAlmostEqual(sum(f.x for f in self.hud._layout_filters), 1)
    self.hud._reset_distance_highlight()
    self.hud._update_longitudinal_layout(standard, 110)
    self.assertEqual([f.x for f in self.hud._layout_filters], [0, 0, 1, 0])

  def test_personality_assets_and_car_only_layout(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
    expected = {0: [], 1: [(18, 139)], 2: [(22, 129), (18, 144)], 3: [(26, 119), (22, 132), (18, 147)]}
    for count in range(4):
      for i, f in enumerate(self.hud._layout_filters):
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
      if count == 0:
        self.assertEqual((glow.x, glow.y, glow.width, glow.height), (-5, 75, 94, 83))
      else:
        self.assertAlmostEqual(glow.x + glow.width * 28 / 124, white.x, places=4)
        self.assertAlmostEqual(glow.y + glow.height * 28 / 110, white.y, places=4)

  def test_collapse_moves_each_bar_once_and_scales_car(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
    for f, weight in zip(self.hud._layout_filters, (0.5, 0, 0, 0.5), strict=True):
      f.x = weight
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as bars, \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car:
      self.hud._draw_distance_bars(self.rect)
      self.hud._draw_lead_car(self.rect)
    self.assertEqual([c.args[0] for c in bars.call_args_list], ['w0', 'w1', 'w2', 'tri', 'tri_green', 'tri_orange'])
    self.assertEqual([c.args[1].y for c in bars.call_args_list[:3]], [111, 124, 139])
    self.assertTrue(all(c.args[3] == 1 for c in bars.call_args_list))
    self.assertEqual(bars.call_args_list[2].args[4].a, round(255 * 0.9 * 0.5))
    white = car.call_args_list[0].args[2]
    self.assertEqual((white.x, white.y, white.width, white.height), (20.5, 91, 43, 34))

  def test_triangle_matches_car_colors_and_crossfade(self):
    from opendbc.car.structs import car
    for resting in (0.0, 0.5, 1.0):
      for i, f in enumerate(self.hud._layout_filters):
        f.x = resting if i == 0 else (1 - resting if i == 3 else 0)
      for state in ('no_lead', 'radar', 'vision', 'fcw'):
        self.sm['longitudinalPlan'].hasLead = state != 'no_lead'
        self.sm['longitudinalPlan'].longitudinalPlanSource = (log.LongitudinalPlan.LongitudinalPlanSource.e2e if state == 'vision'
                                                            else log.LongitudinalPlan.LongitudinalPlanSource.lead0)
        self.sm['selfdriveState'].alertHudVisual = (car.CarControl.HUDControl.VisualAlert.fcw if state == 'fcw'
                                                  else car.CarControl.HUDControl.VisualAlert.none)
        with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as triangle, \
             patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro'):
          self.hud._draw_lead_car(self.rect)
        self.assertEqual(triangle.call_count, 3 if resting else 0)
        for index, call in enumerate(triangle.call_args_list):
          filters = (self.hud._lead_car_white_filter, self.hud._lead_car_green_filter, self.hud._lead_car_orange_filter)
          self.assertEqual(call.args[4].a, round(255 * filters[index].x * resting))
          self.assertEqual((call.args[1].x, call.args[1].y), (33, 137) if index == 0 else (21, 125))

  def test_personality_car_is_white_for_all_lead_states(self):
    from opendbc.car.structs import car
    for count in (1, 2, 3):
      for i, fade in enumerate(self.hud._layout_filters):
        fade.x = float(i == count)
      for has_lead, fcw in ((False, False), (True, False), (True, True)):
        self.sm['longitudinalPlan'].hasLead = has_lead
        self.sm['selfdriveState'].alertHudVisual = (car.CarControl.HUDControl.VisualAlert.fcw if fcw
                                                  else car.CarControl.HUDControl.VisualAlert.none)
        with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as draw:
          self.hud._draw_lead_car(self.rect)
        self.assertEqual([c.args[5].a for c in draw.call_args_list], [round(255 * 0.9), 0, 0])

  def test_override_pulses_only_triangle_and_ignores_stale_events(self):
    import math
    self.sm['onroadEvents'] = [SimpleNamespace(name=log.OnroadEvent.EventName.gasPressedOverride)]
    for now, opacity in ((0, 0.35), (math.pi / 6, 1.0)):
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=now), \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as triangle, \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as car_draw:
        self.hud._draw_lead_car(self.rect)
      self.assertEqual([c.args[4].a for c in triangle.call_args_list], [0, 0, round(255 * opacity)])
      self.assertEqual([c.args[5].a for c in car_draw.call_args_list], [round(255 * 0.9), 0, 0])
    self.sm.alive['onroadEvents'] = False
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', side_effect=AssertionError('No stale pulse')), \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as triangle:
      self.hud._draw_lead_car(self.rect)
    self.assertEqual([c.args[4].a for c in triangle.call_args_list], [round(255 * 0.9), 0, 0])

  def test_car_green_tracks_policy_not_gap(self):
    for distance in (10, 35, 80):
      self.sm['radarState'].leadOne.dRel = distance
      self.hud._draw_lead_car(self.rect)
      self.assertEqual(self.hud._lead_car_green_filter.x, 0)
    self.sm['longitudinalPlan'].longitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource.lead0
    for _ in range(100):
      self.hud._draw_lead_car(self.rect)
    self.assertAlmostEqual(self.hud._lead_car_green_filter.x, 1, places=5)
    self.assertAlmostEqual(self.hud._lead_car_white_filter.x, 0, places=5)


if __name__ == '__main__':
  unittest.main()
