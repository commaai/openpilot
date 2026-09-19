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
      'longitudinalPlan': SimpleNamespace(hasLead=True, longitudinalPlanSource=log.LongitudinalPlan.LongitudinalPlanSource.e2e),
      'carState': SimpleNamespace(vEgo=20),
      'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=0, vRel=0)),
    })
    self.hud = HudRenderer.__new__(HudRenderer)
    for name in ('_lead_car_white_filter', '_lead_car_green_filter', '_lead_car_orange_filter', '_distance_highlight_filter'):
      setattr(self.hud, name, FirstOrderFilter(0, 0.1, 1 / 60, initialized=False))
    self.hud._reset_distance_highlight()
    self.hud._longitudinal_icon_opacity = self.hud._accel_override_alpha = 1.0
    self.hud._txt_lead_car = SimpleNamespace(width=128, height=101)
    self.hud._txt_lead_car_green = self.hud._txt_lead_car_orange = SimpleNamespace(width=184, height=157)
    self.hud._distance_icon_parts = []
    self.rect = rl.Rectangle(0, 0, 536, 240)
    self.addCleanup(patch.stopall)
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.ui_state', SimpleNamespace(sm=self.sm, started_frame=1)).start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex').start()
    patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro').start()
    patch.object(self.hud, '_braking_orange_alpha', return_value=0).start()

  def test_personality_car_size_and_glow_alignment(self):
    for personality, bounds in ((log.LongitudinalPersonality.relaxed, (25, 86, 34, 27)),
                                (log.LongitudinalPersonality.standard, (21, 89, 42, 34)),
                                (log.LongitudinalPersonality.aggressive, (18, 95, 48, 38))):
      self.sm['selfdriveState'].personality = personality
      with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_pro') as draw:
        self.hud._draw_lead_car(self.rect)
      white, glow = (draw.call_args_list[i].args[2] for i in (0, 1))
      np_bounds = (white.x, white.y, white.width, white.height)
      self.assertEqual(np_bounds, bounds)
      self.assertAlmostEqual(glow.x + glow.width * 28 / 184, white.x, places=4)
      self.assertAlmostEqual(glow.y + glow.height * 28 / 157, white.y, places=4)
      self.assertAlmostEqual(glow.width * 128 / 184, white.width, places=4)
      self.assertAlmostEqual(glow.height * 101 / 157, white.height, places=4)

  def test_personality_bar_placement(self):
    self.hud._distance_icon_parts = [(None, 26, 119), (None, 22, 132), (None, 18, 147)]
    for personality, expected in ((log.LongitudinalPersonality.relaxed, [(26, 119), (22, 132), (18, 147)]),
                                   (log.LongitudinalPersonality.standard, [(22, 129), (18, 144)]),
                                   (log.LongitudinalPersonality.aggressive, [(18, 139)])):
      self.sm['selfdriveState'].personality = personality
      with patch.object(self.hud, '_distance_highlight_alpha', return_value=0), \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as draw:
        self.hud._draw_distance_bars(self.rect)
      self.assertEqual([(call.args[1].x, call.args[1].y) for call in draw.call_args_list], expected)

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

  def test_acceleration_override_is_steady_dim(self):
    self.sm['onroadEvents'] = [SimpleNamespace(name=log.OnroadEvent.EventName.gasPressedOverride)]
    self.sm.valid['onroadEvents'] = self.sm.alive['onroadEvents'] = True
    self.sm.recv_frame['onroadEvents'] = 1
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', side_effect=AssertionError('No pulse clock')):
      self.assertEqual(self.hud._acceleration_override_opacity(), 0.35)
      self.assertEqual(self.hud._acceleration_override_opacity(), 0.35)
      self.sm['onroadEvents'] = []
      self.assertEqual(self.hud._acceleration_override_opacity(), 1.0)
    self.sm['onroadEvents'] = [SimpleNamespace(name=log.OnroadEvent.EventName.gasPressedOverride)]
    self.sm.valid['onroadEvents'] = False
    self.assertEqual(self.hud._acceleration_override_opacity(), 1.0)

  def test_bar_only_flashes_on_personality_change(self):
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=100):
      self.hud._draw_distance_bars(self.rect)
    self.assertEqual(self.hud._distance_highlight_filter.x, 0)
    self.sm['longitudinalPlan'].longitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource.lead0
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=101):
      self.hud._draw_distance_bars(self.rect)
    self.assertEqual(self.hud._distance_highlight_filter.x, 0)
    previous = self.hud._distance_highlight_filter.x
    self.sm['selfdriveState'].personality = log.LongitudinalPersonality.relaxed
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=102):
      self.hud._draw_distance_bars(self.rect)
    self.assertGreater(self.hud._distance_highlight_filter.x, previous)
    previous = self.hud._distance_highlight_filter.x
    with patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.get_time', return_value=103 + SET_SPEED_PERSISTENCE):
      self.hud._draw_distance_bars(self.rect)
    self.assertLess(self.hud._distance_highlight_filter.x, previous)


if __name__ == '__main__':
  unittest.main()
