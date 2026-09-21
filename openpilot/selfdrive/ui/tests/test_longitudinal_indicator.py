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

  def test_static_car_size_and_glow_alignment(self):
    for personality, bounds in ((log.LongitudinalPersonality.relaxed, (25, 86, 34, 27)),
                                (log.LongitudinalPersonality.standard, (25, 86, 34, 27)),
                                (log.LongitudinalPersonality.aggressive, (25, 86, 34, 27))):
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
                                   (log.LongitudinalPersonality.standard, [(26, 119), (22, 132), (18, 147)]),
                                   (log.LongitudinalPersonality.aggressive, [(26, 119), (22, 132), (18, 147)])):
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

  def test_bar_highlight_on_engagement_and_personality_change(self):
    self.assertGreater(self.hud._distance_highlight_alpha(1, 100), 0)
    for _ in range(100):
      self.hud._distance_highlight_alpha(1, 101)
    self.assertAlmostEqual(self.hud._distance_highlight_filter.x, 1, places=5)
    self.hud._distance_highlight_alpha(1, 100 + SET_SPEED_PERSISTENCE + 0.1)
    self.assertLess(self.hud._distance_highlight_filter.x, 1)
    # A new selection restarts the timer, including rapid successive presses.
    self.hud._distance_highlight_alpha(2, 103)
    self.hud._distance_highlight_alpha(0, 103.1)
    self.assertEqual(self.hud._personality_highlight_time, 103.1)
    for _ in range(100):
      self.hud._distance_highlight_alpha(0, 107)
    self.assertAlmostEqual(self.hud._distance_highlight_filter.x, 0, places=5)
    self.hud._reset_distance_highlight()
    self.assertGreater(self.hud._distance_highlight_alpha(0, 108), 0)
    self.assertEqual(self.hud._personality_highlight_time, 108)

  def test_live_distance_bands_and_invalid_leads(self):
    lead = self.sm['radarState'].leadOne
    # At 20 m/s the personality gaps are 31, 35, and 41 meters.
    for distance, count in ((1, 1), (31, 1), (31.1, 2), (35, 2), (35.1, 3), (41, 3), (41.1, 0)):
      lead.dRel = distance
      self.assertEqual(self.hud._lead_distance_bar_count(), count)
    self.sm['carState'].vEgo = 0
    for distance, count in ((6, 1), (6.1, 0), (0, 0), (-1, 0), (float('nan'), 0), (float('inf'), 0)):
      lead.dRel = distance
      self.assertEqual(self.hud._lead_distance_bar_count(), count)
    self.sm['carState'].vEgo = 20
    lead.dRel = 20
    lead.present = False
    self.assertEqual(self.hud._lead_distance_bar_count(), 0)
    lead.present = True
    self.sm['longitudinalPlan'].hasLead = False
    self.assertEqual(self.hud._lead_distance_bar_count(), 0)
    self.sm['longitudinalPlan'].hasLead = True
    for service in ('longitudinalPlan', 'radarState', 'carState'):
      for field, invalid in (('valid', False), ('alive', False), ('recv_frame', 0)):
        values = getattr(self.sm, field)
        previous = values[service]
        values[service] = invalid
        self.assertEqual(self.hud._lead_distance_bar_count(), 0)
        values[service] = previous

  def test_live_bars_fill_cumulatively_and_yield_to_personality(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
    for count in (1, 2, 3, 0):
      with patch.object(self.hud, '_lead_distance_bar_count', return_value=count), \
           patch.object(self.hud, '_distance_highlight_alpha', return_value=0), \
           patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as draw:
        for _ in range(100):
          self.hud._draw_distance_bars(self.rect)
        for index, call in enumerate(draw.call_args_list[-3:]):
          self.assertAlmostEqual(call.args[4].a, round(255 * (0.9 if index < count else 0.35)), delta=1)
    self.sm['selfdriveState'].personality = log.LongitudinalPersonality.aggressive
    with patch.object(self.hud, '_lead_distance_bar_count', return_value=3), \
         patch.object(self.hud, '_distance_highlight_alpha', return_value=1), \
         patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as draw:
      self.hud._draw_distance_bars(self.rect)
      self.assertEqual({call.args[0]: call.args[4].a for call in draw.call_args_list},
                       {'w0': 0, 'g0': 255, 'w1': round(255 * 0.35), 'w2': round(255 * 0.35)})

  def test_temporary_bar_colors_and_inactive_grey(self):
    self.hud._distance_icon_parts = [('w0', 26, 119), ('w1', 22, 132), ('w2', 18, 147)]
    self.hud._distance_green_parts = [('g0', 12, 105), ('g1', 8, 118), ('g2', 4, 133)]
    for personality, last_active in ((log.LongitudinalPersonality.aggressive, 0),
                                     (log.LongitudinalPersonality.standard, 1),
                                     (log.LongitudinalPersonality.relaxed, 2)):
      self.sm['selfdriveState'].personality = personality
      for highlight in (1.0, 0.5, 0.0):
        with patch.object(self.hud, '_distance_highlight_alpha', return_value=highlight), \
             patch('openpilot.selfdrive.ui.mici.onroad.hud_renderer.rl.draw_texture_ex') as draw:
          self.hud._draw_distance_bars(self.rect)
        alphas = {call.args[0]: call.args[4].a for call in draw.call_args_list}
        for index in range(3):
          expected = 0.35
          if index < last_active:
            expected = 0.35 * (1 - highlight) + 0.9 * highlight
          elif index == last_active:
            expected = 0.35 * (1 - highlight)
          self.assertEqual(alphas[f'w{index}'], round(255 * expected))
        greens = {key: value for key, value in alphas.items() if key.startswith('g')}
        self.assertEqual(greens, {f'g{last_active}': round(255 * highlight)} if highlight else {})


if __name__ == '__main__':
  unittest.main()
