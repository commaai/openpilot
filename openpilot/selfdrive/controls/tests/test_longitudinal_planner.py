import numpy as np

from opendbc.car.toyota.values import CAR
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.parameterized import parameterized
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.test.process_replay import replay_process_with_name


class TestLongitudinalPlanner(OpenpilotTestCase):
  @parameterized.expand([17.2, 17.8])
  def test_cruise_accel_rms(self, noise_frequency):
    v_cruise = 125 * CV.KPH_TO_MS
    messages = []
    for frame in range(round(40 / DT_CTRL)):
      t = frame * DT_CTRL
      speed_error = 0.0 if t < 20 else 0.5
      noise = 0.03 * np.sin(2 * np.pi * noise_frequency * t)
      values = {
        'carState': {'vEgo': float(v_cruise - speed_error + noise), 'vCruise': 125.0},
        'carControl': {'enabled': True, 'longActive': True},
        'controlsState': {'longControlState': 'pid'},
        'selfdriveState': {'enabled': True, 'active': True, 'personality': 'standard'},
      }
      if frame % round(DT_MDL / DT_CTRL) == 0:
        values['radarState'] = {}
        values['modelV2'] = {'action': {'desiredAcceleration': 1.0}}
      for service, data in values.items():
        messages.append(log.Event.new_message(logMonoTime=1_000_000_000 + round(t * 1e9), valid=True, **{service: data}).as_reader())

    # Wheel-speed ripple aliases into 2.2/2.8 Hz cruise requests at the 20 Hz planner rate.
    replay = replay_process_with_name('plannerd', messages, fingerprint=CAR.TOYOTA_COROLLA_TSS2, disable_progress=True)
    plans = [m for m in replay if m.which() == 'longitudinalPlan']
    self.assertEqual(len(plans), round(40 / DT_MDL))
    for start, speed_error in [(10, 0.0), (30, 0.5)]:
      with self.subTest(speed_error=speed_error):
        window = [m for m in plans if start <= m.longitudinalPlan.modelMonoTime / 1e9 - 1 < start + 10]
        self.assertEqual(len(window), round(10 / DT_MDL))
        self.assertTrue(all(m.valid and m.longitudinalPlan.longitudinalPlanSource == 'cruise' for m in window))
        accel = np.array([m.longitudinalPlan.aTarget for m in window])
        self.assertAlmostEqual(float(np.mean(accel)), speed_error, delta=0.02)
        accel_rms = float(np.sqrt(np.mean((accel - np.mean(accel)) ** 2)))
        jerk_rms = float(np.sqrt(np.mean((np.diff(accel) / DT_MDL) ** 2)))
        self.assertLess(accel_rms, 0.002, f'{accel_rms=:.5f} m/s²')
        self.assertLess(jerk_rms, 0.025, f'{jerk_rms=:.5f} m/s³')
