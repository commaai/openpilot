from collections import deque
import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.tools.lib.logreader import LogIterable


def cruise_windows(inputs: LogIterable, outputs: LogIterable):
  services = ('carState', 'carControl', 'controlsState', 'radarState')
  msgs = [m for m in migrate_all(inputs) if m.which() in services]
  msgs += [m for m in outputs if m.which() == 'longitudinalPlan']
  window = deque(maxlen=round(4 / DT_MDL))
  state = {}
  for m in sorted(msgs, key=lambda m: m.logMonoTime):
    if m.which() in services:
      state[m.which()] = getattr(m, m.which())
      continue
    if len(state) != len(services):
      continue

    cs, cc, controls, radar = (state[k] for k in services)
    plan = m.longitudinalPlan
    if not (m.valid and cc.longActive and controls.longControlState == 'pid' and cs.vEgo > 20 and
            plan.longitudinalPlanSource == 'cruise' and not (cs.gasPressed or cs.brakePressed or controls.forceDecel) and
            not (radar.leadOne.present or radar.leadTwo.present)):
      window.clear()
      continue

    window.append((plan.modelMonoTime / 1e9, cs.vCruise, plan.aTarget))
    if len(window) < window.maxlen:
      continue
    times, cruise, accel = np.array(window).T
    if np.ptp(cruise) > 0.01 or np.max(np.diff(times)) > 2 * DT_MDL:
      continue
    yield times, accel


def check_accel_rms(inputs: LogIterable, outputs: LogIterable) -> None:
  CP = next(m.carParams for m in inputs if m.which() == 'carParams')
  if not CP.openpilotLongitudinalControl:
    return

  one_second = round(1 / DT_MDL)
  edge = one_second // 2
  kernel = np.ones(one_second + 1) / (one_second + 1)
  for times, accel in cruise_windows(inputs, outputs):
    # Allow one second to settle, then measure variation around a one-second moving average.
    times, accel = times[one_second:], accel[one_second:]
    noise = accel[edge:-edge] - np.convolve(accel, kernel, mode='valid')
    jerk = np.diff(noise) / np.diff(times[edge:-edge])
    accel_rms = np.sqrt(np.mean(noise ** 2))
    jerk_rms = np.sqrt(np.mean(jerk ** 2))
    assert accel_rms < 0.012, f'{CP.carFingerprint}: {accel_rms=:.5f}'
    assert jerk_rms < 0.03, f'{CP.carFingerprint}: {jerk_rms=:.5f}'


PROCESS_CHECKS = {
  'plannerd': [check_accel_rms],
}
