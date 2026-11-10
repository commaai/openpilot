from collections import deque
import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.tools.lib.logreader import LogIterable


def check_jerk_rms(inputs: LogIterable, outputs: LogIterable) -> None:
  services = ('carState', 'controlsState')
  msgs = [m for m in migrate_all(inputs) if m.which() in services]
  msgs += [m for m in outputs if m.which() == 'longitudinalPlan']
  window_size = round(4 / DT_MDL)
  window = deque(maxlen=window_size)
  state = {}
  for m in sorted(msgs, key=lambda m: m.logMonoTime):
    if m.which() in services:
      state[m.which()] = getattr(m, m.which())
      continue
    if len(state) != len(services):
      continue

    cs, controls = (state[k] for k in services)
    plan = m.longitudinalPlan
    if not (m.valid and controls.longControlState == 'pid' and cs.vEgo > 20 and
            plan.longitudinalPlanSource == 'cruise' and not plan.hasLead and
            not (cs.gasPressed or cs.brakePressed or controls.forceDecel)):
      window.clear()
      continue

    window.append((plan.modelMonoTime / 1e9, cs.vCruise, plan.aTarget))
    if len(window) < window_size:
      continue
    times, cruise, accel = np.array(window).T
    if np.ptp(cruise) > 0.01 or np.max(np.diff(times)) > 2 * DT_MDL:
      continue
    # Allow one second to settle.
    times, accel = times[round(1 / DT_MDL):], accel[round(1 / DT_MDL):]
    jerk = np.diff(accel) / np.diff(times)
    jerk_rms = np.sqrt(np.mean(jerk ** 2))
    assert jerk_rms < 0.1, f'{jerk_rms=:.5f}'


PROCESS_CHECKS = {
  'plannerd': [check_jerk_rms],
}
