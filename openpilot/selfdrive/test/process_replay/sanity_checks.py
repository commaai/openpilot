from collections import deque
import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.selfdrive.test.process_replay.process_replay import ProcessConfig, check_most_messages_valid
from openpilot.tools.lib.logreader import LogIterable


def check_accel_rms(inputs: LogIterable, outputs: LogIterable) -> int:
  CP = next(m.carParams for m in inputs if m.which() == 'carParams')
  if not CP.openpilotLongitudinalControl:
    return 0

  services = ('carState', 'carControl', 'controlsState', 'radarState')
  msgs = [m for m in migrate_all(inputs) if m.which() in services] + list(outputs)
  n = round(1 / DT_MDL)
  window = deque(maxlen=4 * n)
  state, checked = {}, 0
  for m in sorted(msgs, key=lambda m: m.logMonoTime):
    if m.which() in services:
      state[m.which()] = getattr(m, m.which())
    elif m.which() == 'longitudinalPlan' and len(state) == len(services):
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

      # Allow one second to settle, then remove slow acceleration changes with a one-second moving average.
      times, accel = times[n:], accel[n:]
      noise = accel[n//2:-n//2] - np.convolve(accel, np.ones(n + 1) / (n + 1), mode='valid')
      accel_rms = np.sqrt(np.mean(noise ** 2))
      jerk_rms = np.sqrt(np.mean((np.diff(noise) / np.diff(times[n//2:-n//2])) ** 2))
      assert accel_rms < 0.012, f'{CP.carFingerprint}: {accel_rms=:.5f}'
      assert jerk_rms < 0.03, f'{CP.carFingerprint}: {jerk_rms=:.5f}'
      checked += 1
  return checked


PROCESS_CHECKS = {
  'plannerd': [check_accel_rms],
}


def check_outputs(cfg: ProcessConfig, inputs: LogIterable, outputs: LogIterable) -> dict[str, int]:
  assert check_most_messages_valid(outputs), 'Route did not have enough valid messages'

  # Skip the topic check if the segment is using qcom GPS.
  if cfg.proc_name != 'ubloxd' or any(m.which() in cfg.pubs for m in inputs):
    seen, expected = {m.which() for m in outputs}, set(cfg.subs)
    assert seen == expected, f'Expected messages: {expected}, but got: {seen}'

  return {check.__name__: check(inputs, outputs) for check in PROCESS_CHECKS.get(cfg.proc_name, [])}
