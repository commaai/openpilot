import numpy as np

from openpilot.tools.lib.logreader import LogIterable


def check_jerk_rms(inputs: LogIterable, outputs: LogIterable) -> None:
  msgs = [m for m in inputs if m.which() == 'carControl']
  msgs += [m for m in outputs if m.which() == 'longitudinalPlan']
  active = False
  prev = None
  jerks = []
  for m in sorted(msgs, key=lambda m: m.logMonoTime):
    if m.which() == 'carControl':
      active = m.carControl.longActive
      if not active:
        prev = None
      continue
    plan = m.longitudinalPlan
    if not (m.valid and active and plan.longitudinalPlanSource == 'cruise' and not plan.hasLead):
      prev = None
      continue
    if prev is not None:
      jerks.append((plan.aTarget - prev.aTarget) / ((plan.modelMonoTime - prev.modelMonoTime) / 1e9))
    prev = plan

  if jerks:
    jerk_rms = np.sqrt(np.mean(np.square(jerks)))
    assert jerk_rms < 0.15, f'{jerk_rms=:.5f}'


PROCESS_CHECKS = {
  'plannerd': [check_jerk_rms],
}
