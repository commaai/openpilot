import numpy as np

from openpilot.tools.lib.logreader import LogIterable


def check_jerk_rms(inputs: LogIterable, outputs: LogIterable) -> int:
  replay_plans = {m.longitudinalPlan.modelMonoTime: m.longitudinalPlan for m in outputs if m.which() == 'longitudinalPlan' and m.valid}
  msgs = (m for m in inputs if m.which() in ('carControl', 'longitudinalPlan'))
  active = False
  prev = None
  jerks = []
  for m in sorted(msgs, key=lambda m: m.logMonoTime):
    if m.which() == 'carControl':
      active = m.carControl.longActive
      if not active:
        prev = None
      continue
    recorded = m.longitudinalPlan
    if not (m.valid and active and recorded.longitudinalPlanSource == 'cruise' and not recorded.hasLead):
      prev = None
      continue
    plan = replay_plans.get(recorded.modelMonoTime)
    assert plan is not None, f'Missing valid replay plan at {recorded.modelMonoTime}'
    if prev is not None:
      jerks.append((plan.aTarget - prev.aTarget) / ((plan.modelMonoTime - prev.modelMonoTime) / 1e9))
    prev = plan

  if jerks:
    jerk_rms = np.sqrt(np.mean(np.square(jerks)))
    assert jerk_rms < 0.15, f'{jerk_rms=:.5f} over {len(jerks)} samples'
  return len(jerks)


PROCESS_CHECKS = {
  'plannerd': [check_jerk_rms],
}
