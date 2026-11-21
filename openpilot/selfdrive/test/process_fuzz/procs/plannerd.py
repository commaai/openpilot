import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, J_CRUISE_BP
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P
from openpilot.selfdrive.test.process_fuzz.harness import Spec

LEDGER = U.upstream('modelV2.velocity', 'selfdriveState.personality') | {
  **{f'modelV2.{p}.len': ('enforced', U.MODELD, 'modeld: list lengths')
     for p in ('position.t', 'orientationRate.z', 'meta.desirePrediction')},
  'modelV2.position.t[]': ('enforced', 'fill_model_msg.py:90 t = ModelConstants.T_IDXS', 'modeld: time axes'),
  'CP.brand': ('symmetry', 'only logged, plannerd.py:17', None),
}

LMPC = 'selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py:'
UNREACHED = {
  U.H_LOCKSTEP: ['selfdrive/controls/plannerd.py:main:"if sm.updated[\'modelV2\']:":F'],
  'dead: plannerd always passes CONTROL_N speeds and CONTROL_N_T_IDX (longitudinal_planner.py:73,91)':
    ['selfdrive/controls/lib/drive_helpers.py:get_accel_from_plan:"if len(speeds) == len(t_idxs):":F'],
  'dead: tau=1.0 (longitudinal_planner.py:25)': ['selfdrive/controls/lib/drive_helpers.py:smooth_value:"alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1":F'],
  'enforced modelV2.meta.desirePrediction.len':
    ['selfdrive/controls/lib/ldw.py:LaneDepartureWarning.update:"if len(desire_prediction) and ldw_allowed:":F'],
  'enforced selfdriveState.personality:outofrange': [LMPC + '*:"elif personality==log.LongitudinalPersonality.aggressive:":F'],
  'gap: needs a second solver failure within 5 wall-clock s of the first; none in 6k deep examples':
    [LMPC + 'LongitudinalMpc.run:"if t > self.last_cloudlog_t + 5.0:":F'],
}

KNOWN = {
  'no_chatter longitudinalPlan.shouldStop': ('real', 'should_stop() has no hysteresis (drive_helpers.py:16): vEgo or an accel candidate hovering '
                                             + 'at 0.3/0.1 toggles shouldStop, LongControl state and resume (controlsd.py:163)', 'known_should_stop.json'),
  'hf_gain carState.vEgo -> longitudinalPlan.aTarget': ('real', 'a present lead inside STOP_DISTANCE: the MPC amplifies vEgo noise into aTarget '
                                                        + '(gain ~1 at dRel 0-2 m, long_mpc.py:process_lead)', 'known_hf_gain_close_lead.json'),
  **{f'no_chatter driverAssistance.{s}LaneDeparture': ('real', 'no hysteresis on ldw_allowed or the lane thresholds (ldw.py:21-35): '
                                                       + 'vEgo at LDW_MIN_SPEED flickers the warning', 'known_ldw_chatter.json') for s in ('left', 'right')},
}


def actuated(r):
  # aTarget is actuated: openpilot long, long control on, cruise set, for 1 s (a reset restarts the plan once, longitudinal_planner.py:54-61)
  t, _ = P.series(r.outs, 'longitudinalPlan.aTarget')
  if not r.cp.openpilotLongitudinalControl:
    return np.zeros(len(t), dtype=bool)
  on = [s is not None and s.raw != 0 for s in P.held(r.ins, 'controlsState.longControlState', t)]
  cruise = [v is not None and v != V_CRUISE_UNSET for v in P.held(r.ins, 'carState.vCruise', t)]
  return P.held_for(t, np.array(on, dtype=bool) & np.array(cruise, dtype=bool), 1.)


def actuated_no_close_lead(r):
  t, _ = P.series(r.outs, 'longitudinalPlan.aTarget')
  close = np.zeros(len(t), dtype=bool)
  for lead in ('leadOne', 'leadTwo'):
    close |= np.array([ld is not None and ld.present and not ld.dRel >= STOP_DISTANCE for ld in P.held(r.ins, f'radarState.{lead}', t)], dtype=bool)
  return actuated(r) & P.held_for(t, ~close, 1.)  # the MPC carries a close lead's effect over a few cycles


SPEC = Spec(
  proc='plannerd',
  scope={
    'selfdrive/controls/plannerd.py': None,
    'selfdrive/controls/lib/longitudinal_planner.py': None,
    'selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py': {'LongitudinalMpc', 'get_jerk_factor', 'get_T_FOLLOW'},
    'selfdrive/controls/lib/curve_speed.py': None,
    'selfdrive/controls/lib/ldw.py': {'LaneDepartureWarning.update'},
    'selfdrive/controls/lib/drive_helpers.py': {'should_stop', 'smooth_value', 'get_accel_from_plan'},
  },
  ledger=LEDGER,
  known=KNOWN,
  unreached=UNREACHED,
  template=U.shape,
  bounds={
    'carState.vEgo': [0.3, *J_CRUISE_BP],  # aliased as v_ego: should_stop, J_CRUISE_BP
    'modelV2.action.desiredAcceleration': [0.1, ACCEL_MIN, ACCEL_MAX, A_CRUISE_MIN],
  },
  relations={'carState.vEgo': [('carState.vCruise', CV.KPH_TO_MS)]},  # v_cruise - v_ego, longitudinal_planner.py:25
  checks=[
    P.alive('longitudinalPlan'), P.alive('driverAssistance'),
    P.finite('longitudinalPlan'),  # driverAssistance has only bools
    P.bounded('longitudinalPlan.aTarget', ACCEL_MIN, ACCEL_MAX),
    P.invariant('longitudinalPlan', lambda p: len(p.speeds) == len(p.accels) == len(p.jerks) == CONTROL_N, 'plan length'),
    P.valid_from_inputs('longitudinalPlan'), P.valid_from_inputs('driverAssistance'),
    P.hf_gain('carState.vEgo', 'longitudinalPlan.aTarget', max_gain=0.2, when=actuated),
    P.hf_gain('carState.vEgo', 'longitudinalPlan.aTarget', max_gain=0.2, when=actuated_no_close_lead, name=' (no lead inside STOP_DISTANCE)'),
    P.no_chatter('longitudinalPlan.shouldStop', 1.),
    P.no_chatter('driverAssistance.leftLaneDeparture', 1.), P.no_chatter('driverAssistance.rightLaneDeparture', 1.),
  ],
)
