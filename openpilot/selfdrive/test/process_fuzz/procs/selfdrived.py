import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.realtime import DT_CTRL
from opendbc.car import structs
from opendbc.car.toyota.values import ToyotaFlags
from openpilot.selfdrive.selfdrived.state import ACTIVE_STATES, ENABLED_STATES, SOFT_DISABLE_TIME
from openpilot.selfdrive.test.process_fuzz import harness as H, procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

State = log.SelfdriveState.OpenpilotState
DISABLES = ('userDisable', 'softDisable', 'immediateDisable')
BIG_MODEL_WARMUP = 5.  # s, selfdrived.py:387
IGNORED_SAFETY = (int(structs.CarParams.SafetyModel.silent), int(structs.CarParams.SafetyModel.noOutput))  # raw: pandad may send any

LEDGER = U.upstream('vehicleParameters.roll', 'extrinsicsCalibration.rpyCalib.len') | {
  'P LongitudinalPersonality:outofrange': ('enforced', 'writers: ui toggles.py:183, mici driving.py:17, selfdrived.py:468', None),
  'CP.carFingerprint': ('symmetry', 'only selects interfaces[...].DRIVABLE_GEARS, a per-brand constant (car_events.py:101,107)', None),
  'CP.safetyConfigs[].safetyModel': ('symmetry', 'only compared for equality with pandaStates (selfdrived.py:336); '
                                     + 'the template is the equal class, genes the unequal ones', None),
  'CP.safetyConfigs[].safetyParam': ('symmetry', 'only compared for equality with pandaStates (selfdrived.py:337)', None),
  'P UbloxAvailable': ('symmetry', 'picks gpsLocation vs gpsLocationExternal (common/gps.py), both meta-only (recv_frame, :453)', None),
  'pandaStates[].faults[]:outofrange': ('harness', 'pycapnp cannot write an out-of-schema value into an enum list', None),
  'radarState.radarErrors.*': ('harness', 'any(...to_dict().values()) (selfdrived.py:377); space.py expands and fuzzes every field', None),
}

SD, EV, CE = 'selfdrive/selfdrived/selfdrived.py:SelfdriveD.', 'selfdrive/selfdrived/events.py:', 'selfdrive/car/car_events.py:CarEvents.update:'
STARTUP = SD + '__init__:"self.startup_event = EventName.startup if build_metadata.openpilot.comma_remote and build_metadata.tested_channel ' \
  + 'else EventName.startupMaster"'
UNREACHED = {
  'dead: every non-dashcam openpilot-long Toyota has minEnableSpeed <= 0, so vEgo < minEnableSpeed implies vEgo < 0.001':
    [CE + '"if CS.vEgo < 0.001:":F'],
  'dead: entries only come from add_many, which always sets an alert (alertmanager.py:43-46)':
    ['selfdrive/selfdrived/alertmanager.py:AlertManager.process_alerts:"if not v.alert:":F'],
  'dead: never called, process_alerts compares (priority, start_frame) tuples (alertmanager.py:63)': [EV + 'Alert.__gt__:*'],
  'dead: selfdrived always passes callback args (selfdrived.py update_alerts)': [EV + 'Events.create_alerts:"if callback_args is None:":None'],
  'env: HARDWARE is a PC, not a mici': ['selfdrive/selfdrived/*:"if HARDWARE.get_device_type() == \'mici\':":T',
                                        EV + 'StartupAlert.__init__:"if alert_text_2 == *'],
  'dead: rpyCalib.len is always 3 (ledger)': [EV + 'calibration_invalid_alert:*len(rpy) == 3 else math.nan)":F'],
  'dead: all three cameras are SubMaster services (selfdrived.py:80,96)':
    [EV + 'camera_malfunction_alert:"bad_cams = [s.replace(\'State\', \'\') for s in all_cams if s in sm.data.keys() and not sm.all_checks([s, ])]"#1:F'],
  'dead: car_recognized is True here, False is the if before (selfdrived.py:143)': [SD + '__init__:"elif car_recognized and self.CP.passive:":F'],
  'harness: main() builds SelfdriveD() without a CP (selfdrived.py:595)': [SD + '__init__:"if CP is None:":notNone'],
  'env: build_metadata of this checkout': [STARTUP + '#1:F', STARTUP + '#1:T', STARTUP + ':T'],
  'env: no camerad VisionIPC server in the harness, so no streams': [SD + 'data_sample:"if VisionStreamType.*not in available_streams:":F'],
  'harness: Genome.sim sets SIMULATION and REPLAY together (process replay); SIMULATION alone is the MetaDrive sim':
    [SD + 'data_sample:"if all_valid or timed_out or (SIMULATION and not REPLAY):"#3:F', SD + 'update_events:"if not SIMULATION or REPLAY:"#1:F',
     SD + 'update_events:"not TESTING_CLOSET and (not SIMULATION or REPLAY)):"#2:F'],
  'harness: the loop ends only on process exit and replay kills selfdrived': [SD + 'params_thread:"while not evt.is_set():"*T'],
  'gap: params are fixed for a run, ChestnutLoading never falls while selfdrived runs':
    [SD + 'update_events:"if self.big_model_loading and not loading:"#1:F'],
  'dead: update_events returns while not initialized (selfdrived.py:196-198)':
    [SD + 'update_events:"if self.sm.frame > int(2. / DT_CTRL) and self.initialized:"#1:F'],
  'env: TESTING_CLOSET env var (selfdrived.py:31)': [SD + 'update_events:"not TESTING_CLOSET and (not SIMULATION or REPLAY)):":T'],
  'dead: the elif before handles timer > 0': ['selfdrive/selfdrived/state.py:StateMachine.update:"elif self.soft_disable_timer <= 0:":F'],
  'dead: the outer if handles state != disabled': ['selfdrive/selfdrived/state.py:StateMachine.update:"elif self.state == State.disabled:":F'],
  'dead: the non-disabled states are enabled, softDisabling, preEnabled, overriding':
    ['selfdrive/selfdrived/state.py:StateMachine.update:"elif self.state == State.overriding:":F'],
}

KNOWN = {
  'soft_disable_budget': ('real', 'leaving softDisabling and re-entering rearms the 3 s timer: a flickering soft disable never '
                          + 'disengages (state.py:37-50)', 'known_soft_disable_flicker.json'),
  'engaged_from_inputs (big model settling)': ('theoretical', 'a ChestnutLoading/bigModelFailed edge drops commIssue for 5 s also '
                                               + 'while engaged (selfdrived.py:161-174,387)', 'known_big_model_settling.json'),
  'personality': ('theoretical', 'params_thread writes LongitudinalPersonality unlocked (selfdrived.py:577) over a press (:468): '
                  + 'press reverts or is lost', 'known_personality_params_race.json'),
}


def car_params(g):
  # CarParams as card writes them, passive and its safety config included (card.py:111-134)
  CP = S.get_car_params(g.fp, g.alpha_long, g.fpv)
  CP.alternativeExperience = 0
  CP.passive = S.interfaces[g.fp].CarController is None or not g.params.get('OpenpilotEnabledToggle', True) or CP.dashcamOnly
  if CP.passive:
    sc = structs.CarParams.SafetyConfig()
    sc.safetyModel = structs.CarParams.SafetyModel.noOutput
    CP.safetyConfigs = [sc]
  key = g.params.get('SecOCKey')  # a non-hex key kills card, see card's KNOWN
  CP.secOcKeyAvailable = bool(CP.secOcRequired and key is not None and len(bytes.fromhex(key)) == 16)
  return CP


def template(svc, g=None):
  # producers' nominal output; every read field is still fuzzed by genes over its whole type
  m = S.new_message(svc)
  if svc == 'carState':  # card: CAN valid, cruise on, in drive
    cs = m.carState
    cs.canValid = cs.cruiseState.available = cs.cruiseState.enabled = True
    cs.gearShifter = 'drive'
    cs.vEgo = 20.
  elif svc == 'pandaStates':  # pandad: one panda per safety config, in the car's safety mode
    CP = car_params(g)
    m = messaging.new_message(svc, len(CP.safetyConfigs))
    for ps, sc in zip(m.pandaStates, CP.safetyConfigs, strict=True):
      ps.safetyModel, ps.safetyParam, ps.alternativeExperience = sc.safetyModel, sc.safetyParam, CP.alternativeExperience
      ps.controlsAllowed = True
  elif svc == 'extrinsicsCalibration':
    m.extrinsicsCalibration.calStatus = 'calibrated'
    m.extrinsicsCalibration.rpyCalib = [0., 0., 0.]
  elif svc == 'deviceMotion':
    m.deviceMotion.posenetOK = m.deviceMotion.inputsOK = True
  elif svc == 'vehicleParameters':
    vp = m.vehicleParameters
    vp.valid = vp.angleOffsetValid = vp.steerRatioValid = vp.stiffnessFactorValid = True
  elif svc == 'deviceState':
    m.deviceState.freeSpacePercent, m.deviceState.memoryUsagePercent = 50., 50
  elif svc == 'managerState':
    m.managerState.init('processes', 2)
    for p, n in zip(m.managerState.processes, ('controlsd', 'modeld'), strict=True):
      p.name, p.running, p.shouldBeRunning = n, True, True
  elif svc == 'modelV2':
    m.modelV2.velocity.x = [0.] * 33
  return m


def cp_class(CP):
  # what selfdrived branches on per fingerprint (car_events.py, selfdrived.py); safetyModel/Param only by equality
  hybrid = CP.brand == 'toyota' and bool(CP.flags & ToyotaFlags.HYBRID.value)
  return (CP.brand, CP.dashcamOnly, S.interfaces[CP.carFingerprint].CarController is None, CP.notCar,
          CP.pcmCruise, CP.openpilotLongitudinalControl, CP.minEnableSpeed, CP.minSteerSpeed,
          str(CP.networkLocation), hybrid, CP.secOcRequired, CP.secOcKeyAvailable, CP.alphaLongitudinalAvailable, len(CP.safetyConfigs),
          CP.alternativeExperience)


# *** properties: the ET contract of the state machine (events.py ET, state.py), over (prev state, this cycle's events) ***

FLAGS = ('enable', 'noEntry', 'preEnable', 'overrideLateral', 'overrideLongitudinal', 'userDisable', 'softDisable', 'immediateDisable')


def steps(r):
  # [(t, selfdriveState, {(event name, flag)} of the same step, or None if unknown)]. Each step sends one selfdriveState and,
  # on change or every 1 s, onroadEvents; outputs drained together share a stamp, selfdriveStates first (process_replay.py:270)
  if 'steps' in r.cache:
    return r.cache['steps']
  out, ev, groups = [], set(), {}
  for m in r.outs:
    groups.setdefault(m.logMonoTime, ([], []))[m.which() == 'onroadEvents'].append(m)
  for _, (ss, oe) in sorted(groups.items()):
    if oe and len(oe) != len(ss):  # can't tell which step sent which onroadEvents
      out += [(P.t_of(m), m.selfdriveState, None) for m in ss]
      ev = None
      continue
    for k, m in enumerate(ss):
      if oe:
        ev = {(str(e.name), f) for e in oe[k].onroadEvents for f in FLAGS if getattr(e, f)} | {(str(e.name), '') for e in oe[k].onroadEvents}
      out.append((P.t_of(m), m.selfdriveState, ev))
  r.cache['steps'] = out
  return out


def has(ev, flag):
  return any(f == flag for _, f in ev)


def names(ev):
  return {n for n, f in ev if f == ''}


def first_frame(r):
  # process replay steps on the very first message (MessageBasedRcvCallback first_frame): without a carState if it isn't one
  return int(bool(r.ins) and r.ins[0].which() != 'carState')


def per_step(r):
  # [(Cycle, events)]: each carState cycle with its selfdriveState and the events of that step
  return list(zip(P.cycles(r, ('selfdriveState',), 'carState', skip=first_frame(r)), [ev for _, _, ev in steps(r)[first_frame(r):]], strict=False))


def state_contract(r):
  # each state is only held while its event type is present, and only entered/left through the transitions events allow
  cs = steps(r)
  passive = car_params(r.genome).passive
  init = next((k for k, (_, _, ev) in enumerate(cs) if ev is not None and 'selfdriveInitializing' not in names(ev)), None)
  for k in range(1, len(cs)):
    t, ss, ev = cs[k]
    prev, s = cs[k - 1][1].state, ss.state
    if ev is None:
      continue
    # a passive car never steps the state machine (selfdrived.py:564), REPLAY's init jump still sets its state (:490-491)
    if not passive:
      assert ss.enabled == (s in ENABLED_STATES) and ss.active == (s in ACTIVE_STATES), f't={t:.2f}s {s} enabled={ss.enabled} active={ss.active}'
    assert ss.engageable == (not has(ev, 'noEntry')), f't={t:.2f}s engageable={ss.engageable} with {sorted(names(ev))}'
    for st, flags in ((State.softDisabling, ('softDisable',)), (State.preEnabled, ('preEnable',)),
                      (State.overriding, ('overrideLateral', 'overrideLongitudinal'))):
      assert s != st or any(has(ev, f) for f in flags), f't={t:.2f}s {s} without {flags}: {sorted(names(ev))}'
    if prev == State.disabled and s != State.disabled and k != init:  # REPLAY jumps to enabled at init (selfdrived.py:490)
      assert has(ev, 'enable') and not has(ev, 'noEntry'), f't={t:.2f}s engaged without enable or with noEntry: {sorted(names(ev))}'
    if prev in ACTIVE_STATES and s not in ACTIVE_STATES:
      timer = prev == State.softDisabling and soft_held(cs, k) >= int(SOFT_DISABLE_TIME / DT_CTRL) - 1
      assert has(ev, 'immediateDisable') or has(ev, 'userDisable') or timer, f't={t:.2f}s {prev} -> {s} without a disable: {sorted(names(ev))}'
  return 'checked' if any(c[1].enabled for c in cs) else None


def soft_held(cs, k):
  n = 0
  while k - 1 - n >= 0 and cs[k - 1 - n][1].state == State.softDisabling:
    n += 1
  return n


def enable_with_disable(r):
  # engaging in a cycle whose own events disable: a disable event without noEntry lets the state machine engage on it
  cs = steps(r)
  for k in range(1, len(cs)):
    t, ss, ev = cs[k]
    if ev is not None and cs[k - 1][1].state == State.disabled and ss.state != State.disabled and has(ev, 'enable'):
      bad = sorted(n for n, f in ev if f in DISABLES)
      assert not bad, f't={t:.2f}s engaged with disable events {bad}'


def soft_disable_budget(r):
  # a soft disable condition keeps openpilot engaged for SOFT_DISABLE_TIME at most, also when it flickers:
  # within any 2 * SOFT_DISABLE_TIME window of one engagement, softDisabling frames <= SOFT_DISABLE_TIME
  cs = steps(r)
  n = int(SOFT_DISABLE_TIME / DT_CTRL)
  soft = np.array([c[1].state == State.softDisabling for c in cs], dtype=int)
  start = 0
  for k in range(len(cs)):
    if not cs[k][1].enabled:
      start = k + 1
      continue
    lo = max(start, k - 2 * n + 1)
    tot = soft[lo:k + 1].sum()
    assert tot <= n + 1, f't={cs[k][0]:.2f}s {tot} softDisabling frames in the last {(k - lo + 1) * DT_CTRL:.1f}s, still engaged'


IGNORED = {'accelerometer', 'gyroscope', 'gpsLocation', 'gpsLocationExternal', 'alertDebug', 'lateralManeuverPlan',  # selfdrived.py:85
           'narrowRoadCameraState', 'wideRoadCameraState'}  # REPLAY, or no VIPC stream after init (selfdrived.py:88-90,482-488)
IGNORED_SIM = {'cabinCameraState', 'managerState'}  # SIMULATION, selfdrived.py:86-87


def engaged_from_inputs(settling):
  # actively engaged (not soft disabling) => every checked input seen, valid and (sim=False) alive per SubMaster (an on-demand
  # service only valid once seen); carState's validity is canValid (card.py:206). While the big model loads and for 5 s after it
  # loads or fails, selfdrived drops commIssue on purpose (selfdrived.py:388-389): that window is its own (KNOWN) check
  @P.prop('engaged_from_inputs' + (' (big model settling)' if settling else ''))
  def check(r):
    ignored = IGNORED | (IGNORED_SIM if r.genome.sim else set()) | {'carState'}
    settle_t, out = -1e9, None
    for c, ev in per_step(r):
      ss, t = c.out['selfdriveState'].selfdriveState, c.t
      if ev is not None and names(ev) & {'bigModelLoading', 'bigModelFailed'}:
        settle_t = t
      if not ss.active or ss.state == State.softDisabling or (t - settle_t <= BIG_MODEL_WARMUP + 0.02) != settling:
        continue
      out = 'checked'
      assert c.trig.carState.canValid, f't={t:.2f}s engaged with carState.canValid False'
      for s in sorted(set(r.space.pubs) - ignored):
        f = SERVICE_LIST[s].frequency
        if f <= 1e-5 and s not in c.last:
          continue
        assert s in c.last, f't={t:.2f}s engaged before any {s}'
        assert c.last[s].valid, f't={t:.2f}s engaged with {s} invalid'
        age = (c.trig.logMonoTime - c.recv[s]) / 1e9
        assert r.genome.sim or f <= 1e-5 or age < 10. / f + 0.02, f't={t:.2f}s engaged with {s} {age:.2f}s old'
    return out
  return check


def mismatch_from_current(r):
  # controlsMismatch (an immediate disable) only while the pandas currently disagree: a non-silent panda without
  # controlsAllowed, a safety mode mismatch, or safetyRxChecksInvalid (selfdrived.py:333-347,505-515: "two samples")
  CP = car_params(r.genome)
  cfg = list(CP.safetyConfigs)
  for c, ev in per_step(r):
    if ev is None or 'controlsMismatch' not in names(ev) or 'pandaStates' not in c.last:
      continue
    ps = c.last['pandaStates'].pandaStates
    disagree = any(p.safetyRxChecksInvalid or (p.safetyModel.raw not in IGNORED_SAFETY and not p.controlsAllowed) or
                   (k < len(cfg) and (p.safetyModel.raw != cfg[k].safetyModel.raw or p.safetyParam != cfg[k].safetyParam or
                                      p.alternativeExperience != CP.alternativeExperience)) or
                   (k >= len(cfg) and p.safetyModel.raw not in IGNORED_SAFETY) for k, p in enumerate(ps))
    assert disagree, f't={c.t:.2f}s controlsMismatch while every panda agrees: ' + str([(p.safetyModel.raw, p.controlsAllowed) for p in ps])


def passive_never_enabled(r):
  # a passive car (dashcam, no controller, openpilot toggled off) never engages: card's carControl relies on it
  if car_params(r.genome).passive:
    assert not any(c[1].enabled for c in steps(r)), 'passive car engaged'
    return 'checked'


def personality_range(r):
  for t, ss, _ in steps(r):
    assert ss.personality.raw in (0, 1, 2), f't={t:.2f}s personality {ss.personality.raw}'


def personality(r):
  # changes only on a personalityChanged cycle, by -1 mod 3 (selfdrived.py:466-470)
  cs = steps(r)
  for k in range(1, len(cs)):
    t, ss, ev = cs[k]
    a, b = cs[k - 1][1].personality.raw, ss.personality.raw
    if a != b and ev is not None:
      assert 'personalityChanged' in names(ev) and b == (a - 1) % 3, f't={t:.2f}s personality {a} -> {b} with {sorted(names(ev))}'


def excessive_latch(r):
  # once excessiveActuation fires it stays for the rest of the drive (selfdrived.py:310-315)
  seen = False
  for t, _, ev in steps(r):
    if ev is None:
      continue
    seen |= 'excessiveActuation' in names(ev)
    assert not seen or 'excessiveActuation' in names(ev), f't={t:.2f}s excessiveActuation cleared'


SPEC = Spec(
  proc='selfdrived',
  scope=dict.fromkeys(['selfdrive/selfdrived/selfdrived.py', 'selfdrive/selfdrived/state.py', 'selfdrive/selfdrived/alertmanager.py',
                       'selfdrive/selfdrived/helpers.py', 'selfdrive/selfdrived/events.py', 'selfdrive/car/car_events.py']),
  ledger=LEDGER,
  known=KNOWN,
  unreached=UNREACHED,
  template=template,
  duration=16.,
  examples=16,
  few=10 / 11,  # engaging needs ~30 conditions at once: most draws perturb a few fields of the nominal template
  optional_pubs=('userBookmark', 'lateralManeuverPlan'),
  log_clock=30.,  # selfdrived starts well after boot: from 0 it would sit in the 5 s big model warmup (selfdrived.py:72,387)
  cp_class=cp_class,
  init=lambda g: H.car_params_callback(car_params(g)),
  params={
    'IsMetric': [False, True], 'IsLdwEnabled': [False, True], 'DisengageOnAccelerator': [False, True],
    'OpenpilotEnabledToggle': [False, True, True, True], 'LongitudinalPersonality': [0, 1, 2],
    'ChestnutActive': [None] * 8 + [False, True], 'ChestnutLoading': [None] * 8 + [False, True], 'PrimeType': [None, -1, 0, 1, 5],
    'Offroad_ExcessiveActuation': [None] * 9 + [{'text': 'x', 'extra': 'longitudinal'}],
    'SecOCKey': [None, None, '0123456789abcdef' * 2, 'ab' * 15],  # card's input, only through CP.secOcKeyAvailable
  },
  checks=[P.alive('selfdriveState')] + [P.prop(f.__name__)(f) for f in (
    state_contract, enable_with_disable, soft_disable_budget, mismatch_from_current, passive_never_enabled, personality_range,
    personality, excessive_latch)] + [engaged_from_inputs(False), engaged_from_inputs(True),
    # a blank banner from alertDebug's own empty text is cosmetic
    P.invariant('selfdriveState', lambda ss: ss.alertSize != 'none' or not (ss.alertText1 or ss.alertText2), 'alert text has a size')],
)
