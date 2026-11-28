import math
from collections import deque

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA, V_EGO_STATIONARY, _LEAD_ACCEL_TAU
from openpilot.selfdrive.test.process_fuzz import procs as U, properties as P, space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec

LEADS = ('leadOne', 'leadTwo')
RENAME = 0x2a5  # trackId bijection for the renaming property: id ^ RENAME
BIG_ID = 2 ** 31
THEO, BIG = 'theoretical inputs', 'trackId >= 2^31'  # tags

RADAR = 'radar interfaces decode bounded CAN signals (+ sin/cos/averages), e.g. hyundai/radar_interface.py:43-45, ford:229-241'
MODELD = 'modeld checks finiteness only on chestnut (modeld.py:222)'
LEDGER = U.upstream('carState.vEgo') | {
  'modelV2.leadsV3.len': ('enforced', U.MODELD, 'modeld: list lengths'),
  'modelV2.velocity.x.len': ('enforced', U.MODELD, 'modeld: list lengths'),
  # sigmoid: 1 / (1 + exp(min(-x, 11))) is in [1.7e-5, 1] or NaN; out-of-range finite values are still generated, the liveness
  # check gates on the range
  'modelV2.leadsV3[].prob:inf': ('enforced', 'sigmoid, parse_model_outputs.py:5-9,42 -> fill_model_msg.py:122', 'modeld: lead prob range'),
  'modelV2.leadsV3[].prob:nan|huge': ('theoretical', MODELD, 'modeld: finite'),
  **{f'modelV2.{f}:nan|inf|huge': ('theoretical', MODELD, 'modeld: finite')
     for f in ('velocity.x[]', 'leadsV3[].x[]', 'leadsV3[].y[]', 'leadsV3[].v[]', 'leadsV3[].a[]', 'leadsV3[].xStd[]', 'leadsV3[].yStd[]',
               'leadsV3[].vStd[]')},
  'carState.vEgo:huge': ('theoretical', 'card KF over CAN-decoded wheel speeds, opendbc/car/interfaces.py:302-311', 'card: finite carState'),
  'radarTracks.points[].dRel:nan|inf|huge': ('theoretical', RADAR, 'card: finite radarTracks'),
  'radarTracks.points[].yRel:nan|inf|huge': ('theoretical', RADAR, 'card: finite radarTracks'),
  'radarTracks.points[].vRel:nan|inf|huge': ('theoretical', RADAR, 'card: finite radarTracks'),
  'radarTracks.points[].trackId:int32': ('theoretical', 'ids are per-interface counters from 0 (toyota/radar_interface.py:63-64); '
                                         + '>= 2^31 needs 2^31 new tracks in one card lifetime', None),
  'radarTracks.errors.*': ('harness', 'copied verbatim (radard.py:225); space.py expands and fuzzes every field, checked by provenance', None),
}


def tag(sp, g, ins=None):
  # a trackId past the int32 radarTrackId, or a read float NaN/inf/huge (outside every producer's image: theoretical ledger)
  if ins is None:
    ge = g.genes.get('radarTracks.points[].trackId')
    big = ge is not None and max(ge.a, ge.b, ge.a + abs(ge.b)) + 64 >= BIG_ID
    return BIG if big else P.theoretical(sp, g)
  if any(m.which() == 'radarTracks' and any(p.trackId >= BIG_ID for p in m.radarTracks.points) for m in ins):
    return BIG
  return P.theoretical(sp, g, ins)


def cycles(r, prefix=False):
  # P.cycles plus radard's vEgo history, appended when carState updated (radard.py:197-199); prefix: up to the first bad input
  hist, cs = deque([0.], maxlen=int(round(r.cp.radarDelay / DT_MDL)) + 1), None
  for c in P.cycles(r, ('radarState',)):
    if prefix and any(map(r.space.bad, c.last.values())):
      return
    if c.last.get('carState') is not cs:  # not recv: a dup trigger has the same logMonoTime
      cs = c.last['carState']
      hist.append(cs.carState.vEgo)
    yield c, c.out['radarState'].radarState, hist


def v_ego(c):
  return c.last['carState'].carState.vEgo if 'carState' in c.last else 0.


def points(c):
  return c.last['radarTracks'].radarTracks.points if 'radarTracks' in c.last else []


def eq(a, b):
  return np.float32(a) == np.float32(b) or (a != a and b != b)


def lead_errors(name, ld, md_lead, v_ego, model_v_ego, pts, hist):
  if not ld.present:
    return [] if ld.to_dict() == log.RadarState.LeadData.new_message().to_dict() else ['not present but not default']
  if not ld.radar:
    return [n for n, ok in [
      ('radarTrackId', ld.radarTrackId == -1), ('aLeadTau', eq(ld.aLeadTau, 0.3)), ('modelProb', ld.modelProb >= 0.5),
      ('dRel', eq(ld.dRel, md_lead.x[0] - RADAR_TO_CAMERA)), ('yRel', eq(ld.yRel, -md_lead.y[0])),
      ('vRel', eq(ld.vRel, md_lead.v[0] - model_v_ego)), ('vLead', eq(ld.vLead, v_ego + md_lead.v[0] - model_v_ego)),
      ('vLeadK', eq(ld.vLeadK, ld.vLead)), ('aLeadK', eq(ld.aLeadK, md_lead.a[0]))] if not ok]
  pt = pts.get(ld.radarTrackId)
  if pt is None:
    return [f'radarTrackId {ld.radarTrackId} not in the current radarTracks']
  override = name == 'leadOne' and ld.modelProb == 0 and v_ego < V_EGO_STATIONARY and abs(pt.yRel) < 1 and 0.75 < pt.dRel < 25
  return [n for n, ok in [
    ('dRel', eq(ld.dRel, pt.dRel)), ('yRel', eq(ld.yRel, pt.yRel)), ('vRel', eq(ld.vRel, pt.vRel)),
    ('vLead', eq(ld.vLead, pt.vRel + hist[0])), ('aLeadTau', 0 <= ld.aLeadTau <= _LEAD_ACCEL_TAU),
    ('modelProb', ld.modelProb >= 0.5 or override)] if not ok]


@P.prop('invariant provenance')
def provenance(r):
  # every published lead is exactly one of: default (absent), the vision lead, or a current radar point
  for c, rs, hist in cycles(r):
    md = c.trig.modelV2
    if len(md.leadsV3) < 2:
      continue
    v = v_ego(c)
    model_v_ego = md.velocity.x[0] if len(md.velocity.x) else v
    pts = {p.trackId: p for p in points(c)}
    for i, name in enumerate(LEADS):
      bad = lead_errors(name, getattr(rs, name), md.leadsV3[i], v, model_v_ego, pts, hist)
      assert not bad, f't={c.t:.2f}s {name} {bad}: {getattr(rs, name)}'
    assert rs.mdMonoTime == c.trig.logMonoTime, f't={c.t:.2f}s mdMonoTime'
    errs = (c.last['radarTracks'] if 'radarTracks' in c.last else S.new_message('radarTracks')).radarTracks.errors.to_dict()
    assert rs.radarErrors.to_dict() == errs, f't={c.t:.2f}s radarErrors {rs.radarErrors} != {errs}'
  return True


def prefix_variant(f):
  # the check on every cycle, and untagged on the cycles before the first bad input
  def make(prefix):
    check = P.prop(f.name + (' (finite inputs)' if prefix else ''))(lambda r: f(r, prefix))
    check.untagged = prefix
    return check
  return make(False), make(True)


@P.prop('invariant low speed lead')
def low_speed_override(r, prefix):
  # below V_EGO_STATIONARY, a radar point in the stopping window always yields a leadOne no farther than it (radard.py:99-101,166-174)
  for c, rs, _ in cycles(r, prefix):
    v = v_ego(c)
    near = [p.dRel for p in {p.trackId: p for p in points(c)}.values() if abs(p.yRel) < 1 and v < V_EGO_STATIONARY and 0.75 < p.dRel < 25]
    if near and len(c.trig.modelV2.leadsV3) > 1:
      assert rs.leadOne.present and rs.leadOne.dRel <= min(near), f't={c.t:.2f}s v_ego {v} point at {min(near)}, leadOne {rs.leadOne}'


def in_range(p):
  return 0 <= p <= 1  # modeld's sigmoid image (ledger); NaN is False here, so the filters stay in range


@P.prop('invariant lead liveness')
def liveness(r, prefix):
  # model lead prob held clearly above 0.5 -> present; clearly below -> absent, except leadOne's low speed override
  hi, lo, ok = [0, 0], [0, 0], [True, True]
  for c, rs, _ in cycles(r, prefix):
    lds = c.trig.modelV2.leadsV3
    if len(lds) < 2:
      continue
    for i, lead in enumerate(LEADS):
      p = lds[i].prob
      ok[i] = ok[i] and (in_range(p) or p != p)  # an out-of-range prob is retained by the filter: stop checking that lead
      if not ok[i]:
        continue
      hi[i] = hi[i] + 1 if in_range(p) and p >= 0.9 else 0
      lo[i] = lo[i] + 1 if in_range(p) and p <= 0.4 else 0
      ld = getattr(rs, lead)
      assert hi[i] < 20 or ld.present, f't={c.t:.2f}s model {lead} prob >= 0.9 for 1 s, not present'
      assert lo[i] < 40 or not ld.present or (i == 0 and ld.radar and ld.modelProb == 0 and v_ego(c) < V_EGO_STATIONARY), \
        f't={c.t:.2f}s model {lead} prob <= 0.4 for 2 s, still present: {ld}'


def rename(ms):
  out = []
  for m in ms:
    if m.which() == 'radarTracks':
      b = m.as_builder()
      for pt in b.radarTracks.points:
        pt.trackId ^= RENAME
      m = log.Event.from_bytes(b.to_bytes()).__enter__()
    out.append(m)
  return out


@P.prop('metamorphic trackId renaming')
def renaming(r):
  # trackIds are identities only (radard.py:202-218): a bijection on them changes radarTrackId accordingly and nothing else
  if r.err or not any(m.which() == 'radarTracks' and len(m.radarTracks.points) for m in r.ins):
    return None
  r2 = r.rerun(transform=rename)
  assert r2.err is None, f'renamed run: {r2.err}'
  a, b = [m.radarState for m in P.outs(r, 'radarState')], [m.radarState for m in P.outs(r2, 'radarState')]
  assert len(a) == len(b), f'{len(a)} vs {len(b)} outputs'
  for k, (x, y) in enumerate(zip(a, b, strict=True)):
    for name in LEADS:
      dx, dy = getattr(x, name).to_dict(), getattr(y, name).to_dict()
      if dx.get('radar'):
        dx['radarTrackId'] ^= RENAME
      assert str(dx) == str(dy), f'output {k} {name}: {dx} vs renamed {dy}'
  return True


def steady(i):
  # discrete inputs: model prob side of 0.5, the set of trackIds, which inputs arrived. Following the model or radar tracks
  # appearing is not chatter; a continuous input crossing a threshold is
  def key(r):
    out = []
    for c, _, _ in cycles(r):
      lds = c.trig.modelV2.leadsV3
      p = lds[i].prob if len(lds) > 1 else 0.
      out.append((p > 0.5, in_range(p), frozenset(pt.trackId for pt in points(c)), tuple(sorted(c.last))))
    return out
  return key


DEAD_READY = 'dead: ready = sm.seen[modelV2] (radard.py:195) and main polls modelV2 (:263,269), so it is True from the first update'
RD = 'selfdrive/controls/radard.py:'
UNREACHED = {
  'dead: dt = DT_MDL (radard.py:182)': [RD + 'KalmanParams.__init__:"assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"*:F'],
  'dead: update() runs before every publish (radard.py:271-272)': [RD + 'RadarD.publish:"assert self.radar_state is not None":None'],
  'enforced modelV2.leadsV3.len, modelV2.velocity.x.len': [RD + 'RadarD.update:"if len(leads_v3) > 1:":F',
                                                           RD + 'RadarD.update:"if len(sm[\'modelV2\'].velocity.x):":F'],
  DEAD_READY: [RD + 'get_lead:"if len(tracks) > 0 and ready and lead_prob > .5:"#1:F',
               RD + 'get_lead:"elif (track is None) and ready and (lead_prob > .5):"#1:F'],
  'dead: a matched track takes the if (radard.py:162)': [RD + 'get_lead:"elif (track is None) and ready and (lead_prob > .5):":notNone'],
}

NAN_BUGS = {  # with NaN inputs, also on runs with a big trackId
  'finite radarState': ('theoretical', 'no finiteness check: a NaN model lead, radar point or vEgo is published as a valid present lead '
                        + '(radard.py:136-150)', 'known_nan_vision_lead.json'),
  'finite radarState (1 s after non-finite inputs)': ('theoretical', 'one NaN vRel (or vEgo, via v_lead :214) latches that track\'s KF '
                                                      + 'forever (simple_kalman.py:42-46)', 'known_kf_latch.json'),
  'invariant lead liveness': ('theoretical', 'one NaN model prob latches lead_prob_filters NaN forever (radard.py:236-239, '
                              + 'filter_simple.py:12-14)', 'known_prob_latch.json'),
  'invariant low speed lead': ('theoretical', 'a NaN vision lead dRel makes "closest_track.dRel < lead_dict[dRel]" False (radard.py:173)',
                               'known_nan_low_speed.json'),
}
REAL = {  # on every run
  'no_chatter radarState.leadOne.present (steady discrete inputs)': ('real', 'no hysteresis on the low speed window |y|<1, 0.75<d<25, '
                                                                     + 'v<4 (radard.py:101)', 'known_chatter_low_speed_window.json'),
  'no_chatter radarState.leadOne.radar (steady discrete inputs)': ('real', 'no hysteresis on the low speed window (radard.py:101) or the '
                                                                   + 'dist/vel sanity gates (:128-129)', 'known_chatter_vel_sane.json'),
  'no_chatter radarState.leadTwo.radar (steady discrete inputs)': ('real', 'no hysteresis on the dist/vel sanity gates (radard.py:128-129)',
                                                                   'known_chatter_vel_sane.json'),
}
KNOWN = {
  f'alive radarState ({BIG})': ('theoretical', 'trackId >= 2^31 overflows the Int32 radarTrackId (car.capnp trackId is UInt64) '
                                + 'and kills radard (radard.py:95)', 'known_trackid_int32.json'),
  **{f'{k} ({t})': v for k, v in NAN_BUGS.items() for t in (THEO, BIG)},
} | REAL

SPEC = Spec(
  proc='radard',
  scope={'selfdrive/controls/radard.py': None},
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  template=U.shape,
  tag=tag,
  huge=1e30,  # radard's float32 arithmetic overflows past it
  passthrough=('radarTracks.errors',),  # radard.py:225 radarErrors = rr.errors
  bounds={
    'carState.vEgo': [V_EGO_STATIONARY, 3., 0.],
    'radarTracks.points[].dRel': [RADAR_TO_CAMERA],
    'radarTracks.points[].yRel': [-1.],
    'modelV2.leadsV3[].prob': [0.5, 0.4, 0.9, 1., 1.7e-5],
    'modelV2.leadsV3[].xStd[]': [1e-4], 'modelV2.leadsV3[].yStd[]': [1e-4], 'modelV2.leadsV3[].vStd[]': [1e-4],
    'radarTracks.points[].trackId': [BIG_ID - 1, BIG_ID],
  },
  relations={  # the match gates: dRel vs x0 - 1.52, yRel vs -y0, vRel + vEgo vs v0 (radard.py:116-129)
    'radarTracks.points[].dRel': [('modelV2.leadsV3[].x[]', 1.)],
    'radarTracks.points[].yRel': [('modelV2.leadsV3[].y[]', -1.)],
    'radarTracks.points[].vRel': [('modelV2.leadsV3[].v[]', 1.), ('carState.vEgo', -1.)],
  },
  checks=[
    P.alive('radarState', 'modelV2'),
    P.finite('radarState', degraded=lambda m: False),
    P.recovers(['radarState'], math.inf), P.recovers(['radarState'], 1.),
    P.valid_from_inputs('radarState', 'modelV2'),
    provenance,
    *prefix_variant(low_speed_override),
    *prefix_variant(liveness),
    renaming,
    *(P.no_chatter(f'radarState.{n}.{f}', 0.5, steady=steady(i), name=f'no_chatter radarState.{n}.{f} (steady discrete inputs)')
      for i, n in enumerate(LEADS) for f in ('present', 'radar')),
  ],
)
