# ubloxd's input is a byte stream, so its genome is a list of UBX frames (one gene each) plus a chunking: frames are built
# from the parser's own schemas (ubx.py, gps.py, glonass.py), every field drawn from its integer type's classes and the
# boundaries ubloxd branches on, with valid checksums unless the gene corrupts it.
#   gene key 'ubloxRaw[k]', idx = 'kind key=value ...' (structural fields), a = seed for the other fields, b = corruption,
#   t0 = time, t1 = split point within the frame (0: none). Gene 'ubloxRaw.chunks' rechunks the whole stream (a = seed).
import copy
import math
import random
import struct
from typing import get_type_hints

import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.test.process_fuzz import properties as P
from openpilot.selfdrive.test.process_fuzz import space as S
from openpilot.selfdrive.test.process_fuzz.harness import Spec
from openpilot.system.ubloxd import binary_struct as bs
from openpilot.system.ubloxd.glonass import Glonass
from openpilot.system.ubloxd.gps import Gps
from openpilot.system.ubloxd.ubx import Ubx

TYPES = {'pvt': (0x0107, Ubx.NavPvt), 'rawx': (0x0215, Ubx.RxmRawx), 'sat': (0x0135, Ubx.NavSat), 'hw': (0x0A09, Ubx.MonHw),
         'hw2': (0x0A0B, Ubx.MonHw2)}
KINDS = (*TYPES, 'gps', 'glo', 'sfrbx', 'other', 'garbage')
# boundaries ubloxd and the capnp fields it fills branch on, beyond each integer type's own edges
EDGES = {
  'num_sv': [127, 128], 'leap_s': [-1, 0, 18], 'month': [0, 1, 12, 13], 'year': [0, 1, 1970, 9999, 10000], 'day': [0, 1, 31, 32],
  'hour': [23, 24], 'min': [59, 60], 'sec': [59, 60, 61], 'flags': [0, 1, 2, 3], 'num_meas': [0, 1, 2, 5], 'num_svs': [0, 1, 2, 5],
  'gnss_id': [*range(7), 7, 255], 'a_status': [*range(5), 5], 'a_power': [0, 1, 2, 3], 'cfg_source': [102, 111, 112, 113, 0],
  'trk_stat': [0, 1, 2, 4, 8, 15], 'rec_stat': [0, 1, 4, 5], 'pr_stdev': [15, 16], 'week_no': [852, 853, 1023],
  'tow_count': [0, 100799, 100800, 131071], 't_oe': [0, 1], 'f_t': [15],
}
NOISE = ('garbage',)  # genes that are not frames; corrupted frames (b in BAD) are noise too
BAD = (1,)  # checksum wrong


def fields(cls):
  hints = get_type_hints(cls, include_extras=True)
  return [(n, bs._field_type_from_spec(hints[n]) or hints[n]) for n, _ in cls.__binary_fields__]


class Writer:
  # big-endian bits, byte fields aligned
  def __init__(self):
    self.v, self.n = 0, 0

  def put_bits(self, v, n):
    self.v, self.n = (self.v << n) | (v & ((1 << n) - 1)), self.n + n

  def put_bytes(self, b):
    self.put_bits(0, -self.n % 8)
    self.put_bits(int.from_bytes(b, 'big'), 8 * len(b))

  def data(self):
    self.put_bits(0, -self.n % 8)
    return self.v.to_bytes(self.n // 8, 'big')


def draw_int(rng, name, lo, hi):
  r = rng.random()
  if r < 0.3:
    return rng.choice([lo, hi, 0, 1, hi - 1, -1 if lo < 0 else 2])
  if r < 0.6 and name in EDGES:
    return min(max(rng.choice(EDGES[name]), lo), hi)
  return rng.randint(lo, hi)


def encode(cls, rng, fixed, w):
  # writes cls with every field drawn (or fixed[name]); returns the written values so counts and switches resolve
  vals = {}
  for name, t in fields(cls):
    v = fixed.get(name)
    if isinstance(t, bs.ConstType):
      w.put_bytes(t.expected if v is None else v)
      continue
    if isinstance(t, bs.EnumType):
      t = t.base_type
    if isinstance(t, bs.IntType):
      lo, hi = (-(1 << (t.bits - 1)), (1 << (t.bits - 1)) - 1) if t.signed else (0, (1 << t.bits) - 1)
      v = draw_int(rng, name, lo, hi) if v is None else v
      w.put_bytes(struct.pack(bs._int_format(t), v))
    elif isinstance(t, bs.FloatType):
      v = rng.choice([0., -0., math.nan, math.inf, -math.inf, 1.5e7, 2e7, -1., 3e38, 1e-40] + ([1e300] if t.bits == 64 else [])) if v is None else v
      w.put_bytes(struct.pack(bs._float_format(t), v))
    elif isinstance(t, bs.BitsType):
      v = (draw_int(rng, name, 0, (1 << t.bits) - 1) if t.bits > 1 else rng.random() < 0.5) if v is None else v
      w.put_bits(int(v), t.bits)
    elif isinstance(t, bs.BytesType):
      w.put_bytes(bytes(rng.getrandbits(8) for _ in range(t.size)) if v is None else v)
    elif isinstance(t, bs.ArrayType):
      v = [encode(t.element_type, rng, fixed.get(f'{name}[]', {}), w) for _ in range(vals[t.count_field])]
    elif isinstance(t, bs.SwitchType):
      sel = vals
      for p in t.selector.split('.'):
        sel = sel[p]
      target = t.cases.get(sel, t.default)
      if target is not None:
        v = encode(target, rng, fixed.get(name, {}), w)
    elif isinstance(t, type) and issubclass(t, bs.BinaryStruct):
      v = encode(t, rng, fixed.get(name, {}), w)
    vals[name] = v
  return vals


def payload(cls, rng, fixed):
  w = Writer()
  vals = encode(cls, rng, fixed, w)
  return w.data(), vals


def typed(g):
  # (msg type, payload, field values) of a NAV-PVT / RXM-RAWX / NAV-SAT / MON-HW / MON-HW2 gene
  kind, kv = parse_idx(g.idx)
  t, cls = TYPES[kind]
  fixed = {k: v for k, v in kv.items() if k not in ('extra', 'cut')}
  if 'gnss' in kv:
    fixed = {'meas[]' if kind == 'rawx' else 'svs[]': {'gnss_id': kv['gnss']}}
  return t, *payload(cls, random.Random(int(g.a)), fixed)


def representable(g):
  # every field fits the capnp field ubloxd writes it to, and floats are finite (a receiver never sends otherwise;
  # only a corrupted frame passing the 16-bit checksum does). Else ubloxd drops the frame silently (ubloxd.py:529).
  kind = parse_idx(g.idx)[0]
  if kind not in TYPES:
    return True
  v = typed(g)[2]
  fin = all(math.isfinite(x) for x in [v.get('rcv_tow', 0.)] + [m[k] for m in v.get('meas', []) for k in ('pr_mes', 'cp_mes', 'do_mes')])
  return fin and {'pvt': v.get('num_sv', 0) < 128, 'rawx': v.get('leap_s', 0) >= 0 and all(m['gnss_id'] <= 6 for m in v.get('meas', [])),
                  'sat': all(x['gnss_id'] <= 6 for x in v.get('svs', [])), 'hw': v.get('a_status', 0) <= 4 and v.get('a_power', 0) <= 2}.get(kind, True)


def frame(msg_type, body, bad=False):
  b = struct.pack('>H', msg_type) + struct.pack('<H', len(body)) + body
  a = c = 0
  for x in b:
    a = (a + x) & 0xff
    c = (c + a) & 0xff
  return b'\xb5\x62' + b + bytes([a, c ^ (0x55 if bad else 0)])


def parse_idx(idx):
  kind, *kv = idx.split()
  return kind, {k: int(v) for k, v in (x.split('=') for x in kv)}


def build(g):
  # one gene -> bytes
  kind, kv = parse_idx(g.idx)
  rng = random.Random(int(g.a))
  if kind == 'garbage':
    tail = [b'', b'\xb5', b'\xb5\x62', b'\xb5\x62\x01\x07', b'\xb5\x62\x01\x07\xff\xff', b'\xb5\x62\x02\x15\x00\x10'][kv.get('tail', 0)]
    b = bytes(rng.getrandbits(8) for _ in range(kv.get('n', 8))) + tail
    return b'\x01' + b[1:] if b[:1] == b'\x00' else b  # see stream(): no gene starts with 0x00
  if kind in TYPES:
    t, body, _ = typed(g)
  elif kind == 'gps':
    how = {'subframe_id': kv.get('sf', 1)} | ({'tow_count': 99600 + kv['toe0']} if 'toe0' in kv else {})  # week rollover (ubloxd.py:270)
    tlm = {} if kv.get('tlm', 1) else {'preamble': b'\x8a'}
    body = {'iode': kv['iode']} if 'iode' in kv else {}
    sub = {1: {'iodc_lsb': kv['iode']} if 'iode' in kv else {}, 2: body | ({'t_oe': 0} if 'toe0' in kv else {}), 3: body}.get(how['subframe_id'], {})
    sf = payload(Gps, rng, {'tlm': tlm, 'how': how, 'body': sub})[0]
    words = [((rng.getrandbits(2) << 30) | (int.from_bytes(sf[3 * i:3 * i + 3], 'big') << 6) | rng.getrandbits(6)) for i in range(10)]
    t, body = 0x0213, bytes([0, kv.get('sv', 1), 0, 0, kv.get('nw', 10), 0, 2, 0]) + b''.join(struct.pack('<I', x) for x in words)
  elif kind == 'glo':
    s = payload(Glonass, rng, {'idle_chip': bool(kv.get('idle', 0)), 'string_number': kv.get('n', 1), 'superframe_number': kv.get('sf', 1)})[0]
    words = [int.from_bytes(s[4 * i:4 * i + 4], 'big') for i in range(4)]
    t, body = 0x0213, bytes([6, kv.get('sv', 1), 0, kv.get('fid', 7), kv.get('nw', 4), 0, 2, 0]) + b''.join(struct.pack('<I', x) for x in words)
  elif kind == 'sfrbx':  # another constellation, or a word count that doesn't match the length
    nw = kv.get('nw', 2)
    t, body = 0x0213, bytes([kv.get('gnss', 2), 1, 0, 0, nw, 0, 2, 0]) + bytes(rng.getrandbits(8) for _ in range(4 * kv.get('words', nw)))
  else:  # an unhandled message type, or a handled one with an arbitrary payload
    t, body = kv.get('type', 0x0101), bytes(rng.getrandbits(8) for _ in range(kv.get('n', 16)))
  body = body[:len(body) - kv.get('cut', 0)] + bytes(kv.get('extra', 0))
  return frame(t, body, g.b in BAD)


def stream(g):
  # genome -> [(time, chunk)]: one chunk per gene (split in two at t1), or the whole stream rechunked. A chunk is sent at
  # the time of the frame its last byte belongs to, so rechunking never moves a frame's completion time.
  # pigeond drops a chunk starting with 0x00 (and re-inits the receiver, pigeond.py:320-325), so no chunk starts with one:
  # every chunking of a genome is the same byte stream. Chunks gene b=1: no chunk ends in 0xb5 (the known framer bug).
  frames = [(ge.t0, build(ge), ge.t1) for _, ge in frames_sorted(g)]
  rc = g.genes.get('ubloxRaw.chunks')

  def cut(data, i, n, avoid_b5):
    while i + n < len(data) and (data[i + n] == 0 or (avoid_b5 and data[i + n - 1] == 0xb5)):
      n += 1
    return n
  out = []
  if rc is None:
    for t, b, split in frames:
      k = int(round(split * len(b)))
      k = k + cut(b, k, 0, False) if 0 < k < len(b) else k
      out += [(t, b[:k]), (t + S.DT, b[k:])] if 0 < k < len(b) else [(t, b)]
  else:
    rng = random.Random(int(rc.a))
    owner = [t for t, b, _ in frames for _ in b]
    data, i = b''.join(b for _, b, _ in frames), 0
    while i < len(data):
      n = cut(data, i, rng.choice([1, 2, 3, 7, 64, 0x40 * rng.randint(1, 64)]), rc.b == 1)
      out.append((owner[min(i + n, len(data)) - 1], data[i:i + n]))
      i += n
  return [(t, c) for t, c in out if c]


class UbloxSpace(S.Space):
  def draw(self, rng):
    g = S.Genome(*self.groups[0][0][:2], dur=self.dur, fpv=self.groups[0][0][2])
    for _ in range(rng.choice([1, 5, 20, 60])):
      self.add(rng, g)
    if rng.random() < 0.3:
      g.genes['ubloxRaw.chunks'] = S.Gene('hold', rng.randrange(1 << 30), 0.)
    return g

  def add(self, rng, g):
    kind = rng.choice(KINDS)
    t = rng.uniform(0, self.dur)
    seq = []
    if kind in TYPES:
      kv = rng.choice([{}, {}, {'cut': rng.choice([1, 8])}, {'extra': rng.choice([1, 8])}] +
                      ([{'gnss': rng.choice(EDGES['gnss_id'])}] if kind in ('rawx', 'sat') else []))
      seq = [(kind, kv)]
    elif kind == 'gps':  # a subframe set for one SV, with matching or mismatching IODEs, sometimes incomplete or broken
      sv, iode = rng.choice([1, 2, 32, 255]), rng.randrange(256)
      for sf in rng.choice([[1, 2, 3], [1, 2, 3], [3, 1, 2], [1, 2], [rng.choice([0, 4, 5, 7])]]):
        kv = {'sf': sf, 'sv': sv, 'iode': iode if rng.random() < 0.8 else rng.randrange(256)}
        kv |= rng.choice([{}, {}, {}, {'tlm': 0}, {'nw': rng.choice([9, 11])}, {'toe0': rng.choice([-1, 0])}])
        seq.append(('gps', kv))
    elif kind == 'glo':  # strings 1-5 for one frequency channel, superframe known/unknown, sometimes from another SV
      sv, fid, sf = rng.choice([1, 5, 24, 255]), rng.choice([0, 7, 13, 255]), rng.choice([0, 1, 2])
      for n in rng.choice([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2], [rng.choice([0, 6, 15])]]):
        kv = {'n': n, 'sv': sv if rng.random() < 0.8 else rng.choice([1, 2, 255]), 'fid': fid, 'sf': sf if rng.random() < 0.9 else rng.choice([0, 3])}
        kv |= rng.choice([{}, {}, {}, {'idle': 1}, {'nw': rng.choice([3, 5])}])
        seq.append(('glo', kv))
    elif kind == 'sfrbx':
      # another constellation, GPS/GLONASS (0/6) with a wrong word count, or a word count that doesn't match the length
      seq = [('sfrbx', rng.choice([{'gnss': rng.choice([0, 1, 2, 3, 5, 6, 7, 255])}, {'nw': 2, 'words': 3}, {'nw': 0, 'words': 0}]))]
    elif kind == 'other':
      seq = [('other', {'type': rng.choice([0x0101, 0x0501, 0x0107, 0x0213, 0x0215, 0x0A09, 0x0A0B, 0x0135]), 'n': rng.choice([0, 1, 7, 40, 120])})]
    else:
      seq = [('garbage', {'n': rng.choice([0, 1, 8, 100]), 'tail': rng.randrange(6)})]
    gap = rng.choice([0.05, 2., 11.])  # GLONASS strings come every 2 s; 10 s is ubloxd's timing window
    for k, (kind, kv) in enumerate(seq):
      ge = S.Gene('hold', rng.randrange(1 << 30), float(rng.random() < 0.1), t + k * gap, rng.choice([0.] * 4 + [rng.random()]))
      ge.idx = ' '.join([kind] + [f'{a}={v}' for a, v in kv.items()])
      g.genes[f'ubloxRaw[{len(g.genes):03d}]'] = ge

  def messages(self, g):
    out = []
    for i, (t, c) in enumerate(stream(g)):
      m = messaging.new_message('ubloxRaw', len(c))
      m.ubloxRaw = c
      m.logMonoTime = S.T0 + int(round(t * 1e9)) + i  # strictly increasing, as one publisher's stamps are
      out.append(m.as_reader())
    return out


LEDGER = {
  'ubloxRaw:empty': ('enforced', 'pigeond only sends non-empty chunks (pigeond.py:320)', None),
  'ubloxRaw:leading_zero': ('enforced', 'a chunk starting with 0x00 makes pigeond re-init instead of sending (pigeond.py:321-325); '
                            + 'no gene starts with one and chunk boundaries skip them, so every chunking is the same byte stream', None),
  'logMonoTime:backwards': ('enforced', 'one publisher, new_message stamps time.monotonic (messaging/__init__.py:70)', None),
}

REPR = 'unrepresentable fields'


def tag(sp, g, ins=None):
  return None if all(representable(ge) for k, ge in g.genes.items() if k.startswith('ubloxRaw[')) else REPR


BS, UB = 'system/ubloxd/binary_struct.py:', 'system/ubloxd/ubloxd.py:UbloxMsgParser.'
UNREACHED = {
  'dead: no byte field follows an unaligned bit run in gps.py/glonass.py (checked by walking the structs)':
    [BS + 'BinaryReader._align_to_byte:"if self.bit_pos > 0:":T'],
  'dead: bit structs are only parsed from fixed 30 (GPS) / 16 (GLONASS) byte buffers holding all their bits (ubloxd.py:232, :317)':
    [BS + 'BinaryReader.read_bits_int_be:"if self.pos >= len(self.data):":T'],
  'dead: no bits(0) field': [BS + 'BinaryReader.read_bits_int_be:"while bits_remaining > 0:":F'],
  'harness: runs at import (class definitions), before tracing starts': [BS + 'BinaryStruct.__init_subclass__:*'],
  'dead: an IntEnum lookup only raises ValueError': [BS + '_parse_field:"except ValueError:":F'],
  'dead: only Ubx (the whole frame) has a substream and ubloxd parses frames by hand (ubloxd.py:133-175)':
    [BS + '_parse_field:"if isinstance(spec, SubstreamType):":T'],
  'dead: every field spec is a supported type': [BS + '_parse_field:"if isinstance(spec, type) and issubclass(spec, BinaryStruct):"*:F'],
  'dead: the cached bytes already parsed at :317 / :232, and the string/subframe number picked their slot and type': [
    UB + '_parse_glonass_ephemeris:"assert isinstance(s*:F', UB + '_parse_glonass_ephemeris:"except Exception:"*:T',
    UB + '_parse_gps_ephemeris:"assert isinstance(s*:F'],
  'enforced ubloxRaw:empty': ['system/ubloxd/ubloxd.py:UbxFramer.add_data:"if not incoming:":F'],
}
KNOWN = {
  'noise does not eat frames': ('real', 'a stray b5 62 is taken as a header: the framer waits up to 64 KB before the checksum fails '
                                + '(ubloxd.py:69-72)', 'known_noise_stall.json'),
  'chunking invariant': ('real', 'a chunk ending in b5 with no preamble in the buffer is cleared (ubloxd.py:58-60): its frame is lost',
                         'known_trailing_b5.json'),
  'invariant fix in range': ('theoretical', 'NAV-PVT lat/lon/accuracies are scaled int32s, no range check (ubloxd.py:185-191)',
                             'known_fix_range.json'),
  'glonass ephemeris from one SV': ('theoretical', 'GLONASS strings are cached per frequency channel, not per SV (ubloxd.py:323-353)',
                                    'known_glonass_freq_id.json'),
}
KNOWN |= {
  f'one output per frame ({REPR})': ('theoretical', 'a field outside its capnp type raises in _gen_*, the frame is dropped (ubloxd.py:527-530)',
                                     'known_enum_drop.json'),
  f'finite ubloxGnss ({REPR})': ('theoretical', 'RXM-RAWX floats are copied as is, NaN/inf included (ubloxd.py:434-443)', 'known_nan_rawx.json'),
}


def outputs(r, glonass=True):
  # in order per service: process replay collects each service's socket separately, so the order across services within
  # one step is not the process's. GLONASS strings are correlated by chunk arrival time (ubloxd.py:336), which rechunking
  # legitimately moves
  out = [(m.which(), getattr(m, m.which()).to_dict()) for m in r.outs if m.which() in ('ubloxGnss', 'gpsLocationExternal')]
  return sorted([(w, d) for w, d in out if glonass or 'glonassEphemeris' not in d], key=lambda x: x[0])


def frames_sorted(g):
  return sorted(((k, ge) for k, ge in g.genes.items() if k.startswith('ubloxRaw[')), key=lambda kv: (kv[1].t0, kv[0]))


def without(g, drop):
  g = copy.deepcopy(g)
  g.genes = {k: ge for k, ge in g.genes.items() if not drop(k, ge)}
  return g


def rechunk(g, k, b):
  # the stream rechunked with the genome's chunking seed + k; b=1: no chunk ends in 0xb5
  rc = g.genes.get('ubloxRaw.chunks')
  g = copy.deepcopy(g)
  g.genes['ubloxRaw.chunks'] = S.Gene('hold', (int(rc.a) if rc is not None else 0) + k, float(b))
  return g


def same(a, b):
  return a == b or str(a) == str(b)  # NaN-tolerant


def first_diff(a, b):
  return next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if not same(x, y)), min(len(a), len(b)))


def noisy(k, ge):
  return k.startswith('ubloxRaw[') and (parse_idx(ge.idx)[0] in NOISE or ge.b in BAD)


@P.prop('alive ubloxd')
def alive(r):
  assert r.err is None, r.err


@P.prop('noise does not eat frames')
def noise_invariant(r):
  # garbage and bad-checksum frames between frames must not change what the frames produce (framer resync)
  if r.err or not any(noisy(k, ge) for k, ge in r.genome.genes.items()):
    return None
  gl = 'ubloxRaw.chunks' not in r.genome.genes
  a, b = outputs(r, gl), outputs(r.rerun(without(r.genome, noisy)), gl)
  assert same(a, b), f'{len(a)} outputs with noise, {len(b)} without; first difference at {first_diff(a, b)}'
  return 'noise'


@P.prop('noise does not corrupt frames')
def noise_prefix(r):
  # the above without the known bugs: with noise, the outputs are per service the clean ones or a prefix of them (frames
  # still held by a stray header at the end), never different or extra ones. No chunk ends in 0xb5 ('chunking invariant')
  if r.err or not any(noisy(k, ge) for k, ge in r.genome.genes.items()):
    return None
  ra = r.rerun(rechunk(r.genome, 0, 1))
  if ra.err:
    return None
  a, b = outputs(ra, False), outputs(r.rerun(without(ra.genome, noisy)), False)
  for s in ('gpsLocationExternal', 'ubloxGnss'):
    x, y = [d for w, d in a if w == s], [d for w, d in b if w == s]
    assert same(x, y[:len(x)]), f'{len(a)} outputs with noise, {len(b)} without; first difference at {first_diff(a, b)}'
  return 'noise'


@P.prop('chunking invariant')
def chunking_invariant(r):
  # pigeond reads the serial port in arbitrary pieces: the same bytes split differently must give the same outputs
  if r.err:
    return None
  a, b = outputs(r, False), outputs(r.rerun(rechunk(r.genome, 1, 0)), False)
  assert same(a, b), f'{len(a)} outputs vs {len(b)} rechunked'
  return 'chunking'


@P.prop('chunking invariant (no chunk ends in 0xb5)')
def chunking_no_b5(r):
  # the above without the known trailing-0xb5 loss
  if r.err:
    return None
  a, b = (outputs(r.rerun(rechunk(r.genome, k, 1)), False) for k in (0, 1))
  assert same(a, b), f'{len(a)} outputs vs {len(b)} rechunked'
  return 'chunking'


@P.prop('one output per frame')
def one_per_frame(r):
  # every complete NAV-PVT / RXM-RAWX / NAV-SAT / MON-HW / MON-HW2 frame yields its message: no silent drops
  if r.err or any(k.startswith('ubloxRaw[') and parse_idx(ge.idx)[0] not in TYPES for k, ge in r.genome.genes.items()):
    return None
  name = {'pvt': 'gpsLocationExternal', 'rawx': 'measurementReport', 'sat': 'satReport', 'hw': 'hwStatus', 'hw2': 'hwStatus2'}
  want = [name[kind] for kind, kv in map(parse_idx, (ge.idx for _, ge in frames_sorted(r.genome) if ge.b not in BAD)) if not kv.get('cut')]
  want = sorted(want, key=lambda w: w != 'gpsLocationExternal')  # per service, as outputs()
  got = [w if w == 'gpsLocationExternal' else d and next(iter(d)) for w, d in outputs(r)]
  miss = next((w for w, x in zip(want, got + [None] * len(want), strict=False) if w != x), None)
  assert got == want, f'{len(want)} frames, {len(got)} outputs; first missing {miss}'
  return 'frames'


@P.prop('glonass ephemeris from one SV')
def glonass_one_sv(r):
  # an ephemeris for SV x must be built from x's strings only: dropping every other SV's strings keeps it
  ephs = [d['glonassEphemeris'] for w, d in outputs(r) if w == 'ubloxGnss' and 'glonassEphemeris' in d]
  if r.err or not ephs:
    return None
  sv = ephs[0]['svId']

  def other(k, ge):
    return k.startswith('ubloxRaw[') and parse_idx(ge.idx)[0] == 'glo' and parse_idx(ge.idx)[1].get('sv') != sv
  if not any(other(k, ge) for k, ge in r.genome.genes.items()):
    return 'one sv'
  ephs2 = [d['glonassEphemeris'] for w, d in outputs(r.rerun(without(r.genome, other))) if w == 'ubloxGnss' and 'glonassEphemeris' in d]
  assert any(same(ephs[0], e) for e in ephs2), f'ephemeris for SV {sv} changes without the other SVs\' strings'
  return 'several svs'


def fix_in_range(d):
  return not d.hasFix or (-90 <= d.latitude <= 90 and -180 <= d.longitude <= 180 and d.horizontalAccuracy >= 0 and d.speedAccuracy >= 0
                          and d.verticalAccuracy >= 0 and 0 <= d.satelliteCount)


SPEC = Spec(
  proc='ubloxd',
  scope={
    'system/ubloxd/ubloxd.py': {'UbxFramer', 'UbloxMsgParser'},
    'system/ubloxd/binary_struct.py': {'BinaryReader', '_parse_field', '_resolve_path', 'BinaryStruct'},
  },
  ledger=LEDGER,
  unreached=UNREACHED,
  known=KNOWN,
  tag=tag,
  duration=30.,
  examples=64,
  space=UbloxSpace,
  checks=[
    alive, noise_invariant, noise_prefix, chunking_invariant, chunking_no_b5, one_per_frame, glonass_one_sv,
    P.finite('gpsLocationExternal'), P.finite('ubloxGnss'), P.invariant('gpsLocationExternal', fix_in_range, 'fix in range'),
  ],
)
