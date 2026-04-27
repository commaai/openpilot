#!/usr/bin/env python3
import math
import re
import sys
import tty
import termios
import struct
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import select
from multiprocessing import Process, Pipe

SAMPLE_BUFFER = 4096
SEEK_SECONDS = 5
LOOKAHEAD = 16  # blocks to process ahead


# --- biquad coefficients ---

def _lowpass_coeffs(freq, q, srate):
  x = (freq * 2.0 * math.pi) / srate
  sin_x, cos_x = math.sin(x), math.cos(x)
  y = sin_x / (q * 2.0)
  a0 = y + 1.0
  return ((((1.0 - cos_x) / 2.0) / a0, (1.0 - cos_x) / a0, ((1.0 - cos_x) / 2.0) / a0),
          (1.0, (cos_x * -2.0) / a0, (1.0 - y) / a0))

def _highpass_coeffs(freq, q, srate):
  x = (freq * 2.0 * math.pi) / srate
  sin_x, cos_x = math.sin(x), math.cos(x)
  y = sin_x / (q * 2.0)
  a0 = y + 1.0
  return ((((1.0 + cos_x) / 2.0) / a0, ((1.0 + cos_x) * -1.0) / a0, ((1.0 + cos_x) / 2.0) / a0),
          (1.0, (cos_x * -2.0) / a0, (1.0 - y) / a0))

def _peaking_coeffs(fc, gain_db, q, srate):
  A = 10 ** (gain_db / 40.0)
  w0 = 2.0 * math.pi * fc / srate
  sin_w0, cos_w0 = math.sin(w0), math.cos(w0)
  alpha = sin_w0 / (2.0 * q)
  a0 = 1.0 + alpha / A
  return (((1.0 + alpha * A) / a0, (-2.0 * cos_w0) / a0, (1.0 - alpha * A) / a0),
          (1.0, (-2.0 * cos_w0) / a0, (1.0 - alpha / A) / a0))

def _lowshelf_coeffs(fc, gain_db, q, srate):
  A = 10 ** (gain_db / 40.0)
  w0 = 2.0 * math.pi * fc / srate
  sin_w0, cos_w0 = math.sin(w0), math.cos(w0)
  alpha = sin_w0 / (2.0 * q)
  ap1 = A + 1.0; am1 = A - 1.0
  two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
  a0 = ap1 + am1 * cos_w0 + two_sqrt_a_alpha
  return (((A * (ap1 - am1 * cos_w0 + two_sqrt_a_alpha)) / a0,
           (2.0 * A * (am1 - ap1 * cos_w0)) / a0,
           (A * (ap1 - am1 * cos_w0 - two_sqrt_a_alpha)) / a0),
          (1.0,
           (-2.0 * (am1 + ap1 * cos_w0)) / a0,
           (ap1 + am1 * cos_w0 - two_sqrt_a_alpha) / a0))

def _highshelf_coeffs(fc, gain_db, q, srate):
  A = 10 ** (gain_db / 40.0)
  w0 = 2.0 * math.pi * fc / srate
  sin_w0, cos_w0 = math.sin(w0), math.cos(w0)
  alpha = sin_w0 / (2.0 * q)
  ap1 = A + 1.0; am1 = A - 1.0
  two_sqrt_a_alpha = 2.0 * math.sqrt(A) * alpha
  a0 = ap1 - am1 * cos_w0 + two_sqrt_a_alpha
  return (((A * (ap1 + am1 * cos_w0 + two_sqrt_a_alpha)) / a0,
           (-2.0 * A * (am1 + ap1 * cos_w0)) / a0,
           (A * (ap1 + am1 * cos_w0 - two_sqrt_a_alpha)) / a0),
          (1.0,
           (2.0 * (am1 - ap1 * cos_w0)) / a0,
           (ap1 - am1 * cos_w0 - two_sqrt_a_alpha) / a0))


# --- fused biquad chain (C-accelerated via ctypes) ---

import ctypes, hashlib, tempfile, os

def _compile_c(name, src):
  """Compile C source to /tmp, keyed by a content hash so edits auto-invalidate."""
  digest = hashlib.sha1(src.encode()).hexdigest()[:12]
  cache_path = f'/tmp/_{name}_{digest}.so'
  if os.path.exists(cache_path):
    return ctypes.CDLL(cache_path)
  tmp = tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w')
  tmp.write(src); tmp.close()
  os.system(f'gcc -O3 -shared -fPIC -lm -o {cache_path} {tmp.name}')
  os.unlink(tmp.name)
  return ctypes.CDLL(cache_path)

_C_SRC = r"""
#include <stdint.h>
void fused_biquad(double *out, const double *in, int n, int n_f,
                  const double *b0, const double *b1, const double *b2,
                  const double *a1, const double *a2,
                  double *sx1, double *sx2, double *sy1, double *sy2) {
  for (int i = 0; i < n; i++) {
    double s = in[i];
    for (int j = 0; j < n_f; j++) {
      double y = b0[j]*s + b1[j]*sx1[j] + b2[j]*sx2[j] - a1[j]*sy1[j] - a2[j]*sy2[j];
      sx2[j] = sx1[j]; sx1[j] = s;
      sy2[j] = sy1[j]; sy1[j] = y;
      s = y;
    }
    out[i] = s;
  }
}
"""

_lib = _compile_c('biquad_fused', _C_SRC)
_lib.fused_biquad.restype = None
_lib.fused_biquad.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
  ctypes.c_void_p, ctypes.c_void_p,
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]

def _fused_process(block, coeffs, states):
  n = len(block)
  n_f = len(coeffs)
  inp = np.ascontiguousarray(block, dtype=np.float64)
  out = np.empty(n, dtype=np.float64)
  b0 = np.array([c[0] for c in coeffs]); b1 = np.array([c[1] for c in coeffs])
  b2 = np.array([c[2] for c in coeffs]); a1 = np.array([c[3] for c in coeffs])
  a2 = np.array([c[4] for c in coeffs])
  sx1 = np.array([s[0] for s in states]); sx2 = np.array([s[1] for s in states])
  sy1 = np.array([s[2] for s in states]); sy2 = np.array([s[3] for s in states])
  _lib.fused_biquad(out.ctypes.data, inp.ctypes.data, n, n_f,
                    b0.ctypes.data, b1.ctypes.data, b2.ctypes.data,
                    a1.ctypes.data, a2.ctypes.data,
                    sx1.ctypes.data, sx2.ctypes.data, sy1.ctypes.data, sy2.ctypes.data)
  for j in range(n_f):
    states[j][0] = sx1[j]; states[j][1] = sx2[j]
    states[j][2] = sy1[j]; states[j][3] = sy2[j]
  return out

def _make_coeffs_and_states(coeff_pairs):
  coeffs = [(b[0], b[1], b[2], a[1], a[2]) for b, a in coeff_pairs]
  states = [[0.0, 0.0, 0.0, 0.0] for _ in coeffs]
  return coeffs, states


# --- parse EQ file ---

def _parse_eq_file(path):
  """Parse EqualizerAPO / REW filter file. Supports PK, LSC/LS, HSC/HS, Preamp."""
  filters = []  # list of (type, fc, gain, q)
  preamp_db = 0.0
  BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)
  with open(path) as f:
    for line in f:
      # filter with Q
      m = re.match(r'Filter\s+\d+:\s+ON\s+(PK|LSC?|HSC?)\s+Fc\s+([\d.]+)\s+Hz\s+Gain\s+([-\d.]+)\s+dB\s+Q\s+([\d.]+)', line)
      if m:
        filters.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        continue
      # shelf without Q (default to Butterworth)
      m = re.match(r'Filter\s+\d+:\s+ON\s+(LSC?|HSC?)\s+Fc\s+([\d.]+)\s+Hz\s+Gain\s+([-\d.]+)\s+dB', line)
      if m:
        filters.append((m.group(1), float(m.group(2)), float(m.group(3)), BUTTERWORTH_Q))
        continue
      m = re.match(r'Preamp:\s*([-\d.]+)\s*dB', line)
      if m:
        preamp_db = float(m.group(1))
  return filters, preamp_db


# --- crystallizer constants (high-frequency harmonic exciter) ---

CR_AMT = 0.4          # mix level of generated sparkle
CR_FLOOR = 3000.0     # capture presence band from 3kHz
CR_CEIL = 8000.0      # up to 8kHz
CR_OUTPUT_HP = 6000.0 # output only the NEW harmonics (above the source band)
CR_OUTPUT_LP = 15000.0 # keep below anti-alias
CR_DRIVE = 1.5        # gentle saturation — just enough to generate harmonics

# --- bass enhancement constants (sub-band polynomial waveshaper) ---
# Per-band tanh-saturated waveshaper. Each b^n term produces a specific
# harmonic mix (b² → DC + 2nd, b³ → 1st + 3rd, b⁴ → DC + 2nd + 4th,
# b⁵ → 1st + 3rd + 5th); the DC and fundamental components are rejected
# by BK_OUTPUT_HP. The BK_*_AMT values are weights on each power term
# (not pure nth-harmonic amplitudes) and were tuned by ear.
#
# NOTE on alternatives: the mathematically-cleaner analytic-signal synthesis
# (I + jQ via Hilbert transform, then Re(zⁿ)) was tried and rejected. It
# eliminates DC-offset modulation and difference-tone IMD, but a cheap
# first-order allpass quadrature has too much group delay at low frequencies
# (~1–2 ms at band center), which produces click-like transient artifacts
# that excite the output bandpass into audible ringing at higher volumes.
# A proper polyphase-allpass Hilbert network would fix it, but is not worth
# the complexity — the polynomial waveshaper sounds cleaner at every volume.

BK_BANDS = [           # (low_hz, high_hz) — processed independently
  (50, 125),           # sub-bass
  (125, 250),          # bass
  (250, 500),          # upper bass
]
BK_OUTPUT_HP = 800.0   # only pass harmonics above this
BK_OUTPUT_LP = 1500.0  # tight ceiling — keep out of vocal/presence range
BK_H2_AMT = 0.5        # b² weight (DC + 2nd harmonic)
BK_H3_AMT = 1.2        # b³ weight (1st + 3rd)
BK_H4_AMT = 1.0        # b⁴ weight (DC + 2nd + 4th)
BK_H5_AMT = 0.6        # b⁵ weight (1st + 3rd + 5th)


# --- pipeline worker processes ---

def _eq_coeff_for_filter(ftype, fc, gain_db, q, srate):
  if ftype == 'PK':
    return _peaking_coeffs(fc, gain_db, q, srate)
  elif ftype in ('LS', 'LSC'):
    return _lowshelf_coeffs(fc, gain_db, q, srate)
  elif ftype in ('HS', 'HSC'):
    return _highshelf_coeffs(fc, gain_db, q, srate)
  raise ValueError(f"unknown filter type: {ftype}")

SUBSONIC_FREQ = 60.0  # Hz — nothing below this reaches the speaker

FIR_TAPS = 2048         # FIR length (43ms at 48kHz)
FIR_Q_SCALE = 0.5       # widen filters by 2x when designing FIR (less ringing, smoother)

def _biquad_freq_response(b, a, w):
  """Evaluate biquad at angular frequency w. b = (b0,b1,b2), a = (1,a1,a2)."""
  z_inv = np.exp(-1j * w)
  num = b[0] + b[1] * z_inv + b[2] * z_inv**2
  den = a[0] + a[1] * z_inv + a[2] * z_inv**2
  return num / den

def _design_linear_phase_fir(filters, srate, n_taps=FIR_TAPS):
  """Build a linear-phase FIR matching the combined parametric EQ magnitude response.

  Rings symmetrically before + after transient — each side is half the amplitude of
  a min-phase equivalent. Pre-ringing is largely masked by the transient itself,
  making the total effect less audible than concentrated post-ringing."""
  # oversample for a smooth target spectrum
  fft_size = n_taps * 4

  freqs = np.fft.rfftfreq(fft_size, 1.0 / srate)
  w = 2 * np.pi * freqs / srate
  # combined magnitude response (we discard the biquad chain's phase)
  # widen filter Q for smoother impulse response with less ringing
  mag = np.ones_like(freqs)
  for ftype, fc, gain_db, q in filters:
    b, a = _eq_coeff_for_filter(ftype, fc, gain_db, q * FIR_Q_SCALE, srate)
    mag *= np.abs(_biquad_freq_response(b, a, w))

  # zero-phase IFFT → symmetric impulse response centered at n=0
  # (real-valued target magnitude → real cepstrum → zero-phase h)
  h_full = np.fft.irfft(mag, fft_size)

  # shift so center is at n_taps/2 (makes it causal + symmetric)
  h = np.concatenate([h_full[-n_taps // 2:], h_full[:n_taps // 2]])

  # Hann window to suppress Gibbs ripple at the ends
  h *= np.hanning(n_taps)
  return h

class _FFTConvolver:
  """Overlap-add FFT convolution. Stateful across blocks."""
  def __init__(self, fir, block_size):
    self.block_size = block_size
    self.tail_size = len(fir) - 1
    self.fft_size = block_size + len(fir) - 1
    self.H = np.fft.rfft(fir, self.fft_size)
    self.tail = np.zeros(self.tail_size, dtype=np.float64)

  def process(self, block):
    X = np.fft.rfft(block, self.fft_size)
    Y = np.fft.irfft(X * self.H, self.fft_size)
    out = Y[:self.block_size].copy()
    out[:self.tail_size] += self.tail
    self.tail = Y[self.block_size:].copy()
    return out

  def reset(self):
    self.tail[:] = 0.0

def _eq_worker(in_conn, out_conn, sample_rate, eq_filters, preamp_db):
  """Pipeline stage 1: parametric EQ via linear-phase FFT convolution.
  Supports live reconfiguration via ('reconfig_eq', filters, preamp_db) message."""
  state = {'conv': None, 'preamp': 1.0}

  def rebuild(filters, pdb):
    if filters:
      max_boost = max((g for _, _, g, _ in filters if g > 0), default=0)
      total_preamp_db = pdb - max_boost
      state['preamp'] = 10.0 ** (total_preamp_db / 20.0)
      fir = _design_linear_phase_fir(filters, sample_rate)
      state['conv'] = _FFTConvolver(fir, SAMPLE_BUFFER)
    else:
      state['preamp'] = 10.0 ** (pdb / 20.0)
      state['conv'] = None

  rebuild(eq_filters, preamp_db)

  while True:
    msg = in_conn.recv()
    if msg is None:
      out_conn.send(None)
      break
    if msg == 'reset':
      if state['conv']: state['conv'].reset()
      out_conn.send('reset')
      continue
    if isinstance(msg, tuple) and len(msg) >= 1 and isinstance(msg[0], str) and msg[0] == 'reconfig_eq':
      _, new_filters, new_pdb = msg
      rebuild(new_filters, new_pdb)
      out_conn.send(msg)  # forward through pipeline
      continue
    block, flags, raw, vol = msg
    if flags.get('eq', True) and state['conv'] is not None:
      block = state['conv'].process(block * state['preamp'])
      # Catch extreme FIR overshoots only. Any soft-clip with a gentle knee
      # would engage on everyday peaks and alias its tanh harmonics into
      # the audible range — hard-clip is only rarely triggered so it stays
      # out of the way. The final oversampled clip handles serious limiting.
      np.clip(block, -1.0, 1.0, out=block)
    out_conn.send((block, flags, raw, vol))

_NS_C_SRC = r"""
#include <math.h>
void noise_shape(double *out, const double *in, const double *dither, int n,
                 double step, double max_val, double *e1, double *e2) {
  double _e1 = *e1, _e2 = *e2;
  for (int i = 0; i < n; i++) {
    double shaped = in[i] + dither[i] + 1.6 * _e1 - 0.64 * _e2;
    if (shaped > max_val) shaped = max_val;
    if (shaped < -max_val) shaped = -max_val;
    double quantized = round(shaped / step) * step;
    _e2 = _e1;
    _e1 = quantized - shaped;
    out[i] = quantized;
  }
  *e1 = _e1; *e2 = _e2;
}
"""

_ns_lib = _compile_c('noise_shape', _NS_C_SRC)
_ns_lib.noise_shape.restype = None
_ns_lib.noise_shape.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
  ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p,
]

# --- C-accelerated compressor ---

_COMP_C_SRC = r"""
#include <math.h>
void compress(double *out, const double *in, int n,
              double threshold, double ratio, double attack, double release,
              double makeup, double sc_coeff, double *env, double *sc) {
  double e = *env, s = *sc;
  for (int i = 0; i < n; i++) {
    /* sidechain smoothing: one-pole LPF on |in| to stop envelope wobble
     * at 2x band frequency (especially audible in the low band). */
    double rect = fabs(in[i]);
    s = sc_coeff * s + (1.0 - sc_coeff) * rect;
    double level = s;
    if (level > e) e = attack * e + (1.0 - attack) * level;
    else           e = release * e + (1.0 - release) * level;
    double gain = makeup;
    if (e > threshold) {
      double over_db = 20.0 * log10(e / threshold);
      double reduction = over_db * (1.0 - 1.0 / ratio);
      gain *= pow(10.0, -reduction / 20.0);
    }
    out[i] = in[i] * gain;
  }
  *env = e; *sc = s;
}
"""

_comp_lib = _compile_c('compressor', _COMP_C_SRC)
_comp_lib.compress.restype = None
_comp_lib.compress.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
  ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p,
]

def _compress_band(block, threshold, ratio, attack, release, makeup, sc_coeff, env_state):
  """Single-band compressor with sidechain smoothing. C-accelerated.
  env_state = [env, sidechain]."""
  n = len(block)
  inp = np.ascontiguousarray(block, dtype=np.float64)
  out = np.empty(n, dtype=np.float64)
  env = np.array([env_state[0]])
  sc = np.array([env_state[1]])
  _comp_lib.compress(out.ctypes.data, inp.ctypes.data, n,
                     threshold, ratio, attack, release, makeup, sc_coeff,
                     env.ctypes.data, sc.ctypes.data)
  env_state[0] = env[0]; env_state[1] = sc[0]
  return out

_ns_rng = np.random.default_rng()

def _noise_shape_12bit(block, state):
  """Noise-shaped TPDF dither for 12-bit DAC. C-accelerated."""
  STEP = 1.0 / 4096.0
  MAX_VAL = 0.5 - STEP
  n = len(block)
  inp = np.ascontiguousarray(block, dtype=np.float64)
  out = np.empty(n, dtype=np.float64)
  dither = (_ns_rng.uniform(-0.5, 0.5, n) + _ns_rng.uniform(-0.5, 0.5, n)) * STEP
  e1 = np.array([state[0]]); e2 = np.array([state[1]])
  _ns_lib.noise_shape(out.ctypes.data, inp.ctypes.data, dither.ctypes.data, n,
                      STEP, MAX_VAL, e1.ctypes.data, e2.ctypes.data)
  state[0] = e1[0]; state[1] = e2[0]
  return out

# --- transient sharpener constants ---
TR_FAST_MS = 1.0      # ms — fast envelope (catches transients)
TR_SLOW_MS = 50.0     # ms — slow envelope (follows sustain)
TR_AMOUNT = 0.0       # gain multiplier on the transient difference

_TR_C_SRC = r"""
#include <math.h>
void transient_sharpen(double *out, const double *in, int n,
                       double fast_coeff, double slow_coeff, double amount,
                       double *fast_env, double *slow_env) {
  double fe = *fast_env, se = *slow_env;
  for (int i = 0; i < n; i++) {
    double level = fabs(in[i]);
    /* fast envelope: fast attack + fast release — tracks the transient edge */
    if (level > fe) fe = fe + (1.0 - fast_coeff) * (level - fe);
    else            fe = fe + (1.0 - fast_coeff) * 0.5 * (level - fe);
    /* slow envelope: slow attack + slow release — tracks the sustained body */
    if (level > se) se = se + (1.0 - slow_coeff) * (level - se);
    else            se = se + (1.0 - slow_coeff) * (level - se);
    /* transient = fast minus slow; boost when fast > slow (attack phase) */
    double diff = fe - se;
    double gain = 1.0;
    if (diff > 0.0 && se > 1e-10) {
      gain = 1.0 + amount * (diff / (se + diff));
    }
    out[i] = in[i] * gain;
  }
  *fast_env = fe; *slow_env = se;
}
"""

_tr_lib = _compile_c('transient', _TR_C_SRC)
_tr_lib.transient_sharpen.restype = None
_tr_lib.transient_sharpen.argtypes = [
  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
  ctypes.c_double, ctypes.c_double, ctypes.c_double,
  ctypes.c_void_p, ctypes.c_void_p,
]

def _transient_sharpen(block, fast_coeff, slow_coeff, amount, env_state):
  """Transient sharpener. C-accelerated."""
  n = len(block)
  inp = np.ascontiguousarray(block, dtype=np.float64)
  out = np.empty(n, dtype=np.float64)
  fe = np.array([env_state[0]]); se = np.array([env_state[1]])
  _tr_lib.transient_sharpen(out.ctypes.data, inp.ctypes.data, n,
                            fast_coeff, slow_coeff, amount,
                            fe.ctypes.data, se.ctypes.data)
  env_state[0] = fe[0]; env_state[1] = se[0]
  return out

ANTIALIAS_FREQ = 16000.0  # Hz — roll off above this to reduce DAC harmonic foldback

# --- multiband compressor constants ---
# 3-band split: low / mid / high
MB_XOVER_LOW = 500.0     # Hz — low/mid crossover
MB_XOVER_HIGH = 4000.0   # Hz — mid/high crossover
MB_THRESHOLD = -18.0      # dBFS — compress above this
MB_RATIO = 4.0            # compression ratio
MB_ATTACK_MS = 5.0        # ms
MB_RELEASE_MS = 100.0     # ms
MB_SIDECHAIN_MS = 8.0     # ms — sidechain smoother (tames envelope wobble in low band)
MB_MAKEUP_DB = 2.5        # dB — modest per-band makeup; 3-band sum already ~adds up

SOFT_CLIP_CEILING = 0.48  # leave ~4% headroom under the ±0.5 DAC rail for AA-LPF overshoot
SOFT_CLIP_KNEE = 0.7

def _soft_clip(block, ceiling=SOFT_CLIP_CEILING, knee=SOFT_CLIP_KNEE):
  """Linear below knee*ceiling, tanh saturation above. Unity gain in linear region,
  smooth transition, saturates at ±ceiling."""
  threshold = knee * ceiling
  headroom = ceiling - threshold
  x = np.abs(block)
  tail = np.tanh(np.maximum(x - threshold, 0.0) / headroom) * headroom
  return np.sign(block) * (np.minimum(x, threshold) + tail)

# --- 2x oversampled soft clip (prevents HF aliasing from tanh harmonics) ---

HALFBAND_TAPS = 23  # Kaiser-windowed halfband FIR; odd, symmetric
def _design_halfband(n_taps):
  """Halfband LPF: cutoff at fs_up/4 (= original Nyquist). Every 2nd tap is 0
  except the center, which is 0.5 — classic halfband property, lets us skip
  half the multiplies in the polyphase forms."""
  n = np.arange(n_taps) - (n_taps - 1) / 2.0
  h = np.sinc(n / 2.0) / 2.0        # ideal half-rate sinc
  h *= np.kaiser(n_taps, 8.0)        # Kaiser window for >60 dB stopband
  h /= np.sum(h)                     # unity DC gain
  return h

class _FIRState:
  """Streaming FIR with overlap-save — produces exactly len(x) output samples
  per len(x) input samples, continuous across blocks."""
  def __init__(self, h):
    self.h = h
    self.hist = np.zeros(len(h) - 1, dtype=np.float64)
  def process(self, x):
    sig = np.concatenate([self.hist, x])
    out = np.convolve(sig, self.h, mode='valid')  # length == len(x)
    self.hist = x[-(len(self.h) - 1):].copy()
    return out
  def reset(self):
    self.hist[:] = 0.0

class _Oversample2xClipper:
  """Upsample 2×, apply clipper at 2× rate, LPF, decimate. Harmonics produced
  by the non-linearity up to 48 kHz are correctly represented before the LPF
  removes the ultrasonic portion, so none of them alias into the audible band."""
  def __init__(self, clipper, n_taps=HALFBAND_TAPS):
    h = _design_halfband(n_taps)
    self.clipper = clipper
    self.up_fir = _FIRState(h * 2.0)  # ×2 compensates for zero-stuffing loss
    self.dn_fir = _FIRState(h)
  def process(self, block):
    stuffed = np.zeros(len(block) * 2, dtype=np.float64)
    stuffed[::2] = block
    up = self.up_fir.process(stuffed)
    clipped = self.clipper(up)
    dn = self.dn_fir.process(clipped)
    return dn[::2]
  def reset(self):
    self.up_fir.reset()
    self.dn_fir.reset()

def _bk_worker(in_conn, out_conn, sample_rate):
  """Pipeline stage 2: sub-band harmonic bass synthesis + subsonic filter."""
  Q = 1.0 / math.sqrt(2.0)
  # per-sub-band input bandpass filters
  band_filters = []
  for lo, hi in BK_BANDS:
    c, s = _make_coeffs_and_states([
      _highpass_coeffs(lo, Q, sample_rate),
      _lowpass_coeffs(hi, Q, sample_rate)])
    band_filters.append((c, s))
  # shared output bandpass: only pass harmonics in the speaker's useful range
  out_c, out_s = _make_coeffs_and_states([
    _highpass_coeffs(BK_OUTPUT_HP, Q, sample_rate),
    _lowpass_coeffs(BK_OUTPUT_LP, Q, sample_rate)])
  # subsonic filter
  sub_c, sub_s = _make_coeffs_and_states([
    _highpass_coeffs(SUBSONIC_FREQ, Q, sample_rate),
    _highpass_coeffs(SUBSONIC_FREQ, Q, sample_rate)])

  all_states = []
  for c, s in band_filters:
    all_states.extend(s)
  all_states.extend(out_s)
  all_states.extend(sub_s)

  while True:
    msg = in_conn.recv()
    if msg is None:
      out_conn.send(None)
      break
    if msg == 'reset':
      for s in all_states:
        s[:] = [0.0, 0.0, 0.0, 0.0]
      out_conn.send('reset')
      continue
    if isinstance(msg, tuple) and len(msg) >= 1 and isinstance(msg[0], str) and msg[0] == 'reconfig_eq':
      out_conn.send(msg)
      continue
    block, flags, raw, vol = msg

    # Always run all filters so their states stay in sync with the input —
    # prevents a click/pop when toggling the bass flag back on.
    sig = np.clip(block, -1, 1)
    harmonic_sum = np.zeros(len(block), dtype=np.float64)
    for bp_c, bp_s in band_filters:
      bass = _fused_process(sig, bp_c, bp_s)
      bass = np.tanh(bass * 2.0) * 0.5
      b2 = bass * bass
      b3 = b2 * bass
      harmonic_sum += (b2 * BK_H2_AMT +
                       b3 * BK_H3_AMT +
                       b2 * b2 * BK_H4_AMT +
                       b2 * b3 * BK_H5_AMT)
    harmonic_sum = np.tanh(harmonic_sum)
    harmonic_sum = _fused_process(harmonic_sum, out_c, out_s)

    if flags.get('bass', True):
      block = block + harmonic_sum

    block = _fused_process(block, sub_c, sub_s)
    out_conn.send((block, flags, raw, vol))

# --- loudness contour (Fletcher-Munson compensation) ---
LC_BASS_FC = 200.0      # Hz
LC_BASS_MAX_DB = 10.0   # dB boost at minimum volume
LC_TREBLE_FC = 8000.0   # Hz
LC_TREBLE_MAX_DB = 5.0  # dB boost at minimum volume

def _out_worker(in_conn, out_conn, sample_rate):
  """Pipeline stage 3: transients + crystallizer + loudness + multiband comp + soft clip + AA + NS."""
  Q = 1.0 / math.sqrt(2.0)
  # loudness contour shelves (precomputed at max boost — blended at runtime)
  lc_bass_c, lc_bass_s = _make_coeffs_and_states([
    _lowshelf_coeffs(LC_BASS_FC, LC_BASS_MAX_DB, Q, sample_rate)])
  lc_treb_c, lc_treb_s = _make_coeffs_and_states([
    _highshelf_coeffs(LC_TREBLE_FC, LC_TREBLE_MAX_DB, Q, sample_rate)])
  # crystallizer
  cr_pre_c, cr_pre_s = _make_coeffs_and_states([
    _highpass_coeffs(CR_FLOOR, Q, sample_rate),
    _lowpass_coeffs(CR_CEIL, Q, sample_rate)])
  cr_post_c, cr_post_s = _make_coeffs_and_states([
    _highpass_coeffs(CR_OUTPUT_HP, Q, sample_rate),
    _lowpass_coeffs(CR_OUTPUT_LP, Q, sample_rate)])
  # multiband crossover filters (Linkwitz-Riley: 2x Butterworth cascaded = -12dB/oct per xover)
  mb_lo_c, mb_lo_s = _make_coeffs_and_states([
    _lowpass_coeffs(MB_XOVER_LOW, Q, sample_rate),
    _lowpass_coeffs(MB_XOVER_LOW, Q, sample_rate)])
  mb_hi_c, mb_hi_s = _make_coeffs_and_states([
    _highpass_coeffs(MB_XOVER_HIGH, Q, sample_rate),
    _highpass_coeffs(MB_XOVER_HIGH, Q, sample_rate)])
  mb_mid_lo_c, mb_mid_lo_s = _make_coeffs_and_states([
    _highpass_coeffs(MB_XOVER_LOW, Q, sample_rate),
    _highpass_coeffs(MB_XOVER_LOW, Q, sample_rate)])
  mb_mid_hi_c, mb_mid_hi_s = _make_coeffs_and_states([
    _lowpass_coeffs(MB_XOVER_HIGH, Q, sample_rate),
    _lowpass_coeffs(MB_XOVER_HIGH, Q, sample_rate)])
  # per-band compressor envelope states
  threshold = 10.0 ** (MB_THRESHOLD / 20.0)
  attack = math.exp(-1.0 / (sample_rate * MB_ATTACK_MS / 1000.0))
  release = math.exp(-1.0 / (sample_rate * MB_RELEASE_MS / 1000.0))
  sc_coeff = math.exp(-1.0 / (sample_rate * MB_SIDECHAIN_MS / 1000.0))
  makeup = 10.0 ** (MB_MAKEUP_DB / 20.0)
  comp_env = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # [env, sidechain] per band
  # 4th-order anti-alias LPF
  aa_c, aa_s = _make_coeffs_and_states([
    _lowpass_coeffs(ANTIALIAS_FREQ, Q, sample_rate),
    _lowpass_coeffs(ANTIALIAS_FREQ, Q, sample_rate)])
  # transient sharpener
  tr_fast = math.exp(-1.0 / (sample_rate * TR_FAST_MS / 1000.0))
  tr_slow = math.exp(-1.0 / (sample_rate * TR_SLOW_MS / 1000.0))
  tr_env = [0.0, 0.0]  # fast_env, slow_env
  ns_state = [0.0, 0.0]
  # 2x oversampled soft clip — prevents tanh harmonics above Nyquist from
  # aliasing back into the audible band (shows up as HF "ringing" at high vol)
  oversampled_clip = _Oversample2xClipper(_soft_clip)

  while True:
    msg = in_conn.recv()
    if msg is None:
      out_conn.send(None)
      break
    if msg == 'reset':
      for s in (lc_bass_s + lc_treb_s + cr_pre_s + cr_post_s +
                mb_lo_s + mb_hi_s + mb_mid_lo_s + mb_mid_hi_s + aa_s):
        s[:] = [0.0, 0.0, 0.0, 0.0]
      for e in comp_env: e[0] = e[1] = 0.0
      tr_env[0] = tr_env[1] = 0.0
      ns_state[0] = ns_state[1] = 0.0
      oversampled_clip.reset()
      out_conn.send('reset')
      continue
    if isinstance(msg, tuple) and len(msg) >= 1 and isinstance(msg[0], str) and msg[0] == 'reconfig_eq':
      out_conn.send(msg)
      continue
    block, flags, raw, vol = msg

    # Every stateful stage below runs unconditionally so its state stays in
    # sync with the live audio. Flags only gate whether the result is mixed
    # back into `block` — prevents clicks when a flag is toggled on.

    block_ts = _transient_sharpen(block, tr_fast, tr_slow, TR_AMOUNT, tr_env)
    if flags.get('transient', True):
      block = block_ts

    presence = _fused_process(block, cr_pre_c, cr_pre_s)
    sparkle = np.tanh(presence * CR_DRIVE) * CR_AMT
    sparkle = _fused_process(sparkle, cr_post_c, cr_post_s)
    if flags.get('crystal', True):
      block = block + sparkle

    bass_boosted = _fused_process(block, lc_bass_c, lc_bass_s)
    treb_boosted = _fused_process(bass_boosted, lc_treb_c, lc_treb_s)
    if flags.get('loudness', True):
      if vol > 1e-6:
        vol_db = 20.0 * math.log10(vol)
        lc_mix = max(0.0, min(1.0, -vol_db / 20.0))
      else:
        lc_mix = 1.0
      block = block + lc_mix * (treb_boosted - block)

    lo = _fused_process(block, mb_lo_c, mb_lo_s)
    mid = _fused_process(block, mb_mid_lo_c, mb_mid_lo_s)
    mid = _fused_process(mid, mb_mid_hi_c, mb_mid_hi_s)
    hi = _fused_process(block, mb_hi_c, mb_hi_s)
    lo = _compress_band(lo, threshold, MB_RATIO, attack, release, makeup, sc_coeff, comp_env[0])
    mid = _compress_band(mid, threshold, MB_RATIO, attack, release, makeup, sc_coeff, comp_env[1])
    hi = _compress_band(hi, threshold, MB_RATIO, attack, release, makeup, sc_coeff, comp_env[2])
    if flags.get('mbcomp', True):
      block = lo + mid + hi

    # volume + DAC range
    block = block * vol * 0.5

    if raw:
      # raw mode: just clip to DAC range, no processing
      block = np.clip(block, -0.5, 0.5)
    else:
      # 2x oversampled soft clip — non-linearity runs at 96 kHz so tanh
      # harmonics beyond Nyquist can't fold into the audible band. Always
      # go through the oversampler so its filter state stays continuous —
      # gating it by peak would cause a click whenever it started up.
      block = oversampled_clip.process(block)

      # anti-alias LPF
      block = _fused_process(block, aa_c, aa_s)

      # noise-shape for 12-bit DAC
      block = _noise_shape_12bit(block, ns_state)

    out_conn.send(block)


class ProcessingPipeline:
  def __init__(self, samples, sample_rate, eq_path, preamp_db=0.0):
    self.samples = samples
    self.sample_rate = sample_rate
    self.buf = queue.Queue(maxsize=LOOKAHEAD)
    self.write_pos = 0
    self.flags = {'eq': True, 'bass': True, 'transient': True,
                  'crystal': True, 'loudness': True, 'mbcomp': True}
    self.raw = False
    self.vol = 0.5
    self._stop = threading.Event()
    self._seek_to = None
    self._lock = threading.Lock()

    eq_filters, file_preamp_db = _parse_eq_file(eq_path) if eq_path else ([], 0.0)
    total_preamp = file_preamp_db + preamp_db
    if eq_filters:
      print(f"Loaded {len(eq_filters)} EQ bands from {eq_path} (preamp: {total_preamp:+.1f} dB)")

    # pipeline: feeder -> eq -> bk -> out -> feeder collects
    self._to_eq, eq_in = Pipe()
    eq_out, bk_in = Pipe()
    bk_out, out_in = Pipe()
    out_out, self._from_out = Pipe()

    self._eq_proc = Process(target=_eq_worker, args=(eq_in, eq_out, sample_rate, eq_filters, total_preamp), daemon=True)
    self._bk_proc = Process(target=_bk_worker, args=(bk_in, bk_out, sample_rate), daemon=True)
    self._out_proc = Process(target=_out_worker, args=(out_in, out_out, sample_rate), daemon=True)
    self._eq_proc.start()
    self._bk_proc.start()
    self._out_proc.start()

    # close child ends in parent
    eq_in.close(); eq_out.close(); bk_in.close(); bk_out.close(); out_in.close(); out_out.close()

    self._feeder = threading.Thread(target=self._feed, daemon=True)
    self._feeder.start()

  def _flush(self):
    while not self.buf.empty():
      try: self.buf.get_nowait()
      except queue.Empty: break

  def _feed(self):
    # pipeline: while bk processes block N, eq processes block N+1
    bk_pending = None  # (start, end) of block currently in bk_worker

    while not self._stop.is_set():
      # handle seek
      with self._lock:
        seek_pos = self._seek_to
        self._seek_to = None
      if seek_pos is not None:
        # drain any in-flight result from bk
        if bk_pending is not None:
          self._from_out.recv()
          bk_pending = None
        self._flush()
        self.write_pos = seek_pos
        # reset both workers (propagates through pipeline)
        self._to_eq.send('reset')
        self._from_out.recv()  # wait for reset to flow through
        continue

      # collect bk result if ready and queue has room
      if bk_pending is not None and self._from_out.poll():
        result = self._from_out.recv()
        self.buf.put((*bk_pending, result))
        bk_pending = None

      # if bk is busy or queue is full, wait
      if bk_pending is not None or self.buf.full():
        time.sleep(0.0005)
        continue

      # done?
      if self.write_pos >= len(self.samples):
        time.sleep(0.001)
        continue

      # read next block and send through pipeline
      end = min(self.write_pos + SAMPLE_BUFFER, len(self.samples))
      block = np.ascontiguousarray(self.samples[self.write_pos:end, 0])
      wp = self.write_pos

      self._to_eq.send((block, self.flags, self.raw, self.vol))
      bk_pending = (wp, end)
      self.write_pos = end

    # on stop, wait for any pending result
    if bk_pending is not None:
      try: self._from_out.recv()
      except: pass

  def wait_ready(self):
    """Block until the lookahead buffer is full."""
    while self.buf.qsize() < min(LOOKAHEAD, len(self.samples) // SAMPLE_BUFFER):
      time.sleep(0.01)

  def get_block(self):
    try:
      return self.buf.get_nowait()
    except queue.Empty:
      return None

  def seek(self, new_pos):
    with self._lock:
      self._seek_to = new_pos

  def reset_filters(self):
    self.seek(self.write_pos)

  def stop(self):
    self._stop.set()
    self._feeder.join(timeout=1)
    self._to_eq.send(None)
    try:
      self._from_out.recv()  # None flows through
    except: pass
    self._eq_proc.join(timeout=1)
    self._bk_proc.join(timeout=1)
    self._out_proc.join(timeout=1)


# --- player ---

def read_key():
  if not select.select([sys.stdin], [], [], 0)[0]:
    return None
  ch = sys.stdin.read(1)
  if ch == '\x1b':
    if select.select([sys.stdin], [], [], 0)[0]:
      ch2 = sys.stdin.read(1)
      if ch2 == '[' and select.select([sys.stdin], [], [], 0)[0]:
        return {'D': 'left', 'C': 'right'}.get(sys.stdin.read(1))
  return ch

def load_wav(path):
  with open(path, 'rb') as f:
    riff = f.read(12)
    assert riff[:4] == b'RIFF' and riff[8:12] == b'WAVE'

    fmt_tag = channels = sample_rate = sample_width = None
    data = None

    while True:
      chunk_hdr = f.read(8)
      if len(chunk_hdr) < 8:
        break
      chunk_id = chunk_hdr[:4]
      chunk_size = struct.unpack('<I', chunk_hdr[4:8])[0]

      if chunk_id == b'fmt ':
        fmt = f.read(chunk_size)
        fmt_tag, channels, sample_rate = struct.unpack('<HHI', fmt[:8])
        sample_width = struct.unpack('<H', fmt[14:16])[0]
      elif chunk_id == b'data':
        data = f.read(chunk_size)
      else:
        f.read(chunk_size)

      if chunk_size % 2:
        f.read(1)

  assert fmt_tag in (1, 3), f"unsupported WAV format: {fmt_tag}"
  assert data is not None

  if fmt_tag == 3:
    samples = np.frombuffer(data, dtype=np.float32)
  elif sample_width == 16:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (2**15)
  elif sample_width == 24:
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    i32 = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
    i32[i32 >= 0x800000] -= 0x1000000
    samples = i32.astype(np.float32) / (2**23)
  elif sample_width == 32:
    samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / (2**31)
  else:
    raise ValueError(f"unsupported bit depth: {sample_width}")

  # downmix to mono
  if channels > 1:
    samples = samples.reshape(-1, channels).mean(axis=1, keepdims=True)
  else:
    samples = samples.reshape(-1, 1)

  return samples, sample_rate

def play_wav(path: str, volume: float = 0.5, eq_path: str = None, preamp_db: float = 0.0):
  samples, sample_rate = load_wav(path)

  sd._terminate()
  sd._initialize()

  pipe = ProcessingPipeline(samples, sample_rate, eq_path, preamp_db)
  pipe.vol = volume
  pipe.wait_ready()
  frame = [0]
  paused = [False]

  def callback(data_out: np.ndarray, frames: int, time_info, status):
    if status:
      print(f"\n  stream status: {status}", file=sys.stderr)
    if paused[0]:
      data_out[:] = 0
      return

    result = pipe.get_block()
    if result is None:
      data_out[:] = 0
      return

    start, end, block = result
    n = len(block)
    data_out[:n, 0] = block.astype(np.float32)
    data_out[n:] = 0
    frame[0] = end
    if end >= len(samples):
      raise sd.CallbackStop

  total_sec = len(samples) / sample_rate
  seek_frames = int(SEEK_SECONDS * sample_rate)
  bar_width = 40

  old_settings = termios.tcgetattr(sys.stdin)
  try:
    tty.setraw(sys.stdin.fileno())

    with sd.OutputStream(channels=1, samplerate=sample_rate,
                          callback=callback, blocksize=SAMPLE_BUFFER,
                          finished_callback=lambda: None) as stream:
      while stream.active:
        key = read_key()
        if key == ' ':
          paused[0] = not paused[0]
        elif key == 'q':
          break
        elif key == 'e':
          # toggle ALL DSP at once
          all_on = all(pipe.flags.values())
          for k in pipe.flags: pipe.flags[k] = not all_on
          if not all_on:
            pipe.reset_filters()
        elif key == 'r':
          pipe.raw = not pipe.raw
        # individual stage toggles
        elif key == '1':
          pipe.flags['eq'] = not pipe.flags['eq']
        elif key == '2':
          pipe.flags['bass'] = not pipe.flags['bass']
        elif key == '3':
          pipe.flags['transient'] = not pipe.flags['transient']
        elif key == '4':
          pipe.flags['crystal'] = not pipe.flags['crystal']
        elif key == '5':
          pipe.flags['loudness'] = not pipe.flags['loudness']
        elif key == '6':
          pipe.flags['mbcomp'] = not pipe.flags['mbcomp']
        elif key in ('left', 'a'):
          new = max(0, frame[0] - seek_frames)
          frame[0] = new
          pipe.seek(new)
        elif key in ('right', 'd'):
          new = min(len(samples), frame[0] + seek_frames)
          frame[0] = new
          pipe.seek(new)
        elif key == 'w':
          pipe.vol = min(pipe.vol + 0.1, 2.0)
        elif key == 's':
          pipe.vol = max(pipe.vol - 0.1, 0.0)

        pos = frame[0] / sample_rate
        pct = frame[0] / len(samples)
        filled = int(bar_width * pct)
        bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
        state = 'PAUSED' if paused[0] else '\u25b6'
        # compact flags display: EQ BA TR CR LD MB
        short = {'eq': 'EQ', 'bass': 'BA', 'transient': 'TR', 'crystal': 'CR',
                 'loudness': 'LD', 'mbcomp': 'MB'}
        fxlabel = ' ' + ' '.join(s if pipe.flags[k] else '·' * len(s) for k, s in short.items())
        if pipe.raw:
          fxlabel += ' RAW'
        sys.stdout.write(f'\r  {state} {pos:.1f}s / {total_sec:.1f}s  [{bar}] {pct*100:5.1f}%  vol:{pipe.vol:.1f}{fxlabel}  ')
        sys.stdout.flush()
        sd.sleep(50)

      pos = frame[0] / sample_rate
      sys.stdout.write(f'\r  {pos:.1f}s / {total_sec:.1f}s  [{"\u2588" * bar_width}] 100.0%  \n')
  finally:
    pipe.stop()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <file.wav> [volume] [--eq <file>] [--preamp <dB>]")
    print(f"  Space: pause/resume  a/d: seek {SEEK_SECONDS}s  w/s: volume")
    print(f"  e: toggle all DSP  r: raw (bypass output stage)  q: quit")
    print(f"  1:EQ 2:bass 3:transient 4:crystal 5:loudness 6:mbcomp")
    sys.exit(1)

  args = []
  script_dir = os.path.dirname(os.path.abspath(__file__))
  eq_path = os.path.join(script_dir, 'mici.txt')
  preamp_db = None  # None = auto-calculate
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == '--eq' and i + 1 < len(sys.argv):
      i += 1
      eq_path = sys.argv[i]
    elif sys.argv[i] == '--preamp' and i + 1 < len(sys.argv):
      i += 1
      preamp_db = float(sys.argv[i])
    elif not sys.argv[i].startswith('--'):
      args.append(sys.argv[i])
    i += 1

  # auto-calculate preamp if not specified
  if preamp_db is None and eq_path:
    filters, _ = _parse_eq_file(eq_path)
    if filters:
      gains = sorted(g for _, _, g, _ in filters)
      median_gain = gains[len(gains) // 2]
      preamp_db = -median_gain / 2.0
      print(f"Auto preamp: {preamp_db:+.1f} dB (median filter gain: {median_gain:.1f} dB)")
    else:
      preamp_db = 0.0
  elif preamp_db is None:
    preamp_db = 0.0

  wav_path = args[0]
  vol = float(args[1]) if len(args) > 1 else 0.5
  play_wav(wav_path, vol, eq_path=eq_path, preamp_db=preamp_db)
