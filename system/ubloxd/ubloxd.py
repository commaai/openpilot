#!/usr/bin/env python3
import math
import capnp
import calendar
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from cereal import log
from cereal import messaging
from openpilot.system.ubloxd.generated.ubx import Ubx
from openpilot.system.ubloxd.generated.gps import Gps
from openpilot.system.ubloxd.generated.glonass import Glonass


SECS_IN_MIN = 60
SECS_IN_HR = 60 * SECS_IN_MIN
SECS_IN_DAY = 24 * SECS_IN_HR
SECS_IN_WEEK = 7 * SECS_IN_DAY


class UbxFramer:
  PREAMBLE1 = 0xB5
  PREAMBLE2 = 0x62
  HEADER_SIZE = 6
  CHECKSUM_SIZE = 2

  def __init__(self) -> None:
    self.buf = bytearray()
    self.last_log_time = 0.0

  def reset(self) -> None:
    self.buf.clear()

  @staticmethod
  def _checksum_ok(frame: bytes) -> bool:
    ck_a = 0
    ck_b = 0
    for b in frame[2:-2]:
      ck_a = (ck_a + b) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    return ck_a == frame[-2] and ck_b == frame[-1]

  def add_data(self, log_time: float, incoming: bytes) -> list[bytes]:
    self.last_log_time = log_time
    out: list[bytes] = []
    if not incoming:
      return out
    self.buf += incoming

    while True:
      # find preamble
      if len(self.buf) < 2:
        break
      start = self.buf.find(b"\xB5\x62")
      if start < 0:
        # no preamble in buffer
        self.buf.clear()
        break
      if start > 0:
        # drop garbage before preamble
        self.buf = self.buf[start:]

      if len(self.buf) < self.HEADER_SIZE:
        break

      length_le = int.from_bytes(self.buf[4:6], 'little', signed=False)
      total_len = self.HEADER_SIZE + length_le + self.CHECKSUM_SIZE
      if len(self.buf) < total_len:
        break

      candidate = bytes(self.buf[:total_len])
      if self._checksum_ok(candidate):
        out.append(candidate)
        # consume this frame
        self.buf = self.buf[total_len:]
      else:
        # drop first byte and retry
        self.buf = self.buf[1:]

    return out


def _bit(b: int, shift: int) -> bool:
  return (b & (1 << shift)) != 0


@dataclass
class EphemerisCaches:
  gps_subframes: defaultdict[int, dict[int, bytes]]
  glonass_strings: defaultdict[int, dict[int, bytes]]
  glonass_string_times: defaultdict[int, dict[int, float]]
  glonass_string_superframes: defaultdict[int, dict[int, int]]


class UbloxMsgParser:
  gpsPi = 3.1415926535898

  # user range accuracy in meters
  glonass_URA_lookup: dict[int, float] = {
    0: 1, 1: 2, 2: 2.5, 3: 4, 4: 5, 5: 7,
    6: 10, 7: 12, 8: 14, 9: 16, 10: 32,
    11: 64, 12: 128, 13: 256, 14: 512, 15: 1024,
  }

  def __init__(self) -> None:
    self.framer = UbxFramer()
    self.caches = EphemerisCaches(
      gps_subframes=defaultdict(dict),
      glonass_strings=defaultdict(dict),
      glonass_string_times=defaultdict(dict),
      glonass_string_superframes=defaultdict(dict),
    )

  # Message generation entry point
  def parse_frame(self, frame: bytes) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder] | None:
    # Quick header parse
    msg_type = int.from_bytes(frame[2:4], 'big')
    payload = frame[6:-2]
    if msg_type == 0x0107:
      body = Ubx.NavPvt.from_bytes(payload)
      return self._gen_nav_pvt(body)
    if msg_type == 0x0213:
      # Manually parse RXM-SFRBX to avoid Kaitai EOF on some frames
      if len(payload) < 8:
        return None
      gnss_id = payload[0]
      sv_id = payload[1]
      freq_id = payload[3]
      num_words = payload[4]
      exp = 8 + 4 * num_words
      if exp != len(payload):
        return None
      words: list[int] = []
      off = 8
      for _ in range(num_words):
        words.append(int.from_bytes(payload[off:off+4], 'little'))
        off += 4

      class _SfrbxView:
        def __init__(self, gid: int, sid: int, fid: int, body: list[int]):
          self.gnss_id = Ubx.GnssType(gid)
          self.sv_id = sid
          self.freq_id = fid
          self.body = body
      view = _SfrbxView(gnss_id, sv_id, freq_id, words)
      return self._gen_rxm_sfrbx(view)
    if msg_type == 0x0215:
      body = Ubx.RxmRawx.from_bytes(payload)
      return self._gen_rxm_rawx(body)
    if msg_type == 0x0A09:
      body = Ubx.MonHw.from_bytes(payload)
      return self._gen_mon_hw(body)
    if msg_type == 0x0A0B:
      body = Ubx.MonHw2.from_bytes(payload)
      return self._gen_mon_hw2(body)
    if msg_type == 0x0135:
      body = Ubx.NavSat.from_bytes(payload)
      return self._gen_nav_sat(body)
    return None

  # NAV-PVT -> gpsLocationExternal
  def _gen_nav_pvt(self, msg: Ubx.NavPvt) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder]:
    dat = messaging.new_message('gpsLocationExternal', valid=True)
    gps = dat.gpsLocationExternal
    gps.source = log.GpsLocationData.SensorSource.ublox
    gps.flags = msg.flags
    gps.hasFix = (msg.flags % 2) == 1
    gps.latitude = msg.lat * 1e-07
    gps.longitude = msg.lon * 1e-07
    gps.altitude = msg.height * 1e-03
    gps.speed = msg.g_speed * 1e-03
    gps.bearingDeg = msg.head_mot * 1e-5
    gps.horizontalAccuracy = msg.h_acc * 1e-03
    gps.satelliteCount = msg.num_sv

    # build UTC timestamp millis (NAV-PVT is in UTC)
    # tolerate invalid or unset date values like C++ timegm
    try:
      utc_tt = calendar.timegm((msg.year, msg.month, msg.day, msg.hour, msg.min, msg.sec, 0, 0, 0))
    except Exception:
      utc_tt = 0
    gps.unixTimestampMillis = int(utc_tt * 1e3 + (msg.nano * 1e-6))

    # match C++ float32 rounding semantics exactly
    gps.vNED = [
      float(np.float32(msg.vel_n) * np.float32(1e-03)),
      float(np.float32(msg.vel_e) * np.float32(1e-03)),
      float(np.float32(msg.vel_d) * np.float32(1e-03)),
    ]
    gps.verticalAccuracy = msg.v_acc * 1e-03
    gps.speedAccuracy = msg.s_acc * 1e-03
    gps.bearingAccuracyDeg = msg.head_acc * 1e-05
    return ('gpsLocationExternal', dat)

  # RXM-SFRBX dispatch to GPS or GLONASS ephemeris
  def _gen_rxm_sfrbx(self, msg) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder] | None:
    if msg.gnss_id == Ubx.GnssType.gps:
      return self._parse_gps_ephemeris(msg)
    if msg.gnss_id == Ubx.GnssType.glonass:
      return self._parse_glonass_ephemeris(msg)
    return None

  def _parse_gps_ephemeris(self, msg: Ubx.RxmSfrbx) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder] | None:
    # body is list of 10 words; convert to 30-byte subframe (strip parity/padding)
    body = msg.body
    if len(body) != 10:
      return None
    subframe_data = bytearray()
    for word in body:
      word >>= 6
      subframe_data.append((word >> 16) & 0xFF)
      subframe_data.append((word >> 8) & 0xFF)
      subframe_data.append(word & 0xFF)

    sf = Gps.from_bytes(bytes(subframe_data))
    subframe_id = sf.how.subframe_id
    if subframe_id < 1 or subframe_id > 3:
      return None
    self.caches.gps_subframes[msg.sv_id][subframe_id] = bytes(subframe_data)

    if len(self.caches.gps_subframes[msg.sv_id]) != 3:
      return None

    dat = messaging.new_message('ubloxGnss', valid=True)
    eph = dat.ubloxGnss.init('ephemeris')
    eph.svId = msg.sv_id

    iode_s2 = 0
    iode_s3 = 0
    iodc_lsb = 0
    week = 0

    # Subframe 1
    sf1 = Gps.from_bytes(self.caches.gps_subframes[msg.sv_id][1])
    s1 = sf1.body
    assert isinstance(s1, Gps.Subframe1)
    week = s1.week_no
    week += 1024
    if week < 1877:
      week += 1024
    eph.tgd = s1.t_gd * math.pow(2, -31)
    eph.toc = s1.t_oc * math.pow(2, 4)
    eph.af2 = s1.af_2 * math.pow(2, -55)
    eph.af1 = s1.af_1 * math.pow(2, -43)
    eph.af0 = s1.af_0 * math.pow(2, -31)
    eph.svHealth = s1.sv_health
    eph.towCount = sf1.how.tow_count
    iodc_lsb = s1.iodc_lsb

    # Subframe 2
    sf2 = Gps.from_bytes(self.caches.gps_subframes[msg.sv_id][2])
    s2 = sf2.body
    assert isinstance(s2, Gps.Subframe2)
    if s2.t_oe == 0 and sf2.how.tow_count * 6 >= (SECS_IN_WEEK - 2 * SECS_IN_HR):
      week += 1
    eph.crs = s2.c_rs * math.pow(2, -5)
    eph.deltaN = s2.delta_n * math.pow(2, -43) * self.gpsPi
    eph.m0 = s2.m_0 * math.pow(2, -31) * self.gpsPi
    eph.cuc = s2.c_uc * math.pow(2, -29)
    eph.ecc = s2.e * math.pow(2, -33)
    eph.cus = s2.c_us * math.pow(2, -29)
    eph.a = math.pow(s2.sqrt_a * math.pow(2, -19), 2.0)
    eph.toe = s2.t_oe * math.pow(2, 4)
    iode_s2 = s2.iode

    # Subframe 3
    sf3 = Gps.from_bytes(self.caches.gps_subframes[msg.sv_id][3])
    s3 = sf3.body
    assert isinstance(s3, Gps.Subframe3)
    eph.cic = s3.c_ic * math.pow(2, -29)
    eph.omega0 = s3.omega_0 * math.pow(2, -31) * self.gpsPi
    eph.cis = s3.c_is * math.pow(2, -29)
    eph.i0 = s3.i_0 * math.pow(2, -31) * self.gpsPi
    eph.crc = s3.c_rc * math.pow(2, -5)
    eph.omega = s3.omega * math.pow(2, -31) * self.gpsPi
    eph.omegaDot = s3.omega_dot * math.pow(2, -43) * self.gpsPi
    eph.iode = s3.iode
    eph.iDot = s3.idot * math.pow(2, -43) * self.gpsPi
    iode_s3 = s3.iode

    eph.toeWeek = week
    eph.tocWeek = week

    # clear cache for this SV
    self.caches.gps_subframes[msg.sv_id].clear()
    if not (iodc_lsb == iode_s2 == iode_s3):
      return None
    return ('ubloxGnss', dat)

  def _parse_glonass_ephemeris(self, msg: Ubx.RxmSfrbx) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder] | None:
    # words are 4 bytes each; Glonass parser expects 16 bytes (string)
    body = msg.body
    if len(body) != 4:
      return None
    string_bytes = bytearray()
    for word in body:
      for i in (3, 2, 1, 0):
        string_bytes.append((word >> (8 * i)) & 0xFF)

    gl = Glonass.from_bytes(bytes(string_bytes))
    string_number = gl.string_number
    if string_number < 1 or string_number > 5 or gl.idle_chip:
      return None

    # correlate by superframe and timing, similar to C++ logic
    freq_id = msg.freq_id
    superframe_unknown = False
    needs_clear = False
    for i in range(1, 6):
      if i not in self.caches.glonass_strings[freq_id]:
        continue
      sf_prev = self.caches.glonass_string_superframes[freq_id].get(i, 0)
      if sf_prev == 0 or gl.superframe_number == 0:
        superframe_unknown = True
      elif sf_prev != gl.superframe_number:
        needs_clear = True
      if superframe_unknown:
        prev_time = self.caches.glonass_string_times[freq_id].get(i, 0.0)
        if abs((prev_time - 2.0 * i) - (self.framer.last_log_time - 2.0 * string_number)) > 10:
          needs_clear = True

    if needs_clear:
      self.caches.glonass_strings[freq_id].clear()
      self.caches.glonass_string_superframes[freq_id].clear()
      self.caches.glonass_string_times[freq_id].clear()

    self.caches.glonass_strings[freq_id][string_number] = bytes(string_bytes)
    self.caches.glonass_string_superframes[freq_id][string_number] = gl.superframe_number
    self.caches.glonass_string_times[freq_id][string_number] = self.framer.last_log_time

    if msg.sv_id == 255:
      # unknown SV id
      return None
    if len(self.caches.glonass_strings[freq_id]) != 5:
      return None

    dat = messaging.new_message('ubloxGnss', valid=True)
    eph = dat.ubloxGnss.init('glonassEphemeris')
    eph.svId = msg.sv_id
    eph.freqNum = msg.freq_id - 7

    current_day = 0
    tk = 0

    # string 1
    try:
      s1 = Glonass.from_bytes(self.caches.glonass_strings[freq_id][1]).data
    except Exception:
      return None
    assert isinstance(s1, Glonass.String1)
    eph.p1 = int(s1.p1)
    tk = int(s1.t_k)
    eph.tkDEPRECATED = tk
    eph.xVel = float(s1.x_vel) * math.pow(2, -20)
    eph.xAccel = float(s1.x_accel) * math.pow(2, -30)
    eph.x = float(s1.x) * math.pow(2, -11)

    # string 2
    try:
      s2 = Glonass.from_bytes(self.caches.glonass_strings[freq_id][2]).data
    except Exception:
      return None
    assert isinstance(s2, Glonass.String2)
    eph.svHealth = int(s2.b_n >> 2)
    eph.p2 = int(s2.p2)
    eph.tb = int(s2.t_b)
    eph.yVel = float(s2.y_vel) * math.pow(2, -20)
    eph.yAccel = float(s2.y_accel) * math.pow(2, -30)
    eph.y = float(s2.y) * math.pow(2, -11)

    # string 3
    try:
      s3 = Glonass.from_bytes(self.caches.glonass_strings[freq_id][3]).data
    except Exception:
      return None
    assert isinstance(s3, Glonass.String3)
    eph.p3 = int(s3.p3)
    eph.gammaN = float(s3.gamma_n) * math.pow(2, -40)
    eph.svHealth = int(eph.svHealth | (1 if s3.l_n else 0))
    eph.zVel = float(s3.z_vel) * math.pow(2, -20)
    eph.zAccel = float(s3.z_accel) * math.pow(2, -30)
    eph.z = float(s3.z) * math.pow(2, -11)

    # string 4
    try:
      s4 = Glonass.from_bytes(self.caches.glonass_strings[freq_id][4]).data
    except Exception:
      return None
    assert isinstance(s4, Glonass.String4)
    current_day = int(s4.n_t)
    eph.nt = current_day
    eph.tauN = float(s4.tau_n) * math.pow(2, -30)
    eph.deltaTauN = float(s4.delta_tau_n) * math.pow(2, -30)
    eph.age = int(s4.e_n)
    eph.p4 = int(s4.p4)
    eph.svURA = float(self.glonass_URA_lookup.get(int(s4.f_t), 0.0))
    # consistency check: SV slot number
    # if it doesn't match, keep going but note mismatch (no logging here)
    eph.svType = int(s4.m)

    # string 5
    try:
      s5 = Glonass.from_bytes(self.caches.glonass_strings[freq_id][5]).data
    except Exception:
      return None
    assert isinstance(s5, Glonass.String5)
    eph.n4 = int(s5.n_4)
    tk_seconds = int(SECS_IN_HR * ((tk >> 7) & 0x1F) + SECS_IN_MIN * ((tk >> 1) & 0x3F) + (tk & 0x1) * 30)
    eph.tkSeconds = tk_seconds

    self.caches.glonass_strings[freq_id].clear()
    return ('ubloxGnss', dat)

  def _gen_rxm_rawx(self, msg: Ubx.RxmRawx) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder]:
    dat = messaging.new_message('ubloxGnss', valid=True)
    mr = dat.ubloxGnss.init('measurementReport')
    mr.rcvTow = msg.rcv_tow
    mr.gpsWeek = msg.week
    mr.leapSeconds = msg.leap_s

    mb = mr.init('measurements', msg.num_meas)
    for i, m in enumerate(msg.meas):
      mb[i].svId = m.sv_id
      mb[i].pseudorange = m.pr_mes
      mb[i].carrierCycles = m.cp_mes
      mb[i].doppler = m.do_mes
      mb[i].gnssId = int(m.gnss_id.value)
      mb[i].glonassFrequencyIndex = m.freq_id
      mb[i].locktime = m.lock_time
      mb[i].cno = m.cno
      mb[i].pseudorangeStdev = 0.01 * (math.pow(2, (m.pr_stdev & 15)))
      mb[i].carrierPhaseStdev = 0.004 * (m.cp_stdev & 15)
      mb[i].dopplerStdev = 0.002 * (math.pow(2, (m.do_stdev & 15)))

      ts = mb[i].init('trackingStatus')
      trk = m.trk_stat
      ts.pseudorangeValid = _bit(trk, 0)
      ts.carrierPhaseValid = _bit(trk, 1)
      ts.halfCycleValid = _bit(trk, 2)
      ts.halfCycleSubtracted = _bit(trk, 3)

    mr.numMeas = msg.num_meas
    rs = mr.init('receiverStatus')
    rs.leapSecValid = _bit(msg.rec_stat, 0)
    rs.clkReset = _bit(msg.rec_stat, 2)
    return ('ubloxGnss', dat)

  def _gen_nav_sat(self, msg: Ubx.NavSat) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder]:
    dat = messaging.new_message('ubloxGnss', valid=True)
    sr = dat.ubloxGnss.init('satReport')
    sr.iTow = msg.itow
    svs = sr.init('svs', msg.num_svs)
    for i, s in enumerate(msg.svs):
      svs[i].svId = s.sv_id
      svs[i].gnssId = int(s.gnss_id.value)
      svs[i].flagsBitfield = s.flags
      svs[i].cno = s.cno
      svs[i].elevationDeg = s.elev
      svs[i].azimuthDeg = s.azim
      svs[i].pseudorangeResidual = s.pr_res * 0.1
    return ('ubloxGnss', dat)

  def _gen_mon_hw(self, msg: Ubx.MonHw) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder]:
    dat = messaging.new_message('ubloxGnss', valid=True)
    hw = dat.ubloxGnss.init('hwStatus')
    hw.noisePerMS = msg.noise_per_ms
    hw.flags = msg.flags
    hw.agcCnt = msg.agc_cnt
    hw.aStatus = int(msg.a_status.value)
    hw.aPower = int(msg.a_power.value)
    hw.jamInd = msg.jam_ind
    return ('ubloxGnss', dat)

  def _gen_mon_hw2(self, msg: Ubx.MonHw2) -> tuple[str, capnp.lib.capnp._DynamicStructBuilder]:
    dat = messaging.new_message('ubloxGnss', valid=True)
    hw = dat.ubloxGnss.init('hwStatus2')
    hw.ofsI = msg.ofs_i
    hw.magI = msg.mag_i
    hw.ofsQ = msg.ofs_q
    hw.magQ = msg.mag_q
    # Map Ubx enum to cereal enum {undefined=0, rom=1, otp=2, configpins=3, flash=4}
    cfg_map = {
      Ubx.MonHw2.ConfigSource.rom: 1,
      Ubx.MonHw2.ConfigSource.otp: 2,
      Ubx.MonHw2.ConfigSource.config_pins: 3,
      Ubx.MonHw2.ConfigSource.flash: 4,
    }
    hw.cfgSource = cfg_map.get(msg.cfg_source, 0)
    hw.lowLevCfg = msg.low_lev_cfg
    hw.postStatus = msg.post_status
    return ('ubloxGnss', dat)


def main():
  parser = UbloxMsgParser()
  pm = messaging.PubMaster(['ubloxGnss', 'gpsLocationExternal'])
  sock = messaging.sub_sock('ubloxRaw', timeout=100, conflate=False)

  while True:
    msg = messaging.recv_one_or_none(sock)
    if msg is None:
      continue

    data = bytes(msg.ubloxRaw)
    log_time = msg.logMonoTime * 1e-9
    frames = parser.framer.add_data(log_time, data)
    for frame in frames:
      try:
        res = parser.parse_frame(frame)
      except Exception:
        continue
      if not res:
        continue
      service, dat = res
      pm.send(service, dat)

if __name__ == '__main__':
  main()
