from math import sqrt
from typing import Dict, List, Optional, Union

import numpy as np
import datetime
import struct

from . import constants
from .ephemeris import Ephemeris
from .lib.coordinates import LocalCoord
from .gps_time import GPSTime
from .helpers import ConstellationId, get_constellation_and_sv_id, get_nmea_id_from_constellation_and_svid, \
  rinex3_obs_from_rinex2_obs


def array_from_normal_meas(meas):
  return np.concatenate(([meas.get_nmea_id()],
                         [meas.recv_time_week],
                         [meas.recv_time_sec],
                         [meas.glonass_freq],
                         [meas.observables['C1C']],
                         [meas.observables_std['C1C']],
                         [meas.observables['D1C']],
                         [meas.observables_std['D1C']],
                         [meas.observables['S1C']],
                         [meas.observables['L1C']]))


def normal_meas_from_array(arr):
  observables, observables_std = {}, {}
  observables['C1C'] = arr[4]
  observables_std['C1C'] = arr[5]
  observables['D1C'] = arr[6]
  observables_std['D1C'] = arr[7]
  observables['S1C'] = arr[8]
  observables['L1C'] = arr[9]
  constellation_id, sv_id = get_constellation_and_sv_id(nmea_id=arr[0])
  return GNSSMeasurement(constellation_id, sv_id, arr[1], arr[2],
                         observables, observables_std, arr[3])


class GNSSMeasurement:
  PRN = 0
  RECV_TIME_WEEK = 1
  RECV_TIME_SEC = 2
  GLONASS_FREQ = 3

  PR = 4
  PR_STD = 5
  PRR = 6
  PRR_STD = 7

  SAT_POS = slice(8, 11)
  SAT_VEL = slice(11, 14)

  def __init__(self, constellation_id: ConstellationId, sv_id: int, recv_time_week: int, recv_time_sec: float, observables: Dict[str, float], observables_std: Dict[str, float],
               glonass_freq: Union[int, float] = None):
    # Metadata
    # prn: unique satellite id
    self.prn = "%s%02d" % (constellation_id.to_rinex_char(), sv_id)  # satellite ID in rinex convention
    self.constellation_id = constellation_id
    self.sv_id = sv_id  # satellite id per constellation

    self.recv_time_week = recv_time_week
    self.recv_time_sec = recv_time_sec
    self.recv_time = GPSTime(recv_time_week, recv_time_sec)
    self.glonass_freq = glonass_freq  # glonass channel

    # Measurements
    self.observables = observables
    self.observables_std = observables_std

    # flags
    self.processed = False
    self.corrected = False

    # sat info
    self.sat_pos = np.array([np.nan, np.nan, np.nan])
    self.sat_vel = np.array([np.nan, np.nan, np.nan])
    self.sat_clock_err = np.nan
    self.sat_ephemeris: Optional[Ephemeris] = None

    self.sat_pos_final = np.array([np.nan, np.nan, np.nan])  # sat_pos in receiver time's ECEF frame instead of satellite time's ECEF frame
    self.observables_final: Dict[str, float] = {}

  def process(self, dog):
    sat_time = self.recv_time - self.observables['C1C']/constants.SPEED_OF_LIGHT
    sat_info = dog.get_sat_info(self.prn, sat_time)
    if sat_info is None:
      return False
    self.sat_pos, self.sat_vel, self.sat_clock_err, _, self.sat_ephemeris = sat_info
    self.processed = True
    return True

  def correct(self, est_pos, dog):
    for obs in self.observables:
      if obs[0] == 'C':  # or obs[0] == 'L':
        delay = dog.get_delay(self.prn, self.recv_time, est_pos, signal=obs)
        if delay is not None:
          self.observables_final[obs] = (self.observables[obs] +
                                         self.sat_clock_err*constants.SPEED_OF_LIGHT -
                                         delay)
      else:
        self.observables_final[obs] = self.observables[obs]
    if 'C1C' in self.observables_final and 'C2P' in self.observables_final:
      self.observables_final['IOF'] = (((constants.GPS_L1**2)*self.observables_final['C1C'] -
                                        (constants.GPS_L2**2)*self.observables_final['C2P'])/
                                       (constants.GPS_L1**2 - constants.GPS_L2**2))

    geometric_range = np.linalg.norm(self.sat_pos - est_pos)
    theta_1 = constants.EARTH_ROTATION_RATE * geometric_range / constants.SPEED_OF_LIGHT
    self.sat_pos_final = np.array([self.sat_pos[0] * np.cos(theta_1) + self.sat_pos[1] * np.sin(theta_1),
                                   self.sat_pos[1] * np.cos(theta_1) - self.sat_pos[0] * np.sin(theta_1),
                                   self.sat_pos[2]])
    if 'C1C' in self.observables_final and np.isfinite(self.observables_final['C1C']):
      self.corrected = True
      return True
    return False

  def as_array(self, only_corrected=True):
    observables = self.observables_final
    sat_pos = self.sat_pos_final
    if not self.corrected:
      if only_corrected:
        raise NotImplementedError('Only corrected measurements can be put into arrays')
      else:
        observables = self.observables
        sat_pos = self.sat_pos
    ret = np.array([self.get_nmea_id(), self.recv_time_week, self.recv_time_sec, self.glonass_freq,
                    observables['C1C'], self.observables_std['C1C'],
                    observables['D1C'], self.observables_std['D1C']])
    return np.concatenate((ret, sat_pos, self.sat_vel))

  def __repr__(self):
    time = self.recv_time.as_datetime().strftime('%Y-%m-%dT%H:%M:%S.%f')
    return f"<GNSSMeasurement from {self.prn} at {time}>"

  def get_nmea_id(self):
    return get_nmea_id_from_constellation_and_svid(self.constellation_id, self.sv_id)


def process_measurements(measurements: List[GNSSMeasurement], dog) -> List[GNSSMeasurement]:
  proc_measurements = []
  for meas in measurements:
    if meas.process(dog):
      proc_measurements.append(meas)
  return proc_measurements


def correct_measurements(measurements: List[GNSSMeasurement], est_pos, dog) -> List[GNSSMeasurement]:
  corrected_measurements = []
  for meas in measurements:
    if meas.correct(est_pos, dog):
      corrected_measurements.append(meas)
  return corrected_measurements


def group_measurements_by_epoch(measurements):
  meas_filt_by_t = [[measurements[0]]]
  for m in measurements[1:]:
    if abs(m.recv_time - meas_filt_by_t[-1][-1].recv_time) > 1e-9:
      meas_filt_by_t.append([])
    meas_filt_by_t[-1].append(m)
  return meas_filt_by_t


def group_measurements_by_sat(measurements):
  measurements_by_sat = {}
  sats = {m.prn for m in measurements}
  for sat in sats:
    measurements_by_sat[sat] = [m for m in measurements if m.prn == sat]
  return measurements_by_sat


def read_raw_qcom(report):
  dr = 'DrMeasurementReport' in str(report.schema)
  # Only gps/sbas and glonass are supported
  constellation_id = ConstellationId.from_qcom_source(report.source)
  if constellation_id in [ConstellationId.GPS, ConstellationId.SBAS]:  # gps/sbas
    if dr:
      recv_tow = report.gpsMilliseconds / 1000.0  # seconds
      time_bias_ms = struct.unpack("f", struct.pack("I", report.gpsTimeBiasMs))[0]
    else:
      recv_tow = report.milliseconds / 1000.0  # seconds
      time_bias_ms = report.timeBias
    recv_time = GPSTime(report.gpsWeek, recv_tow)
  elif constellation_id == ConstellationId.GLONASS:
    if dr:
      recv_tow = report.glonassMilliseconds / 1000.0  # seconds
      recv_time = GPSTime.from_glonass(report.glonassYear, report.glonassDay, recv_tow)
      time_bias_ms = report.glonassTimeBias
    else:
      recv_tow = report.milliseconds / 1000.0  # seconds
      recv_time = GPSTime.from_glonass(report.glonassCycleNumber, report.glonassNumberOfDays, recv_tow)
      time_bias_ms = report.timeBias
  else:
    raise NotImplementedError('Only GPS (0), SBAS (1) and GLONASS (6) are supported from qcom, not:', {report.source})
  # logging.debug(recv_time, report.source, time_bias_ms, dr)
  measurements = []
  for i in report.sv:
    nmea_id = i.svId  # todo change svId to nmea_id in cereal message. Or better: change the publisher to publish correct svId's, since constellation id is also given
    if nmea_id == 255:
      # TODO nmea_id is not valid. Fix publisher
      continue
    _, sv_id = get_constellation_and_sv_id(nmea_id)
    if not i.measurementStatus.measurementNotUsable and i.measurementStatus.satelliteTimeIsKnown:
      sat_tow = (i.unfilteredMeasurementIntegral + i.unfilteredMeasurementFraction + i.latency + time_bias_ms) / 1000
      observables, observables_std = {}, {}
      observables['C1C'] = (recv_tow - sat_tow)*constants.SPEED_OF_LIGHT
      observables_std['C1C'] = i.unfilteredTimeUncertainty * 1e-3 * constants.SPEED_OF_LIGHT
      if i.measurementStatus.fineOrCoarseVelocity:
        # about 10x better, perhaps filtered with carrier phase?
        observables['D1C'] = i.fineSpeed
        observables_std['D1C'] = i.fineSpeedUncertainty
      else:
        observables['D1C'] = i.unfilteredSpeed
        observables_std['D1C'] = i.unfilteredSpeedUncertainty
      observables['S1C'] = (i.carrierNoise/100.) if i.carrierNoise != 0 else np.nan
      observables['L1C'] = np.nan
      # logging.debug("  %.5f %3d %10.2f %7.2f %7.2f %.2f %d" % (recv_time.tow, nmea_id,
      # observables['C1C'], observables_std['C1C'],
      # observables_std['D1C'], observables['S1C'], i.latency), i.observationState, i.measurementStatus.fineOrCoarseVelocity)
      glonass_freq = (i.glonassFrequencyIndex - 7) if constellation_id == ConstellationId.GLONASS else np.nan
      measurements.append(GNSSMeasurement(constellation_id, sv_id,
                                          recv_time.week,
                                          recv_time.tow,
                                          observables,
                                          observables_std,
                                          glonass_freq))
  return measurements


def read_raw_ublox(report) -> List[GNSSMeasurement]:
  recv_tow = report.rcvTow  # seconds
  recv_week = report.gpsWeek
  measurements = []
  for i in report.measurements:
    # only add Gps and Glonass fixes
    if i.gnssId in [ConstellationId.GPS, ConstellationId.GLONASS]:
      if i.svId > 32 or i.pseudorange > 2**32:
        continue
      observables = {}
      observables_std = {}
      if i.trackingStatus.pseudorangeValid and i.sigId == 0:
        observables['C1C'] = i.pseudorange
        # Empirically it seems obvious ublox's std is
        # actually a variation
        observables_std['C1C'] = sqrt(i.pseudorangeStdev)*10
        if i.gnssId == ConstellationId.GLONASS:
          glonass_freq = i.glonassFrequencyIndex - 7
          observables['D1C'] = -(constants.SPEED_OF_LIGHT / (constants.GLONASS_L1 + glonass_freq * constants.GLONASS_L1_DELTA)) * i.doppler
        else:  # GPS
          glonass_freq = np.nan
          observables['D1C'] = -(constants.SPEED_OF_LIGHT / constants.GPS_L1) * i.doppler
        observables_std['D1C'] = (constants.SPEED_OF_LIGHT / constants.GPS_L1) * i.dopplerStdev
        observables['S1C'] = i.cno
        if i.trackingStatus.carrierPhaseValid:
          observables['L1C'] = i.carrierCycles
        else:
          observables['L1C'] = np.nan

        measurements.append(GNSSMeasurement(ConstellationId(i.gnssId), i.svId, recv_week, recv_tow,
                                            observables, observables_std, glonass_freq))
  return measurements


def read_rinex_obs(obsdata) -> List[List[GNSSMeasurement]]:
  measurements: List[List[GNSSMeasurement]] = []
  obsdata_keys = list(obsdata.data.keys())
  first_sat = obsdata_keys[0]
  n = len(obsdata.data[first_sat]['Epochs'])
  for i in range(n):
    recv_time_datetime = obsdata.data[first_sat]['Epochs'][i]
    recv_time_datetime = recv_time_datetime.astype(datetime.datetime)
    recv_time = GPSTime.from_datetime(recv_time_datetime)
    measurements.append([])
    for sat_str in obsdata_keys:
      if np.isnan(obsdata.data[sat_str]['C1'][i]):
        continue
      observables, observables_std = {}, {}
      for obs in obsdata.data[sat_str]:
        if obs == 'Epochs':
          continue
        rinex3_obs_key = rinex3_obs_from_rinex2_obs(obs)
        observables[rinex3_obs_key] = obsdata.data[sat_str][obs][i]
        observables_std[rinex3_obs_key] = 1.

      constellation_id, sv_id = get_constellation_and_sv_id(int(sat_str))
      measurements[-1].append(GNSSMeasurement(constellation_id, sv_id,
                                              recv_time.week, recv_time.tow,
                                              observables, observables_std))
  return measurements


def get_Q(recv_pos, sat_positions):
  local = LocalCoord.from_ecef(recv_pos)
  sat_positions_rel = local.ecef2ned(sat_positions)
  sat_distances = np.linalg.norm(sat_positions_rel, axis=1)
  A = np.column_stack((sat_positions_rel[:,0]/sat_distances,  # pylint: disable=unsubscriptable-object
                       sat_positions_rel[:,1]/sat_distances,  # pylint: disable=unsubscriptable-object
                       sat_positions_rel[:,2]/sat_distances,  # pylint: disable=unsubscriptable-object
                       -np.ones(len(sat_distances))))
  if A.shape[0] < 4 or np.linalg.matrix_rank(A) < 4:
    return np.inf*np.ones((4,4))
  Q = np.linalg.inv(A.T.dot(A))
  return Q


def get_DOP(recv_pos, sat_positions):
  Q = get_Q(recv_pos, sat_positions)
  return np.sqrt(np.trace(Q))


def get_HDOP(recv_pos, sat_positions):
  Q = get_Q(recv_pos, sat_positions)
  return np.sqrt(np.trace(Q[:2,:2]))


def get_VDOP(recv_pos, sat_positions):
  Q = get_Q(recv_pos, sat_positions)
  return np.sqrt(Q[2,2])


def get_TDOP(recv_pos, sat_positions):
  Q = get_Q(recv_pos, sat_positions)
  return np.sqrt(Q[3,3])


def get_PDOP(recv_pos, sat_positions):
  Q = get_Q(recv_pos, sat_positions)
  return np.sqrt(np.trace(Q[:3,:3]))
