from enum import IntEnum
from typing import Dict

import numpy as np
from .lib.coordinates import LocalCoord


class ConstellationId(IntEnum):
  # Int values match Ublox gnssid version 8
  GPS = 0
  SBAS = 1
  GALILEO = 2
  BEIDOU = 3
  IMES = 4
  QZNSS = 5
  GLONASS = 6
  # Not supported by Ublox:
  IRNSS = 7

  def to_rinex_char(self) -> str:
    # returns single character id
    return RINEX_CONSTELLATION_TO_ID[self]

  @classmethod
  def from_rinex_char(cls, c: str):
    if c in RINEX_ID_TO_CONSTELLATION:
      return RINEX_ID_TO_CONSTELLATION[c]
    else:
      raise ValueError("Unknown rinex constellation id: ", c)

  @classmethod
  def from_qcom_source(cls, report_source: int):
    if report_source == 0:
      return ConstellationId.GPS
    if report_source == 1:
      return ConstellationId.GLONASS
    if report_source == 2:
      return ConstellationId.BEIDOU
    if report_source == 6:
      return ConstellationId.SBAS
    raise NotImplementedError('Only GPS (0), GLONASS (1), BEIDOU (2) and SBAS (6) are supported from qcom, not:', {report_source})


# From https://gpsd.gitlab.io/gpsd/NMEA.html#_satellite_ids
# NmeaId is the unique 3 digits id for every satellite globally. (Example: 001, 201)
# SvId is the 2 digits satellite id that is unique within a constellation. (Get the unique satellite with the constellation id. Examples: G01, R01)
CONSTELLATION_TO_NMEA_RANGES = {
  # NmeaId ranges for each constellation with its svId offset.
  # constellation: [(start, end, svIdOffset)]
  # svId = nmeaId + offset
  ConstellationId.GPS: [(1, 32, 0)],  # svId [1,32]
  ConstellationId.SBAS: [(33, 64, -32), (120, 158, -87)],  # svId [1,71]
  ConstellationId.GLONASS: [(65, 96, -64)],  # svId [1,31]
  ConstellationId.IMES: [(173, 182, -172)],  # svId [1,9]
  ConstellationId.QZNSS: [(193, 200, -192)],  # svId [1,28]  # todo should be QZSS
  ConstellationId.BEIDOU: [(201, 235, -200), (401, 437, -365)],  # svId 1-72
  ConstellationId.GALILEO: [(301, 336, -300)]  # svId 1-36
}
#
# # Source: RINEX 3.04
RINEX_CONSTELLATION_TO_ID: Dict[ConstellationId, str] = {
  ConstellationId.GPS: 'G',
  ConstellationId.GLONASS: 'R',
  ConstellationId.SBAS: 'S',
  ConstellationId.GALILEO: 'E',
  ConstellationId.BEIDOU: 'C',
  ConstellationId.QZNSS: 'J',
  ConstellationId.IRNSS: 'I'
}

# Make above dictionary bidirectional map:
# Now you can ask for constellation using:
# >>> RINEX_CONSTELLATION_IDENTIFIERS['R']
#     "GLONASS"
RINEX_ID_TO_CONSTELLATION: Dict[str, ConstellationId] = {con_id: con for con, con_id in RINEX_CONSTELLATION_TO_ID.items()}


def get_el_az(pos, sat_pos):
  converter = LocalCoord.from_ecef(pos)
  sat_ned = converter.ecef2ned(sat_pos)
  sat_range = np.linalg.norm(sat_ned)

  el = np.arcsin(-sat_ned[2] / sat_range)  # pylint: disable=unsubscriptable-object
  az = np.arctan2(sat_ned[1], sat_ned[0])  # pylint: disable=unsubscriptable-object
  return el, az


def get_closest(time, candidates, recv_pos=None):
  if recv_pos is None:
    # Takes a list of object that have an epoch(GPSTime) value
    # and return the one that is closest the given time (GPSTime)
    return min(candidates, key=lambda candidate: abs(time - candidate.epoch), default=None)

  return min(
    (candidate for candidate in candidates if candidate.valid(time, recv_pos)),
    key=lambda candidate: np.linalg.norm(recv_pos - candidate.pos),
    default=None,
  )

def get_constellation(prn: str):
  identifier = prn[0]
  return ConstellationId.from_rinex_char(identifier)

def get_sv_id(prn: str):
  return int(prn[1:])

def get_constellation_and_sv_id(nmea_id):
  for c, ranges in CONSTELLATION_TO_NMEA_RANGES.items():
    for (start, end, sv_id_offset) in ranges:
      if start <= nmea_id <= end:
        sv_id = nmea_id + sv_id_offset
        return c, sv_id

  raise ValueError(f"constellation not found for nmeaid {nmea_id}")


def get_prn_from_nmea_id(nmea_id: int):
  c_id, sv_id = get_constellation_and_sv_id(nmea_id)
  return "%s%02d" % (c_id.to_rinex_char(), sv_id)


def get_nmea_id_from_prn(prn: str):
  constellation = get_constellation(prn)
  sv_id = int(prn[1:])  # satellite id
  return get_nmea_id_from_constellation_and_svid(constellation, sv_id)


def get_nmea_id_from_constellation_and_svid(constellation: ConstellationId, sv_id: int):
  ranges = CONSTELLATION_TO_NMEA_RANGES[constellation]
  for (start, end, sv_id_offset) in ranges:
    new_nmea_id = sv_id - sv_id_offset
    if start <= new_nmea_id <= end:
      return new_nmea_id

  raise ValueError(f"NMEA ID not found for constellation {constellation.name} with satellite id {sv_id}")


def rinex3_obs_from_rinex2_obs(observable):
  if observable == 'P2':
    return 'C2P'
  if len(observable) == 2:
    return observable + 'C'
  raise NotImplementedError("Don't know this: " + observable)


class TimeRangeHolder:
  '''Class to support test if date is in any of the multiple, sparse ranges'''

  def __init__(self):
    # Sorted list
    self._ranges = []

  def _previous_and_contains_index(self, time):
    prev = None
    current = None

    for idx, (start, end) in enumerate(self._ranges):
      # Time may be in next range
      if time > end:
        continue

      # Time isn't in any next range
      if time < start:
        prev = idx - 1
        current = None
      # Time is in current range
      else:
        prev = idx - 1
        current = idx
      break

    # Break in last loop
    if prev is None:
      prev = len(self._ranges) - 1

    return prev, current

  def add(self, start_time, end_time):
    prev_start, current_start = self._previous_and_contains_index(start_time)
    _, current_end = self._previous_and_contains_index(end_time)

    # Merge ranges
    if current_start is not None and current_end is not None:
      # If ranges are different then merge
      if current_start != current_end:
        new_start, _ = self._ranges[current_start]
        _, new_end = self._ranges[current_end]
        new_range = (new_start, new_end)
        # Required reversed order to correct remove
        del self._ranges[current_end]
        del self._ranges[current_start]
        self._ranges.insert(current_start, new_range)
    # Extend range - left
    elif current_start is not None:
      new_start, _ = self._ranges[current_start]
      new_range = (new_start, end_time)
      del self._ranges[current_start]
      self._ranges.insert(current_start, new_range)
    # Extend range - right
    elif current_end is not None:
      _, new_end = self._ranges[current_end]
      new_range = (start_time, new_end)
      del self._ranges[current_end]
      self._ranges.insert(prev_start + 1, new_range)
    # Create new range
    else:
      new_range = (start_time, end_time)
      self._ranges.insert(prev_start + 1, new_range)

  def __contains__(self, time):
    for start, end in self._ranges:
      # Time may be in next range
      if time > end:
        continue

      # Time isn't in any next range
      if time < start:
        return False
      # Time is in current range
      return True
    return False
