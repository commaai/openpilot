#!/usr/bin/env python3
from typing import List

from cereal import log, messaging
from laika import AstroDog
from laika.helpers import CONSTELLATION_ID_TO_GNSS_ID, get_nmea_id_from_prn
from laika.raw_gnss import GNSSMeasurement, calc_pos_fix, correct_measurements, process_measurements, read_raw_ublox


def process_report(ublox_gnss, dog: AstroDog):
  processed_measurements = None
  report = ublox_gnss.measurementReport
  if len(report.measurements) > 0:
    new_meas = read_raw_ublox(report)
    processed_measurements = process_measurements(new_meas, dog)
  return processed_measurements


def correct_and_pos_fix(processed_measurements: List[GNSSMeasurement], dog: AstroDog):
  # pos fix needs more than 5 processed_measurements
  # todo solve the calc_pos_fix when not enough measurements for this timestamp
  #   could to keep in mind older measurements within time range to ensure
  #   Or use less satellites to create a fix.
  pos_fix = calc_pos_fix(processed_measurements)

  if len(pos_fix) == 0:
    return [], []
  est_pos = pos_fix[0][:3]
  corrected = correct_measurements(processed_measurements, est_pos, dog)
  return calc_pos_fix(corrected), corrected


def process_ublox_msg(ublox_msg, dog: AstroDog, pm: messaging.PubMaster, ublox_mono_time: int):
  if ublox_msg.which == 'measurementReport':
    processed_measurements = process_report(ublox_msg, dog)
    if processed_measurements is None:
      return False

    corrected = correct_and_pos_fix(processed_measurements, dog)
    pos_fix, corrected_measurements = corrected  # pylint: disable=unused-variable
    dat = messaging.new_message('gnssMeasurements')
    # todo send corrected messages instead of processed_measurements. Need fix for less than 6 measurements
    correct_meas_msgs = [create_measurement_msg(m) for m in processed_measurements]
    # pos fix can be an empty list if not enough correct measurements are available
    if len(pos_fix) > 0:
      corrected_pos = pos_fix[0].tolist()
    else:
      corrected_pos = [0., 0., 0.]
    dat.gnssMeasurements = {
      "position": corrected_pos,
      "ubloxMonoTime": ublox_mono_time,
      "correctedMeasurements": correct_meas_msgs
    }

    pm.send('gnssMeasurements', dat)
    return True

def create_measurement_msg(meas: GNSSMeasurement):
  c = log.GnssMeasurements.CorrectedMeasurement.new_message()
  c.nmeaId = get_nmea_id_from_prn(meas.prn)
  # c.glonassFrequency = 0 if math.isnan(meas.glonass_freq) else meas.glonass_freq # todo fix
  c.gnssId = CONSTELLATION_ID_TO_GNSS_ID[meas.prn[0]]
  c.pseudorange = float(meas.observables['C1C'])  # todo should be observables_final when using corrected measurements
  c.pseudorangeStd = float(meas.observables_std['C1C'])
  c.pseudorangeRate = float(meas.observables['D1C'])  # todo should be observables_final when using corrected measurements
  c.pseudorangeRateStd = float(meas.observables_std['D1C'])
  c.satPos = meas.sat_pos_final.tolist()
  c.satVel = meas.sat_vel.tolist()
  return c


def main():
  dog = AstroDog()
  sm = messaging.SubMaster(['ubloxGnss'])
  pm = messaging.PubMaster(['gnssMeasurements'])

  while True:
    sm.update()

    # Todo if no internet available use latest ephemeris
    if sm.updated['ubloxGnss']:
      ublox_msg = sm['ubloxGnss']
      process_ublox_msg(ublox_msg, dog, pm, sm.logMonoTime['ubloxGnss'])


if __name__ == "__main__":
  main()
