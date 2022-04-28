#!/usr/bin/env python3
from typing import List

from cereal import messaging, log
from laika import AstroDog, helpers
from laika.raw_gnss import GNSSMeasurement, calc_pos_fix, correct_measurements, process_measurements, read_raw_ublox


def process_report(ublox_gnss, dog: AstroDog):
  measurements = None
  report = ublox_gnss.measurementReport
  if len(report.measurements) > 0:
    new_meas = read_raw_ublox(report)
    measurements = [m for m in new_meas if helpers.get_constellation(m.prn) == 'GPS']
    measurements = process_measurements(measurements, dog)
  return measurements


def correct_and_pos_fix(processed_measurements: List[GNSSMeasurement], dog: AstroDog):
  # pos fix needs more than 5 processed_measurements
  pos_fix = calc_pos_fix(processed_measurements)

  est_pos = pos_fix[0][:3]
  corrected = correct_measurements(processed_measurements, est_pos, dog)
  if len(corrected) < 6:
    return None
  return calc_pos_fix(corrected), corrected


def process_ublox_msg(ublox_msg, dog: AstroDog, pm: messaging.PubMaster):
  if ublox_msg.which == 'measurementReport':
    processed_measurements = process_report(ublox_msg, dog)
    if processed_measurements is None or len(processed_measurements) < 6:
      return None
    processed = correct_and_pos_fix(processed_measurements, dog)
    if processed is None:
      return None
    (corrected_pos, _), corrected_measurements = processed
    dat = messaging.new_message('gnssMeasurements')
    correct_meas_msgs = [create_measurement_msg(m) for m in corrected_measurements]

    dat.gnssMeasurements = {
      "latitude": corrected_pos[0],
      "longitude": corrected_pos[1],
      "altitude": corrected_pos[2],
      "correctedMeasurements": correct_meas_msgs
    }

    pm.send('gnssMeasurements', dat)


def create_measurement_msg(meas: GNSSMeasurement):
  m = meas.as_array()
  # state = log.ManagerState.ProcessState.new_message()
  c = log.GnssMeasurements.CorrectedMeasurement.new_message()
  c.CorrectedMeasurement = {
    "nmea_id": m[0],
    "gpsWeek": m[1],
    "gpsTimeOfWeek": m[2],
    "glonassFreq": m[3],
    "c1c": m[4],
    "c1c_std": m[5],
    "d1c": m[6],
    "d1c_std": m[7],
    "sat_pos": m[8],
    "sat_vel": m[9],
  }
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
      process_ublox_msg(ublox_msg, dog, pm)


if __name__ == "__main__":
  main()
