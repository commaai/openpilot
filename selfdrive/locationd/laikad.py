#!/usr/bin/env python3
from typing import List

from cereal import log, messaging
from laika import AstroDog
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, calc_pos_fix, calc_vel_fix, correct_measurements, process_measurements, read_raw_ublox


def correct_and_vel_pos_fix(processed_measurements: List[GNSSMeasurement], dog: AstroDog):
  # pos fix needs more than 5 processed_measurements
  pos_fix = calc_pos_fix(processed_measurements)

  if len(pos_fix) == 0:
    return [], [], []
  est_pos = pos_fix[0][:3]
  corrected = correct_measurements(processed_measurements, est_pos, dog)
  corrected_pos = calc_pos_fix(corrected)
  corrected_vel = calc_vel_fix(corrected, corrected_pos[0])
  return corrected_pos, corrected_vel, corrected


def process_ublox_msg(ublox_msg, dog, ublox_mono_time: int):
  if ublox_msg.which == 'measurementReport':
    report = ublox_msg.measurementReport
    if len(report.measurements) == 0:
      return None
    new_meas = read_raw_ublox(report)
    processed_measurements = process_measurements(new_meas, dog)

    pos_fix, vel_fix, corrected_measurements = correct_and_vel_pos_fix(processed_measurements, dog)
    # pos or vel fixes can be an empty list if not enough correct measurements are available
    correct_meas_msgs = [create_measurement_msg(m) for m in corrected_measurements]

    if len(pos_fix) > 0:
      corrected_pos = pos_fix[0][:3].tolist()
      corrected_vel = vel_fix[0][:3].tolist()
    else:
      corrected_pos = corrected_vel = [0., 0., 0.]

    dat = messaging.new_message('gnssMeasurements')
    dat.gnssMeasurements = {
      "positionECEF": corrected_pos,
      "velocityECEF": corrected_vel,
      "ubloxMonoTime": ublox_mono_time,
      "correctedMeasurements": correct_meas_msgs
    }
    return dat


def create_measurement_msg(meas: GNSSMeasurement):
  c = log.GnssMeasurements.CorrectedMeasurement.new_message()
  c.constellationId = meas.constellation_id.value
  c.svId = int(meas.prn[1:])
  c.glonassFrequency = meas.glonass_freq if meas.constellation_id == ConstellationId.GLONASS else 0
  c.pseudorange = float(meas.observables_final['C1C'])
  c.pseudorangeStd = float(meas.observables_std['C1C'])
  c.pseudorangeRate = float(meas.observables_final['D1C'])
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
      msg = process_ublox_msg(ublox_msg, dog, sm.logMonoTime['ubloxGnss'])
      if msg is None:
        msg = messaging.new_message('gnssMeasurements')
      pm.send('gnssMeasurements', msg)


if __name__ == "__main__":
  main()
