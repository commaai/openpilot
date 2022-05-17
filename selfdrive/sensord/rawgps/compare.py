#!/usr/bin/env python3
import cereal.messaging as messaging
from laika import constants

if __name__ == "__main__":
  sm = messaging.SubMaster(['ubloxGnss', 'qcomGnss'])

  meas = None
  while 1:
    sm.update()
    if sm['ubloxGnss'].which() == "measurementReport":
      meas = sm['ubloxGnss'].measurementReport.measurements
    if not sm.updated['qcomGnss'] or meas is None:
      continue
    report = sm['qcomGnss'].measurementReport
    if report.source not in [0, 1]:
      continue
    GLONASS = report.source == 1
    recv_time = report.milliseconds / 1000

    car = []
    print("qcom has ", list(sorted([x.svId for x in report.sv])))
    print("ublox has", list(sorted([x.svId for x in meas if x.gnssId == (6 if GLONASS else 0)])))
    for i in report.sv:
      # match to ublox
      tm = None
      for m in meas:
        if i.svId == m.svId and m.gnssId == 0 and m.sigId == 0 and not GLONASS:
          tm = m
        if (i.svId-64) == m.svId and m.gnssId == 6 and m.sigId == 0 and GLONASS:
          tm = m
      if tm is None:
        continue

      if not i.measurementStatus.measurementNotUsable and i.measurementStatus.satelliteTimeIsKnown:
        sat_time = (i.unfilteredMeasurementIntegral + i.unfilteredMeasurementFraction + i.latency) / 1000
        ublox_psuedorange = tm.pseudorange
        qcom_psuedorange = (recv_time - sat_time)*constants.SPEED_OF_LIGHT
        if GLONASS:
          glonass_freq = tm.glonassFrequencyIndex - 7
          ublox_speed = -(constants.SPEED_OF_LIGHT / (constants.GLONASS_L1 + glonass_freq*constants.GLONASS_L1_DELTA)) * (tm.doppler)
        else:
          ublox_speed = -(constants.SPEED_OF_LIGHT / constants.GPS_L1) * tm.doppler
        qcom_speed = i.unfilteredSpeed
        car.append((i.svId, tm.pseudorange, ublox_speed, qcom_psuedorange, qcom_speed, tm.cno))

    if len(car) == 0:
      print("nothing to compare")
      continue

    pr_err, speed_err = 0., 0.
    for c in car:
      ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed = c[1:5]
      pr_err += ublox_psuedorange - qcom_psuedorange
      speed_err += ublox_speed - qcom_speed
    pr_err /= len(car)
    speed_err /= len(car)
    print("avg psuedorange err %f avg speed err %f" % (pr_err, speed_err))
    for c in sorted(car, key=lambda x: abs(x[1] - x[3] - pr_err)):  # type: ignore
      svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed, cno = c
      print("svid: %3d  pseudorange: %10.2f m  speed: %8.2f m/s   meas: %12.2f  speed: %10.2f   meas_err: %10.3f speed_err: %8.3f cno: %d" %
        (svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed,
        ublox_psuedorange - qcom_psuedorange - pr_err, ublox_speed - qcom_speed - speed_err, cno))



