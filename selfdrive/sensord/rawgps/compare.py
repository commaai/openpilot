#!/usr/bin/env python3
import os
import time
import numpy as np
from serial import Serial
from crcmod import mkCrcFun
from hexdump import hexdump
from struct import pack, unpack_from, calcsize, unpack
import cereal.messaging as messaging

from laika import constants
from selfdrive.sensord.rawgps.modemdiag import *
from selfdrive.sensord.rawgps.structs import *

TYPES_FOR_RAW_PACKET_LOGGING = [
  LOG_GNSS_GPS_MEASUREMENT_REPORT
]

if __name__ == "__main__":
  st, nams = parse_struct(position_report)
  unpack_gps_meas = dict_unpacker(gps_measurement_report, True)
  unpack_gps_meas_sv = dict_unpacker(gps_measurement_report_sv, True)

  os.system("mmcli -m 0 --location-enable-gps-raw --location-enable-gps-nmea")
  diag = ModemDiag()
  setup_logs(diag, TYPES_FOR_RAW_PACKET_LOGGING)

  sm = messaging.SubMaster(['ubloxGnss'])
  meas = None

  while 1:
    opcode, payload = diag.recv()
    assert opcode == DIAG_LOG_F
    (pending_msgs, log_outer_length), inner_log_packet = unpack_from('<BH', payload), payload[calcsize('<BH'):]
    (log_inner_length, log_type, log_time), log_payload = unpack_from('<HHQ', inner_log_packet), inner_log_packet[calcsize('<HHQ'):]
    print("%x len %d" % (log_type, len(log_payload)))

    sm.update()
    if sm['ubloxGnss'].which() == "measurementReport":
      meas = sm['ubloxGnss'].measurementReport.measurements

    if log_type == 0x1476 and log_payload[5] != 4:
      #hexdump(log_payload[0:calcsize(st)]) #+0x100])
      #ret = unpack(st, log_payload[0:227])
      ret = unpack(st, log_payload[0:calcsize(st)])
      sats = log_payload[calcsize(st):]
      dd = {}
      for x,y in list(zip(nams, ret)):
        print(x,y)
        dd[x] = y
    
      #hexdump(sats)
      for i in range(dd['u_NumGpsSvsUsed']):
        s = unpack("<BBBBHffff", sats[0x20+i*0x16:0x20+(i+1)*0x16])
        print(s)


      pass

    if log_type == 0x1477: # or log_type == 0x1480:
      dat = unpack_gps_meas(log_payload)
      ll = 28
      print(dat)
      sats = log_payload[ll:]
      L = 70
      assert len(sats)//dat['svCount'] == L
      car = []
      for i in range(dat['svCount']):
        sat = unpack_gps_meas_sv(sats[L*i:L*i+L])

        if (sat['measurementStatus'] & (1<<2)) == 0:
          continue
        if (sat['measurementStatus'] & (1<<13)) != 0:
          continue

        #if sat['observationState'] not in [5,7]:
        #  continue
        #  car.append((sat['svId'], "svid: %3d  pseudorange: %f m  doppler: %f hz" % (sat['svId'], pr, sat['unfilteredSpeed'])))
        tm = None
        if meas is not None:
          for m in meas:
            if sat['svId'] == m.svId and m.gnssId == 0 and m.sigId == 0:
              tm = m
        if tm is None:
          continue

        ublox_speed = -(constants.SPEED_OF_LIGHT / constants.GPS_L1) * tm.doppler
        recv_time = dat['milliseconds'] / 1000.
        sat_time = (sat['unfilteredMeasurementIntegral'] + sat['unfilteredMeasurementFraction'] + sat['latency']) / 1000.
        qcom_psuedorange = (recv_time - sat_time)*constants.SPEED_OF_LIGHT

        #print(sat)
        car.append((sat['svId'], tm.pseudorange, ublox_speed, qcom_psuedorange, sat['unfilteredSpeed'], tm.cno))

        #pr = (dat[3] - (sat['unfilteredMeasurementIntegral'] + sat['unfilteredMeasurementFraction'])) * C

        #print("svid: %3d  pseudorange: %10.2f m  speed: %8.2f m/s   meas: %12.2f  speed: %10.2f   rat %f off %f lat %d" % \
        #  (sat['svId'], tm.pseudorange, ublox_speed, qcom_psuedorange, sat['unfilteredSpeed'],
        #    ublox_speed - sat['unfilteredSpeed'] + 65.153366, tm.pseudorange - qcom_psuedorange - 733596.055438, sat['latency']))

        """
        if sat['observationState'] in [5,7]:
          prr = (dat[3]-sat['unfilteredMeasurementIntegral']) - sat['unfilteredMeasurementFraction']
          prr_std = sat['unfilteredTimeUncertainty']
          print("svid: %2d -- pr(m): %.2f  pr_std: %.2f  speed: %.2f m/s  speed_std: %.2f m/s" % (sat['svId'], prr*C, prr_std*C, sat['unfilteredSpeed'], sat['unfilteredSpeedUncertainty']))
        """
        #print(sat['svId'], sat['observationState'], sat['unfilteredMeasurementIntegral'], sat['unfilteredMeasurementFraction'], C*sat['unfilteredTimeUncertainty'], sat['unfilteredSpeed'], sat['unfilteredSpeedUncertainty'])
        #print("  ", sat)

      pr_err = []
      speed_err = []
      for c in car:
        ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed = c[1:5]
        pr_err.append(ublox_psuedorange - qcom_psuedorange)
        speed_err.append(ublox_speed - qcom_speed)
      pr_err = np.mean(pr_err)
      speed_err = np.mean(speed_err)
      print("avg psuedorange err %f avg speed err %f" % (pr_err, speed_err))

      for c in sorted(car, key=lambda x: abs(x[1] - x[3] - pr_err)):
        svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed, cno = c
        print("svid: %3d  pseudorange: %10.2f m  speed: %8.2f m/s   meas: %12.2f  speed: %10.2f   meas_err: %10.3f speed_err: %8.3f cno: %d" % \
          (svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed,
          ublox_psuedorange - qcom_psuedorange - pr_err, ublox_speed - qcom_speed - speed_err, cno))
