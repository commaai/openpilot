#!/usr/bin/env python3
import os
import itertools
from struct import unpack_from
import cereal.messaging as messaging

from selfdrive.sensord.rawgps.modemdiag import ModemDiag, DIAG_LOG_F, setup_logs
from selfdrive.sensord.rawgps.structs import *

miscStatusFields = {
  "multipathEstimateIsValid": 0,
  "directionIsValid": 1,
}

measurementStatusFields = {
  "subMillisecondIsValid": 0,
  "subBitTimeIsKnown": 1,
  "satelliteTimeIsKnown": 2,
  "bitEdgeConfirmedFromSignal": 3,
  "measuredVelocity": 4,
  "fineOrCoarseVelocity": 5,
  "lockPointValid": 6,
  "lockPointPositive": 7,

  "lastUpdateFromDifference": 9,
  "lastUpdateFromVelocityDifference": 10,
  "strongIndicationOfCrossCorelation": 11,
  "tentativeMeasurement": 12,
  "measurementNotUsable": 13,
  "sirCheckIsNeeded": 14,
  "probationMode": 15,

  "multipathIndicator": 24,
  "imdJammingIndicator": 25,
  "lteB13TxJammingIndicator": 26,
  "freshMeasurementIndicator": 27,
}

measurementStatusGPSFields = {
  "gpsRoundRobinRxDiversity": 18,
  "gpsRxDiversity": 19,
  "gpsLowBandwidthRxDiversityCombined": 20,
  "gpsHighBandwidthNu4": 21,
  "gpsHighBandwidthNu8": 22,
  "gpsHighBandwidthUniform": 23,
}

measurementStatusGlonassFields = {
  "glonassMeanderBitEdgeValid": 16,
  "glonassTimeMarkValid": 17
}

if __name__ == "__main__":
  unpack_gps_meas, size_gps_meas = dict_unpacker(gps_measurement_report, True)
  unpack_gps_meas_sv, size_gps_meas_sv = dict_unpacker(gps_measurement_report_sv, True)

  unpack_glonass_meas, size_glonass_meas = dict_unpacker(glonass_measurement_report, True)
  unpack_glonass_meas_sv, size_glonass_meas_sv = dict_unpacker(glonass_measurement_report_sv, True)

  log_types = [
    LOG_GNSS_GPS_MEASUREMENT_REPORT,
    LOG_GNSS_GLONASS_MEASUREMENT_REPORT,
  ]

  os.system("mmcli -m 0 --location-enable-gps-raw --location-enable-gps-nmea")
  diag = ModemDiag()
  setup_logs(diag, log_types)

  pm = messaging.PubMaster(['qcomGnss'])

  while 1:
    opcode, payload = diag.recv()
    assert opcode == DIAG_LOG_F
    (pending_msgs, log_outer_length), inner_log_packet = unpack_from('<BH', payload), payload[calcsize('<BH'):]
    (log_inner_length, log_type, log_time), log_payload = unpack_from('<HHQ', inner_log_packet), inner_log_packet[calcsize('<HHQ'):]
    if log_type not in log_types:
      continue
    
    msg = messaging.new_message('qcomGnss')
    if log_type in [LOG_GNSS_GPS_MEASUREMENT_REPORT, LOG_GNSS_GLONASS_MEASUREMENT_REPORT]:
      gnss = msg.qcomGnss
      gnss.logTs = log_time
      gnss.init('measurementReport')
      report = gnss.measurementReport

      if log_type == LOG_GNSS_GPS_MEASUREMENT_REPORT:
        dat = unpack_gps_meas(log_payload)
        sats = log_payload[size_gps_meas:]
        unpack_meas_sv, size_meas_sv = unpack_gps_meas_sv, size_gps_meas_sv
        report.source = 0  # gps
        get_measurement_status_fields = lambda: itertools.chain(measurementStatusFields.items(), measurementStatusGPSFields.items())
      elif log_type == LOG_GNSS_GLONASS_MEASUREMENT_REPORT:
        dat = unpack_glonass_meas(log_payload)
        sats = log_payload[size_glonass_meas:]
        unpack_meas_sv, size_meas_sv = unpack_glonass_meas_sv, size_glonass_meas_sv
        report.source = 1  # glonass
        get_measurement_status_fields = lambda: itertools.chain(measurementStatusFields.items(), measurementStatusGlonassFields.items())
      else:
        assert False

      for k,v in dat.items():
        if k == "version":
          assert v == 0
        elif k == "week":
          report.gpsWeek = v
        elif k == "svCount":
          pass
        else:
          setattr(report, k, v)
      assert len(sats)//dat['svCount'] == size_meas_sv
      report.init('sv', dat['svCount'])
      for i in range(dat['svCount']):
        sv = report.sv[i]
        sv.init('measurementStatus')
        sat = unpack_meas_sv(sats[size_meas_sv*i:size_meas_sv*(i+1)])
        for k,v in sat.items():
          if k == "parityErrorCount":
            sv.gpsParityErrorCount = v
          elif k == "frequencyIndex":
            sv.glonassFrequencyIndex = v
          elif k == "hemmingErrorCount":
            sv.glonassHemmingErrorCount = v
          elif k == "measurementStatus":
            for kk,vv in get_measurement_status_fields():
              setattr(sv.measurementStatus, kk, bool(v & (1<<vv)))
          elif k == "miscStatus":
            for kk,vv in miscStatusFields.items():
              setattr(sv.measurementStatus, kk, bool(v & (1<<vv)))
          elif k == "pad":
            pass
          else:
            setattr(sv, k, v)
    
    pm.send('qcomGnss', msg)
