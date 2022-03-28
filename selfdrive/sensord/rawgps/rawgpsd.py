#!/usr/bin/env python3
import os
import sys
import signal
import itertools
import math
import time
from typing import NoReturn
from struct import unpack_from, calcsize

import cereal.messaging as messaging
from cereal import log
from selfdrive.swaglog import cloudlog

from selfdrive.sensord.rawgps.modemdiag import ModemDiag, DIAG_LOG_F, setup_logs
from selfdrive.sensord.rawgps.structs import dict_unpacker
from selfdrive.sensord.rawgps.structs import gps_measurement_report, gps_measurement_report_sv
from selfdrive.sensord.rawgps.structs import glonass_measurement_report, glonass_measurement_report_sv
from selfdrive.sensord.rawgps.structs import LOG_GNSS_GPS_MEASUREMENT_REPORT, LOG_GNSS_GLONASS_MEASUREMENT_REPORT
from selfdrive.sensord.rawgps.structs import position_report, LOG_GNSS_POSITION_REPORT

DEBUG = int(os.getenv("DEBUG", "0"))==1

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

def main() -> NoReturn:
  unpack_gps_meas, size_gps_meas = dict_unpacker(gps_measurement_report, True)
  unpack_gps_meas_sv, size_gps_meas_sv = dict_unpacker(gps_measurement_report_sv, True)

  unpack_glonass_meas, size_glonass_meas = dict_unpacker(glonass_measurement_report, True)
  unpack_glonass_meas_sv, size_glonass_meas_sv = dict_unpacker(glonass_measurement_report_sv, True)

  log_types = [
    LOG_GNSS_GPS_MEASUREMENT_REPORT,
    LOG_GNSS_GLONASS_MEASUREMENT_REPORT,
  ]
  pub_types = ['qcomGnss']
  if int(os.getenv("PUBLISH_EXTERNAL", "0")) == 1:
    unpack_position, _ = dict_unpacker(position_report)
    log_types.append(LOG_GNSS_POSITION_REPORT)
    pub_types.append("gpsLocationExternal")

  # disable DPO power savings for more accuracy
  os.system("mmcli -m 0 --command='AT+QGPSCFG=\"dpoenable\",0'")
  os.system("mmcli -m 0 --location-enable-gps-raw --location-enable-gps-nmea")
  diag = ModemDiag()

  def try_setup_logs(diag, log_types):
    for _ in range(5):
      try:
        setup_logs(diag, log_types)
        break
      except Exception:
        pass

  def disable_logs(sig, frame):
    os.system("mmcli -m 0 --location-disable-gps-raw --location-disable-gps-nmea")
    cloudlog.warning("rawgpsd: shutting down")
    try_setup_logs(diag, [])
    cloudlog.warning("rawgpsd: logs disabled")
    sys.exit(0)
  signal.signal(signal.SIGINT, disable_logs)
  try_setup_logs(diag, log_types)
  cloudlog.warning("rawgpsd: setup logs done")

  pm = messaging.PubMaster(pub_types)

  while 1:
    opcode, payload = diag.recv()
    assert opcode == DIAG_LOG_F
    (pending_msgs, log_outer_length), inner_log_packet = unpack_from('<BH', payload), payload[calcsize('<BH'):]
    if pending_msgs > 0:
      cloudlog.debug("have %d pending messages" % pending_msgs)
    assert log_outer_length == len(inner_log_packet)
    (log_inner_length, log_type, log_time), log_payload = unpack_from('<HHQ', inner_log_packet), inner_log_packet[calcsize('<HHQ'):]
    assert log_inner_length == len(inner_log_packet)
    if log_type not in log_types:
      continue
    if DEBUG:
      print("%.2f: got log: %x len %d" % (time.time(), log_type, len(log_payload)))
    if log_type == LOG_GNSS_POSITION_REPORT:
      report = unpack_position(log_payload)
      if report["u_PosSource"] != 2:
        continue
      vNED = [report["q_FltVelEnuMps[1]"], report["q_FltVelEnuMps[0]"], -report["q_FltVelEnuMps[2]"]]
      vNEDsigma = [report["q_FltVelSigmaMps[1]"], report["q_FltVelSigmaMps[0]"], -report["q_FltVelSigmaMps[2]"]]

      msg = messaging.new_message('gpsLocationExternal')
      gps = msg.gpsLocationExternal
      gps.flags = 1
      gps.latitude = report["t_DblFinalPosLatLon[0]"] * 180/math.pi
      gps.longitude = report["t_DblFinalPosLatLon[1]"] * 180/math.pi
      gps.altitude = report["q_FltFinalPosAlt"]
      gps.speed = math.sqrt(sum([x**2 for x in vNED]))
      gps.bearingDeg = report["q_FltHeadingRad"] * 180/math.pi
      # TODO: this probably isn't right, use laika for this
      gps.timestamp = report['w_GpsWeekNumber']*604800*1000 + report['q_GpsFixTimeMs']
      gps.source = log.GpsLocationData.SensorSource.qcomdiag
      gps.vNED = vNED
      gps.verticalAccuracy = report["q_FltVdop"]
      gps.bearingAccuracyDeg = report["q_FltHeadingUncRad"] * 180/math.pi
      gps.speedAccuracy = math.sqrt(sum([x**2 for x in vNEDsigma]))

      pm.send('gpsLocationExternal', msg)

    if log_type in [LOG_GNSS_GPS_MEASUREMENT_REPORT, LOG_GNSS_GLONASS_MEASUREMENT_REPORT]:
      msg = messaging.new_message('qcomGnss')

      gnss = msg.qcomGnss
      gnss.logTs = log_time
      gnss.init('measurementReport')
      report = gnss.measurementReport

      if log_type == LOG_GNSS_GPS_MEASUREMENT_REPORT:
        dat = unpack_gps_meas(log_payload)
        sats = log_payload[size_gps_meas:]
        unpack_meas_sv, size_meas_sv = unpack_gps_meas_sv, size_gps_meas_sv
        report.source = 0  # gps
        measurement_status_fields = (measurementStatusFields.items(), measurementStatusGPSFields.items())
      elif log_type == LOG_GNSS_GLONASS_MEASUREMENT_REPORT:
        dat = unpack_glonass_meas(log_payload)
        sats = log_payload[size_glonass_meas:]
        unpack_meas_sv, size_meas_sv = unpack_glonass_meas_sv, size_glonass_meas_sv
        report.source = 1  # glonass
        measurement_status_fields = (measurementStatusFields.items(), measurementStatusGlonassFields.items())
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
      report.init('sv', dat['svCount'])
      if dat['svCount'] > 0:
        assert len(sats)//dat['svCount'] == size_meas_sv
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
              for kk,vv in itertools.chain(*measurement_status_fields):
                setattr(sv.measurementStatus, kk, bool(v & (1<<vv)))
            elif k == "miscStatus":
              for kk,vv in miscStatusFields.items():
                setattr(sv.measurementStatus, kk, bool(v & (1<<vv)))
            elif k == "pad":
              pass
            else:
              setattr(sv, k, v)

      pm.send('qcomGnss', msg)

if __name__ == "__main__":
  main()
