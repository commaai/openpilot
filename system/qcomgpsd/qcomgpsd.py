#!/usr/bin/env python3
import os
import sys
import signal
import itertools
import math
import time
from serial import Serial
import datetime
from typing import NoReturn
from struct import unpack_from, calcsize, pack

from cereal import log
import cereal.messaging as messaging
from openpilot.common.gpio import gpio_init, gpio_set
from openpilot.common.utils import retry
from openpilot.common.time_helpers import system_time_valid
from openpilot.system.hardware.tici.pins import GPIO
from openpilot.common.swaglog import cloudlog
from openpilot.system.qcomgpsd.modemdiag import ModemDiag, DIAG_LOG_F, setup_logs, send_recv
from openpilot.system.qcomgpsd.structs import (dict_unpacker, position_report, relist,
                                              gps_measurement_report, gps_measurement_report_sv,
                                              glonass_measurement_report, glonass_measurement_report_sv,
                                              oemdre_measurement_report, oemdre_measurement_report_sv, oemdre_svpoly_report,
                                              LOG_GNSS_GPS_MEASUREMENT_REPORT, LOG_GNSS_GLONASS_MEASUREMENT_REPORT,
                                              LOG_GNSS_POSITION_REPORT, LOG_GNSS_OEMDRE_MEASUREMENT_REPORT,
                                              LOG_GNSS_OEMDRE_SVPOLY_REPORT)

DEBUG = int(os.getenv("DEBUG", "0"))==1

LOG_TYPES = [
  LOG_GNSS_GPS_MEASUREMENT_REPORT,
  LOG_GNSS_GLONASS_MEASUREMENT_REPORT,
  LOG_GNSS_OEMDRE_MEASUREMENT_REPORT,
  LOG_GNSS_POSITION_REPORT,
  LOG_GNSS_OEMDRE_SVPOLY_REPORT,
]


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

@retry(attempts=10, delay=1.0)
def try_setup_logs(diag, logs):
  return setup_logs(diag, logs)

AT_PORT = "/dev/modem_at0"

@retry(attempts=5, delay=1.0)
def at_cmd(cmd: str) -> str:
  with Serial(AT_PORT, baudrate=115200, timeout=5) as ser:
    ser.reset_input_buffer()
    ser.write(f"{cmd}\r".encode())
    lines = []
    while True:
      line = ser.readline()
      if not line:
        raise RuntimeError(f"AT command timeout: {cmd}")
      line = line.decode('utf-8', errors='replace').strip()
      if line in ("OK", "ERROR") or line.startswith("+CME ERROR"):
        break
      if line and line != cmd:
        lines.append(line)
  return '\n'.join(lines)

def gps_enabled() -> bool:
  return "QGPS: 1" in at_cmd("AT+QGPS?")

@retry(attempts=5, delay=1.0)
def setup_quectel(diag: ModemDiag):
  # enable OEMDRE in the NV
  # TODO: it has to reboot for this to take effect
  DIAG_NV_READ_F = 38
  DIAG_NV_WRITE_F = 39
  NV_GNSS_OEM_FEATURE_MASK = 7165
  send_recv(diag, DIAG_NV_WRITE_F, pack('<HI', NV_GNSS_OEM_FEATURE_MASK, 1))
  send_recv(diag, DIAG_NV_READ_F, pack('<H', NV_GNSS_OEM_FEATURE_MASK))

  try_setup_logs(diag, LOG_TYPES)

  if gps_enabled():
    at_cmd("AT+QGPSEND")

  # disable DPO power savings for more accuracy
  at_cmd("AT+QGPSCFG=\"dpoenable\",0")
  # don't automatically turn on GNSS on powerup
  at_cmd("AT+QGPSCFG=\"autogps\",0")

  if system_time_valid():
    time_str = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).strftime("%Y/%m/%d,%H:%M:%S")
    at_cmd(f"AT+QGPSXTRATIME=0,\"{time_str}\",1,1,1000")

  at_cmd("AT+QGPSCFG=\"outport\",\"usbnmea\"")
  at_cmd("AT+QGPS=1")

  # enable OEMDRE mode
  DIAG_SUBSYS_CMD_F = 75
  DIAG_SUBSYS_GPS = 13
  CGPS_DIAG_PDAPI_CMD = 0x64
  CGPS_OEM_CONTROL = 202
  GPSDIAG_OEMFEATURE_DRE = 1
  GPSDIAG_OEM_DRE_ON = 1

  # gpsdiag_OemControlReqType
  send_recv(diag, DIAG_SUBSYS_CMD_F, pack('<BHBBIIII',
    DIAG_SUBSYS_GPS,           # Subsystem Id
    CGPS_DIAG_PDAPI_CMD,       # Subsystem Command Code
    CGPS_OEM_CONTROL,          # CGPS Command Code
    0,                         # Version
    GPSDIAG_OEMFEATURE_DRE,
    GPSDIAG_OEM_DRE_ON,
    0,0
  ))


def teardown_quectel(diag):
  at_cmd("AT+QGPSCFG=\"outport\",\"none\"")
  if gps_enabled():
    at_cmd("AT+QGPSEND")
  try_setup_logs(diag, [])


def wait_for_modem():
  cloudlog.warning("waiting for modem to come up")
  while not os.path.exists(AT_PORT):
    time.sleep(0.5)
  # wait until the modem GNSS subsystem responds
  while True:
    try:
      resp = at_cmd("AT+QGPS?")
      if "+QGPS:" in resp:
        return
    except Exception:
      pass
    time.sleep(0.5)


def main() -> NoReturn:
  unpack_gps_meas, size_gps_meas = dict_unpacker(gps_measurement_report, True)
  unpack_gps_meas_sv, size_gps_meas_sv = dict_unpacker(gps_measurement_report_sv, True)

  unpack_glonass_meas, size_glonass_meas = dict_unpacker(glonass_measurement_report, True)
  unpack_glonass_meas_sv, size_glonass_meas_sv = dict_unpacker(glonass_measurement_report_sv, True)

  unpack_oemdre_meas, size_oemdre_meas = dict_unpacker(oemdre_measurement_report, True)
  unpack_oemdre_meas_sv, size_oemdre_meas_sv = dict_unpacker(oemdre_measurement_report_sv, True)

  unpack_svpoly, _ = dict_unpacker(oemdre_svpoly_report, True)
  unpack_position, _ = dict_unpacker(position_report)

  unpack_position, _ = dict_unpacker(position_report)

  wait_for_modem()

  def cleanup(sig, frame):
    cloudlog.warning("caught sig disabling quectel gps")

    gpio_set(GPIO.GNSS_PWR_EN, False)
    try:
      teardown_quectel(diag)
      cloudlog.warning("quectel cleanup done")
    except NameError:
      cloudlog.warning('quectel not yet setup')

    sys.exit(0)
  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  # connect to modem
  diag = ModemDiag()
  setup_quectel(diag)
  cloudlog.warning("quectel setup done")
  gpio_init(GPIO.GNSS_PWR_EN, True)
  gpio_set(GPIO.GNSS_PWR_EN, True)

  pm = messaging.PubMaster(['qcomGnss', 'gpsLocation'])

  while 1:
    opcode, payload = diag.recv()
    if opcode != DIAG_LOG_F:
      cloudlog.error(f"Unhandled opcode: {opcode}")
      continue

    (pending_msgs, log_outer_length), inner_log_packet = unpack_from('<BH', payload), payload[calcsize('<BH'):]
    if pending_msgs > 0:
      cloudlog.debug(f"have {pending_msgs} pending messages")
    assert log_outer_length == len(inner_log_packet)

    (log_inner_length, log_type, log_time), log_payload = unpack_from('<HHQ', inner_log_packet), inner_log_packet[calcsize('<HHQ'):]
    assert log_inner_length == len(inner_log_packet)

    if log_type not in LOG_TYPES:
      continue

    if DEBUG:
      print(f"{time.time():.4f}: got log: {log_type} len {len(log_payload)}")  # noqa: TID251

    if log_type == LOG_GNSS_OEMDRE_MEASUREMENT_REPORT:
      msg = messaging.new_message('qcomGnss', valid=True)

      gnss = msg.qcomGnss
      gnss.logTs = log_time
      gnss.init('drMeasurementReport')
      report = gnss.drMeasurementReport

      dat = unpack_oemdre_meas(log_payload)
      for k,v in dat.items():
        if k in ["gpsTimeBias", "gpsClockTimeUncertainty"]:
          k += "Ms"
        if k == "version":
          assert v == 2
        elif k == "svCount" or k.startswith("cdmaClockInfo["):
          # TODO: should we save cdmaClockInfo?
          pass
        elif k == "systemRtcValid":
          setattr(report, k, bool(v))
        else:
          setattr(report, k, v)

      report.init('sv', dat['svCount'])
      sats = log_payload[size_oemdre_meas:]
      for i in range(dat['svCount']):
        sat = unpack_oemdre_meas_sv(sats[size_oemdre_meas_sv*i:size_oemdre_meas_sv*(i+1)])
        sv = report.sv[i]
        sv.init('measurementStatus')
        for k,v in sat.items():
          if k in ["unkn", "measurementStatus2"]:
            pass
          elif k == "multipathEstimateValid":
            sv.measurementStatus.multipathEstimateIsValid = bool(v)
          elif k == "directionValid":
            sv.measurementStatus.directionIsValid = bool(v)
          elif k == "goodParity":
            setattr(sv, k, bool(v))
          elif k == "measurementStatus":
            for kk,vv in measurementStatusFields.items():
              setattr(sv.measurementStatus, kk, bool(v & (1<<vv)))
          else:
            setattr(sv, k, v)
      pm.send('qcomGnss', msg)
    elif log_type == LOG_GNSS_POSITION_REPORT:
      report = unpack_position(log_payload)
      if report["u_PosSource"] != 2:
        continue
      # uint16_t max is an invalid sentinel value from the modem
      if report['w_GpsWeekNumber'] >= 0xFFFF:
        continue
      vNED = [report["q_FltVelEnuMps[1]"], report["q_FltVelEnuMps[0]"], -report["q_FltVelEnuMps[2]"]]
      vNEDsigma = [report["q_FltVelSigmaMps[1]"], report["q_FltVelSigmaMps[0]"], -report["q_FltVelSigmaMps[2]"]]

      msg = messaging.new_message('gpsLocation', valid=True)
      gps = msg.gpsLocation
      gps.latitude = report["t_DblFinalPosLatLon[0]"] * 180/math.pi
      gps.longitude = report["t_DblFinalPosLatLon[1]"] * 180/math.pi
      gps.altitude = report["q_FltFinalPosAlt"]
      gps.speed = math.sqrt(sum([x**2 for x in vNED]))
      gps.bearingDeg = report["q_FltHeadingRad"] * 180/math.pi

      # TODO needs update if there is another leap second, after june 2024?
      dt_timestamp = (datetime.datetime(1980, 1, 6, 0, 0, 0, 0, datetime.UTC) +
                      datetime.timedelta(weeks=report['w_GpsWeekNumber']) +
                      datetime.timedelta(seconds=(1e-3*report['q_GpsFixTimeMs'] - 18)))
      gps.unixTimestampMillis = dt_timestamp.timestamp()*1e3
      gps.source = log.GpsLocationData.SensorSource.qcomdiag
      gps.vNED = vNED
      gps.verticalAccuracy = report["q_FltVdop"]
      gps.bearingAccuracyDeg = report["q_FltHeadingUncRad"] * 180/math.pi if (report["q_FltHeadingUncRad"] != 0) else 180
      gps.speedAccuracy = math.sqrt(sum([x**2 for x in vNEDsigma]))
      # quectel gps verticalAccuracy is clipped to 500, set invalid if so
      gps.hasFix = gps.verticalAccuracy != 500
      pm.send('gpsLocation', msg)

    elif log_type == LOG_GNSS_OEMDRE_SVPOLY_REPORT:
      msg = messaging.new_message('qcomGnss', valid=True)
      dat = unpack_svpoly(log_payload)
      dat = relist(dat)
      gnss = msg.qcomGnss
      gnss.logTs = log_time
      gnss.init('drSvPoly')
      poly = gnss.drSvPoly
      for k,v in dat.items():
        if k == "version":
          assert v == 2
        elif k == "flags":
          pass
        else:
          setattr(poly, k, v)

      '''
      # Timestamp glonass polys with GPSTime
      from laika.gps_time import GPSTime, utc_to_gpst, get_leap_seconds
      from laika.helpers import get_prn_from_nmea_id
      prn = get_prn_from_nmea_id(poly.svId)
      if prn[0] == 'R':
        epoch = GPSTime(current_gps_time.week, (poly.t0 - 3*SECS_IN_HR + SECS_IN_DAY) % (SECS_IN_WEEK) + get_leap_seconds(current_gps_time))
      else:
        epoch = GPSTime(current_gps_time.week, poly.t0)

      # handle week rollover
      if epoch.tow < SECS_IN_DAY and current_gps_time.tow > 6*SECS_IN_DAY:
        epoch.week += 1
      elif epoch.tow > 6*SECS_IN_DAY and current_gps_time.tow < SECS_IN_DAY:
        epoch.week -= 1

      poly.gpsWeek = epoch.week
      poly.gpsTow = epoch.tow
      '''
      pm.send('qcomGnss', msg)

    elif log_type in [LOG_GNSS_GPS_MEASUREMENT_REPORT, LOG_GNSS_GLONASS_MEASUREMENT_REPORT]:
      msg = messaging.new_message('qcomGnss', valid=True)

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
        raise RuntimeError(f"invalid log_type: {log_type}")

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
