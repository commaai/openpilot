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

def unlock_serial():
  os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/unbind\'')
  os.system('sudo su -c \'echo "1-1.1:1.0" > /sys/bus/usb/drivers/option/bind\'')
  time.sleep(0.5)
  os.system("sudo chmod 666 /dev/ttyUSB0")

def open_serial():
  # TODO: this is a hack to get around modemmanager's exclusive open
  try:
    return Serial("/dev/ttyUSB0", baudrate=115200, rtscts=True, dsrdtr=True)
  except Exception:
    print("unlocking serial...")
    unlock_serial()
    return Serial("/dev/ttyUSB0", baudrate=115200, rtscts=True, dsrdtr=True)

ccitt_crc16 = mkCrcFun(0x11021, initCrc=0, xorOut=0xffff)
ESCAPE_CHAR = b'\x7d'
TRAILER_CHAR = b'\x7e'

def hdlc_encapsulate(payload):
  payload += pack('<H', ccitt_crc16(payload))
  payload = payload.replace(ESCAPE_CHAR, bytes([ESCAPE_CHAR[0], ESCAPE_CHAR[0] ^ 0x20]))
  payload = payload.replace(TRAILER_CHAR, bytes([ESCAPE_CHAR[0], TRAILER_CHAR[0] ^ 0x20]))
  payload += TRAILER_CHAR
  return payload

def hdlc_decapsulate(payload):
  assert len(payload) >= 3
  assert payload[-1:] == TRAILER_CHAR
  payload = payload[:-1]
  payload = payload.replace(bytes([ESCAPE_CHAR[0], TRAILER_CHAR[0] ^ 0x20]), TRAILER_CHAR)
  payload = payload.replace(bytes([ESCAPE_CHAR[0], ESCAPE_CHAR[0] ^ 0x20]), ESCAPE_CHAR)
  assert payload[-2:] == pack('<H', ccitt_crc16(payload[:-2]))
  return payload[:-2]

DIAG_LOG_F = 16
DIAG_LOG_CONFIG_F = 115
LOG_CONFIG_RETRIEVE_ID_RANGES_OP = 1
LOG_CONFIG_SET_MASK_OP = 3
LOG_CONFIG_SUCCESS_S = 0

def recv(serial):
  raw_payload = []
  while 1:
    char_read = serial.read()
    raw_payload.append(char_read)
    if char_read.endswith(TRAILER_CHAR):
      break
  raw_payload = b''.join(raw_payload)
  unframed_message = hdlc_decapsulate(raw_payload)
  return unframed_message[0], unframed_message[1:]

def send_recv(serial, packet_type, packet_payload):
  serial.write(hdlc_encapsulate(bytes([packet_type]) + packet_payload))
  while 1:
    opcode, payload = recv(serial)
    if opcode != DIAG_LOG_F:
      break
  return opcode, payload

TYPES_FOR_RAW_PACKET_LOGGING = [
  #0x1476,
  0x1477,
  #0x1480,

  #0x1478,
  #0x1756,
  #0x1886,

  #0x14DE,
  #0x14E1,

  #0x1838,
  #0x147B,
  #0x147E,
  #0x1488,
  #0x1516,
]

def setup_rawgps():
  os.system("mmcli -m 0 --location-enable-gps-raw --location-enable-gps-nmea")
  opcode, payload = send_recv(serial, DIAG_LOG_CONFIG_F, pack('<3xI', LOG_CONFIG_RETRIEVE_ID_RANGES_OP))

  header_spec = '<3xII'
  operation, status = unpack_from(header_spec, payload)
  assert operation == LOG_CONFIG_RETRIEVE_ID_RANGES_OP
  assert status == LOG_CONFIG_SUCCESS_S

  log_masks = unpack_from('<16I', payload, calcsize(header_spec))
  print(log_masks)

  for log_type, log_mask_bitsize in enumerate(log_masks):
    if log_mask_bitsize:
      log_mask = [0] * ((log_mask_bitsize+7)//8)
      for i in range(log_mask_bitsize):
        if ((log_type<<12)|i) in TYPES_FOR_RAW_PACKET_LOGGING:
          log_mask[i//8] |= 1 << (i%8)
      opcode, payload = send_recv(serial, DIAG_LOG_CONFIG_F, pack('<3xIII',
          LOG_CONFIG_SET_MASK_OP,
          log_type,
          log_mask_bitsize
      ) + bytes(log_mask))
      operation, status = unpack_from(header_spec, payload)
      assert operation == LOG_CONFIG_SET_MASK_OP
      assert status == LOG_CONFIG_SUCCESS_S

svStructNames = ["svId", "observationState", "observations", 
  "goodObservations", "gpsParityErrorCount", "filterStages",
  "carrierNoise", "latency", "predetectInterval", "postdetections",
  "unfilteredMeasurementIntegral", "unfilteredMeasurementFraction", 
  "unfilteredTimeUncertainty", "unfilteredSpeed", "unfilteredSpeedUncertainty",
  "measurementStatus", "miscStatus", "multipathEstimate", 
  "azimuth", "elevation", "carrierPhaseCyclesIntegral", "carrierPhaseCyclesFraction",
  "fineSpeed", "fineSpeedUncertainty", "cycleSlipCount"]


position_packet = """
  uint8       u_Version;                /* Version number of DM log */
  uint32      q_Fcount;                 /* Local millisecond counter */
  uint8       u_PosSource;              /* Source of position information */ /*  0: None 1: Weighted least-squares 2: Kalman filter 3: Externally injected 4: Internal database    */
  uint32      q_Reserved1;              /* Reserved memory field */
  uint16      w_PosVelFlag;             /* Position velocity bit field: (see DM log 0x1476 documentation) */
  uint32      q_PosVelFlag2;            /* Position velocity 2 bit field: (see DM log 0x1476 documentation) */
  uint8       u_FailureCode;            /* Failure code: (see DM log 0x1476 documentation) */
  uint16      w_FixEvents;              /* Fix events bit field: (see DM log 0x1476 documentation) */
  uint32 _fake_align_week_number;
  uint16      w_GpsWeekNumber;          /* GPS week number of position */
  uint32      q_GpsFixTimeMs;           /* GPS fix time of week of in milliseconds */
  uint8       u_GloNumFourYear;         /* Number of Glonass four year cycles */
  uint16      w_GloNumDaysInFourYear;   /* Glonass calendar day in four year cycle */
  uint32      q_GloFixTimeMs;           /* Glonass fix time of day in milliseconds */
  uint32      q_PosCount;               /* Integer count of the number of unique positions reported */
  uint64      t_DblFinalPosLatLon[2];   /* Final latitude and longitude of position in radians */
  uint32      q_FltFinalPosAlt;         /* Final height-above-ellipsoid altitude of position */
  uint32      q_FltHeadingRad;          /* User heading in radians */
  uint32      q_FltHeadingUncRad;       /* User heading uncertainty in radians */
  uint32      q_FltVelEnuMps[3];        /* User velocity in east, north, up coordinate frame. In meters per second. */
  uint32      q_FltVelSigmaMps[3];      /* Gaussian 1-sigma value for east, north, up components of user velocity */
  uint32      q_FltClockBiasMeters;     /* Receiver clock bias in meters */
  uint32      q_FltClockBiasSigmaMeters;  /* Gaussian 1-sigma value for receiver clock bias in meters */
  uint32      q_FltGGTBMeters;          /* GPS to Glonass time bias in meters */
  uint32      q_FltGGTBSigmaMeters;     /* Gaussian 1-sigma value for GPS to Glonass time bias uncertainty in meters */
  uint32      q_FltGBTBMeters;          /* GPS to BeiDou time bias in meters */
  uint32      q_FltGBTBSigmaMeters;     /* Gaussian 1-sigma value for GPS to BeiDou time bias uncertainty in meters */
  uint32      q_FltBGTBMeters;          /* BeiDou to Glonass time bias in meters */
  uint32      q_FltBGTBSigmaMeters;     /* Gaussian 1-sigma value for BeiDou to Glonass time bias uncertainty in meters */
  uint32      q_FltFiltGGTBMeters;      /* Filtered GPS to Glonass time bias in meters */
  uint32      q_FltFiltGGTBSigmaMeters; /* Filtered Gaussian 1-sigma value for GPS to Glonass time bias uncertainty in meters */
  uint32      q_FltFiltGBTBMeters;      /* Filtered GPS to BeiDou time bias in meters */
  uint32      q_FltFiltGBTBSigmaMeters; /* Filtered Gaussian 1-sigma value for GPS to BeiDou time bias uncertainty in meters */
  uint32      q_FltFiltBGTBMeters;      /* Filtered BeiDou to Glonass time bias in meters */
  uint32      q_FltFiltBGTBSigmaMeters; /* Filtered Gaussian 1-sigma value for BeiDou to Glonass time bias uncertainty in meters */
  uint32      q_FltSftOffsetSec;        /* SFT offset as computed by WLS in seconds */
  uint32      q_FltSftOffsetSigmaSec;   /* Gaussian 1-sigma value for SFT offset in seconds */
  uint32      q_FltClockDriftMps;       /* Clock drift (clock frequency bias) in meters per second */
  uint32      q_FltClockDriftSigmaMps;  /* Gaussian 1-sigma value for clock drift in meters per second */
  uint32      q_FltFilteredAlt;         /* Filtered height-above-ellipsoid altitude in meters as computed by WLS */
  uint32      q_FltFilteredAltSigma;    /* Gaussian 1-sigma value for filtered height-above-ellipsoid altitude in meters */
  uint32      q_FltRawAlt;              /* Raw height-above-ellipsoid altitude in meters as computed by WLS */
  uint32      q_FltRawAltSigma;         /* Gaussian 1-sigma value for raw height-above-ellipsoid altitude in meters */
  uint32   align_Flt[14];
  uint32      q_FltPdop;                /* 3D position dilution of precision as computed from the unweighted 
  uint32      q_FltHdop;                /* Horizontal position dilution of precision as computed from the unweighted least-squares covariance matrix */
  uint32      q_FltVdop;                /* Vertical position dilution of precision as computed from the unweighted least-squares covariance matrix */
  uint8       u_EllipseConfidence;      /* Statistical measure of the confidence (percentage) associated with the uncertainty ellipse values */
  uint32      q_FltEllipseAngle;        /* Angle of semimajor axis with respect to true North, with increasing angles moving clockwise from North. In units of degrees. */
  uint32      q_FltEllipseSemimajorAxis;  /* Semimajor axis of final horizontal position uncertainty error ellipse.  In units of meters. */
  uint32      q_FltEllipseSemiminorAxis;  /* Semiminor axis of final horizontal position uncertainty error ellipse.  In units of meters. */
  uint32      q_FltPosSigmaVertical;    /* Gaussian 1-sigma value for final position height-above-ellipsoid altitude in meters */
  uint8       u_HorizontalReliability;  /* Horizontal position reliability 0: Not set 1: Very Low 2: Low 3: Medium 4: High    */
  uint8       u_VerticalReliability;    /* Vertical position reliability */
  uint16      w_Reserved2;              /* Reserved memory field */
  uint32      q_FltGnssHeadingRad;      /* User heading in radians derived from GNSS only solution  */
  uint32      q_FltGnssHeadingUncRad;   /* User heading uncertainty in radians derived from GNSS only solution  */
  uint32      q_SensorDataUsageMask;    /* Denotes which additional sensor data were used to compute this position fix.  BIT[0] 0x00000001 <96> Accelerometer BIT[1] 0x00000002 <96> Gyro 0x0000FFFC - Reserved A bit set to 1 indicates that certain fields as defined by the SENSOR_AIDING_MASK were aided with sensor data*/
  uint32      q_SensorAidMask;         /* Denotes which component of the position report was assisted with additional sensors defined in SENSOR_DATA_USAGE_MASK BIT[0] 0x00000001 <96> Heading aided with sensor data BIT[1] 0x00000002 <96> Speed aided with sensor data BIT[2] 0x00000004 <96> Position aided with sensor data BIT[3] 0x00000008 <96> Velocity aided with sensor data 0xFFFFFFF0 <96> Reserved */
  uint8       u_NumGpsSvsUsed;          /* The number of GPS SVs used in the fix */
  uint8       u_TotalGpsSvs;            /* Total number of GPS SVs detected by searcher, including ones not used in position calculation */
  uint8       u_NumGloSvsUsed;          /* The number of Glonass SVs used in the fix */
  uint8       u_TotalGloSvs;            /* Total number of Glonass SVs detected by searcher, including ones not used in position calculation */
  uint8       u_NumBdsSvsUsed;          /* The number of BeiDou SVs used in the fix */
  uint8       u_TotalBdsSvs;            /* Total number of BeiDou SVs detected by searcher, including ones not used in position calculation */
"""


if __name__ == "__main__":
  st = "<"
  nams = []
  for l in position_packet.strip().split("\n"):
    typ, nam = l.split(";")[0].split()
    #print(typ, nam)
    if '_Flt' in nam:
      st += "f"
    elif '_Dbl' in nam:
      st += "d"
    elif typ == "uint8":
      st += "B"
    elif typ == "uint32":
      st += "I"
    elif typ == "uint16":
      st += "H"
    elif typ == "uint64":
      st += "Q"
    else:
      assert False
    nams.append(nam)
    if '[' in nam:
      more = int(nam.split("[")[1].split("]")[0])-1
      st += st[-1]*more
      nams += nams[-1:]*more

  serial = open_serial()
  serial.flush()

  #setup_rawgps()

  C = 299792458 #/1000.0

  sm = messaging.SubMaster(['ubloxGnss'])
  meas = None

  while 1:
    opcode, payload = recv(serial)
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
      if log_type == 0x1477:
        dat = unpack("<BIHIffffB", log_payload[0:28])
        ll = 28
      else:
        dat = unpack("<BIBHIffffB", log_payload[0:29])
        ll = 29
      print(dat)
      sats = log_payload[ll:]
      L = 70
      assert len(sats)//dat[-1] == L
      car = []
      for i in range(dat[-1]):
        sat = dict(zip(svStructNames, unpack("<BBBBHBHhBHIffffIBIffiHffBI", sats[L*i:L*i+L])[:-1]))

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
        recv_time = dat[3] / 1000.
        sat_time = (sat['unfilteredMeasurementIntegral'] + sat['unfilteredMeasurementFraction'] + sat['latency']) / 1000.
        qcom_psuedorange = (recv_time - sat_time)*constants.SPEED_OF_LIGHT

        #print(sat)
        car.append((sat['svId'], tm.pseudorange, ublox_speed, qcom_psuedorange, sat['unfilteredSpeed']))

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
        svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed = c
        pr_err.append(ublox_psuedorange - qcom_psuedorange)
        speed_err.append(ublox_speed - qcom_speed)
      pr_err = np.mean(pr_err)
      speed_err = np.mean(speed_err)
      print("avg psuedorange err %f avg speed err %f" % (pr_err, speed_err))

      for c in sorted(car):
        svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed = c
        print("svid: %3d  pseudorange: %10.2f m  speed: %8.2f m/s   meas: %12.2f  speed: %10.2f   meas_err: %f speed_err: %f" % \
          (svid, ublox_psuedorange, ublox_speed, qcom_psuedorange, qcom_speed,
          ublox_psuedorange - qcom_psuedorange - pr_err, ublox_speed - qcom_speed - speed_err))



  #header_spec = '<3xII'
  #operation, status = unpack_from(header_spec, payload)
  #log_masks = unpack_from('<16I', payload, calcsize(header_spec))

  """
  opcode, payload = self.diag_input.send_recv(DIAG_LOG_CONFIG_F, pack('<3xIII',
    LOG_CONFIG_SET_MASK_OP,
    log_type,
    log_mask_bitsize
  ) + log_mask)
  """

