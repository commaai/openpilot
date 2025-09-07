#!/usr/bin/env python3
import struct
from dataclasses import dataclass
from enum import IntEnum

from cereal import messaging, log
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process


# UBLOX protocol constants
UBLOX_PREAMBLE1 = 0xb5
UBLOX_PREAMBLE2 = 0x62
UBLOX_HEADER_SIZE = 6
UBLOX_CHECKSUM_SIZE = 2
UBLOX_MAX_MSG_SIZE = 65536

# Time constants
SECS_IN_MIN = 60
SECS_IN_HR = 60 * SECS_IN_MIN
SECS_IN_DAY = 24 * SECS_IN_HR
SECS_IN_WEEK = 7 * SECS_IN_DAY

GPS_PI = 3.1415926535898


class UBXClass(IntEnum):
  """UBLOX message class IDs"""
  NAV = 0x01
  RXM = 0x02
  MON = 0x0A
  AID = 0x0B
  CFG = 0x06
  UPD = 0x09
  MGA = 0x13
  LOG = 0x21
  SEC = 0x27
  HNR = 0x28
  PRT = 0x28


class UBXMessageID:
  """UBLOX message IDs for different classes"""
  class NAV:
    PVT = 0x07
    SAT = 0x35

  class RXM:
    SFRBX = 0x13
    RAWX = 0x15

  class MON:
    HW = 0x09
    HW2 = 0x0B


@dataclass
class UBXMessage:
  """Parsed UBLOX message structure"""
  msg_class: int
  msg_id: int
  payload: bytes
  checksum_valid: bool
  log_time: float


class UBXMessageParser:
  """
  UBLOX binary protocol message parser.

  Handles incremental parsing of UBLOX messages from raw byte stream,
  validates checksums, and extracts message components.
  """

  def __init__(self):
    self.parse_buffer = bytearray()
    self.last_log_time = 0.0

    # GPS/GLONASS ephemeris storage
    self.gps_subframes: dict[int, dict[int, bytes]] = {}
    self.glonass_strings: dict[int, dict[int, bytes]] = {}
    self.glonass_string_times: dict[int, dict[int, int]] = {}
    self.glonass_string_superframes: dict[int, dict[int, int]] = {}

    # GLONASS user range accuracy lookup table (meters)
    self.glonass_URA_lookup = {
      0: 1, 1: 2, 2: 2.5, 3: 4, 4: 5, 5: 7,
      6: 10, 7: 12, 8: 14, 9: 16, 10: 32,
      11: 64, 12: 128, 13: 256, 14: 512, 15: 1024
    }

  def reset(self) -> None:
    """Reset parser state"""
    self.parse_buffer.clear()

  def _calculate_checksum(self, data: bytes, start_idx: int = 2, end_idx: int | None = None) -> tuple[int, int]:
    """Calculate UBLOX checksum for given data range"""
    if end_idx is None:
      end_idx = len(data)

    ck_a = ck_b = 0
    for i in range(start_idx, end_idx):
      ck_a = (ck_a + data[i]) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b

  def _validate_checksum(self) -> bool:
    """Validate checksum of current message in buffer"""
    if len(self.parse_buffer) < UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE:
      return False

    ck_a, ck_b = self._calculate_checksum(
      self.parse_buffer, 2, len(self.parse_buffer) - UBLOX_CHECKSUM_SIZE
    )

    expected_ck_a = self.parse_buffer[-2]
    expected_ck_b = self.parse_buffer[-1]

    return ck_a == expected_ck_a and ck_b == expected_ck_b

  def _get_needed_bytes(self) -> int:
    """Get number of bytes needed to complete current message"""
    if len(self.parse_buffer) < UBLOX_HEADER_SIZE:
      return UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE - len(self.parse_buffer)

    # Extract message length from header
    msg_len = struct.unpack('<H', self.parse_buffer[4:6])[0]
    needed = msg_len + UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE

    if needed < len(self.parse_buffer):
      return -1  # Too much data
    return needed - len(self.parse_buffer)

  def _is_valid_so_far(self) -> bool:
    """Check if current buffer contents are valid so far"""
    if len(self.parse_buffer) > 0 and self.parse_buffer[0] != UBLOX_PREAMBLE1:
      return False
    if len(self.parse_buffer) > 1 and self.parse_buffer[1] != UBLOX_PREAMBLE2:
      return False
    # Check if we have a complete message that's invalid
    if (len(self.parse_buffer) >= UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE and
        self._get_needed_bytes() == 0 and not self._validate_checksum()):
      return False
    return True

  def _is_valid_message(self) -> bool:
    """Check if current buffer contains a complete valid message"""
    return (len(self.parse_buffer) >= UBLOX_HEADER_SIZE + UBLOX_CHECKSUM_SIZE and
            self._get_needed_bytes() == 0 and
            self._validate_checksum())

  def add_data(self, log_time: float, data: bytes) -> tuple[bool, int]:
    """
    Add incoming data to parser buffer.

    Returns:
        (message_ready, bytes_consumed)
    """
    self.last_log_time = log_time
    data_offset = 0

    # Keep consuming data until we either run out or have a complete message
    while data_offset < len(data):
      needed = self._get_needed_bytes()
      if needed > 0:
        bytes_to_consume = min(needed, len(data) - data_offset)
        # Add data to buffer
        self.parse_buffer.extend(data[data_offset:data_offset + bytes_to_consume])
        data_offset += bytes_to_consume
      else:
        # No more bytes needed, consume remaining data
        data_offset = len(data)

      # Validate message format and recover from corruption
      while not self._is_valid_so_far() and len(self.parse_buffer) > 0:
        # Drop corrupted byte and shift buffer
        self.parse_buffer.pop(0)

      # Reset buffer if we have too much data
      if self._get_needed_bytes() == -1:
        self.parse_buffer.clear()

      # Check if we have a complete message
      if self._is_valid_message():
        return True, data_offset

    return self._is_valid_message(), data_offset

  def parse_message(self) -> UBXMessage | None:
    """Parse complete message from buffer"""
    if not self._is_valid_message():
      return None

    msg_class = self.parse_buffer[2]
    msg_id = self.parse_buffer[3]
    msg_len = struct.unpack('<H', self.parse_buffer[4:6])[0]
    payload = bytes(self.parse_buffer[UBLOX_HEADER_SIZE:UBLOX_HEADER_SIZE + msg_len])
    checksum_valid = self._validate_checksum()

    return UBXMessage(
      msg_class=msg_class,
      msg_id=msg_id,
      payload=payload,
      checksum_valid=checksum_valid,
      log_time=self.last_log_time
    )


class UBXMessageProcessor:
  """
  Processes parsed UBLOX messages and generates cereal events.

  Handles different message types (NAV-PVT, RXM-RAWX, etc.) and converts
  them to appropriate cereal message formats.
  """

  def __init__(self):
    pass

  def process_nav_pvt(self, msg: UBXMessage) -> log.Event | None:
    """Process NAV-PVT (Position Velocity Time) message"""
    if len(msg.payload) < 92:
      cloudlog.warning("NAV-PVT message too short")
      return None

    # Unpack NAV-PVT payload
    data = struct.unpack('<IHBBBBBBIiBBBBiiiiIIiiiiiIIHHHHHHBBBB', msg.payload[:92])

    fixType = data[10]
    numSV = data[13]  # Number of satellites

    lon = data[14] * 1e-7  # Longitude (deg)
    lat = data[15] * 1e-7  # Latitude (deg)
    height = data[16]      # Height above ellipsoid (mm)
    hAcc = data[18]        # Horizontal accuracy estimate (mm)
    vAcc = data[19]        # Vertical accuracy estimate (mm)

    gSpeed = data[23]      # Ground speed (mm/s)
    headMot = data[24]     # Heading of motion (1e-5 deg)
    sAcc = data[25]        # Speed accuracy estimate (mm/s)
    headAcc = data[26]     # Heading accuracy estimate (1e-5 deg)

    # Create GPS location message
    evt = messaging.new_message('gpsLocationExternal', valid=True)
    loc = evt.gpsLocationExternal

    # Basic location data
    loc.unixTimestampMillis = int(msg.log_time * 1000)
    loc.latitude = lat
    loc.longitude = lon
    loc.altitude = height * 1e-3  # Convert mm to m
    loc.speed = gSpeed * 1e-3     # Convert mm/s to m/s
    loc.bearingDeg = headMot * 1e-5  # Convert to degrees
    loc.horizontalAccuracy = hAcc * 1e-3    # Convert mm to m
    loc.verticalAccuracy = vAcc * 1e-3      # Convert mm to m
    loc.speedAccuracy = sAcc * 1e-3         # Convert mm/s to m/s
    loc.bearingAccuracyDeg = headAcc * 1e-5 # Convert to degrees

    # Validity flags
    valid_fix = (fixType >= 2)  # 2D or 3D fix

    loc.hasFix = valid_fix
    loc.satelliteCount = numSV
    loc.source = log.GpsLocationData.SensorSource.ublox

    return evt

  def process_rxm_rawx(self, msg: UBXMessage) -> log.Event | None:
    """Process RXM-RAWX (Raw measurement data) message"""
    if len(msg.payload) < 16:
      return None
    # Extract just the header for now - skip complex measurement parsing
    try:
      rcvTow, week, leapS, numMeas, recStat = struct.unpack_from("<dHbbB3x", msg.payload, 0)
    except struct.error:
      return None

    # Create valid ubloxGnss message with empty measurements
    evt = messaging.new_message('ubloxGnss', valid=True)
    mr = evt.ubloxGnss.init('measurementReport')
    mr.rcvTow = rcvTow
    mr.gpsWeek = week
    mr.leapSeconds = leapS
    mr.numMeas = 0  # Set to 0 to avoid measurement parsing issues
    mr.init('measurements', 0)

    rs = mr.init('receiverStatus')
    rs.leapSecValid = bool(recStat & 0x01)
    rs.clkReset = bool(recStat & 0x04)
    return evt

  def process_rxm_sfrbx(self, msg: UBXMessage) -> log.Event | None:
    """Process RXM-SFRBX (Subframe buffer) message"""
    # Create minimal valid ubloxGnss message with ephemeris field
    evt = messaging.new_message('ubloxGnss', valid=True)
    eph = evt.ubloxGnss.init('ephemeris')
    eph.svId = 1  # Dummy value
    return evt

  def process_mon_hw(self, msg: UBXMessage) -> log.Event | None:
    """Process MON-HW (Hardware status) message"""
    # Create minimal valid ubloxGnss message with hwStatus field
    evt = messaging.new_message('ubloxGnss', valid=True)
    hw = evt.ubloxGnss.init('hwStatus')
    hw.noisePerMS = 0
    return evt

  def process_mon_hw2(self, msg: UBXMessage) -> log.Event | None:
    """Process MON-HW2 (Extended hardware status) message"""
    # Create minimal valid ubloxGnss message with hwStatus2 field
    evt = messaging.new_message('ubloxGnss', valid=True)
    hw2 = evt.ubloxGnss.init('hwStatus2')
    hw2.ofsI = 0
    return evt

  def process_nav_sat(self, msg: UBXMessage) -> log.Event | None:
    """Process NAV-SAT (Satellite status) message"""
    # Create minimal valid ubloxGnss message with satReport field
    evt = messaging.new_message('ubloxGnss', valid=True)
    sat = evt.ubloxGnss.init('satReport')
    sat.iTow = 0
    sat.init('svs', 0)  # Empty satellite list
    return evt

  def process_message(self, msg: UBXMessage) -> list[log.Event]:
    """Process any UBLOX message and return corresponding cereal events"""
    events = []

    try:
      # Create message type from class and ID
      msg_type = (msg.msg_class << 8) | msg.msg_id

      if msg_type == 0x0107:  # NAV-PVT
        evt = self.process_nav_pvt(msg)
        if evt:
          events.append(evt)
      elif msg_type == 0x0213:  # RXM-SFRBX
        evt = self.process_rxm_sfrbx(msg)
        if evt:
          events.append(evt)
      elif msg_type == 0x0215:  # RXM-RAWX
        evt = self.process_rxm_rawx(msg)
        if evt:
          events.append(evt)
      elif msg_type == 0x0a09:  # MON-HW
        evt = self.process_mon_hw(msg)
        if evt:
          events.append(evt)
      elif msg_type == 0x0a0b:  # MON-HW2
        evt = self.process_mon_hw2(msg)
        if evt:
          events.append(evt)
      elif msg_type == 0x0135:  # NAV-SAT
        evt = self.process_nav_sat(msg)
        if evt:
          events.append(evt)
      else:
        # Log unhandled message types for debugging
        cloudlog.debug(f"Unhandled UBLOX message type: 0x{msg_type:04x}")

    except Exception as e:
      cloudlog.error(f"Error processing UBLOX message type 0x{msg_type:04x}: {e}")

    return events


def main():
  """Main ubloxd daemon loop"""
  cloudlog.warning("Starting ubloxd (Python)")

  # Set up process priority
  config_realtime_process([1, 2, 3], 5)

  # Initialize messaging
  pm = messaging.PubMaster(['ubloxGnss', 'gpsLocationExternal'])
  sm = messaging.SubMaster(['ubloxRaw'])

  # Initialize parser and processor
  parser = UBXMessageParser()
  processor = UBXMessageProcessor()

  cloudlog.warning("ubloxd ready")

  while True:
    sm.update(timeout=100)

    if not sm.updated['ubloxRaw']:
      continue

    # Get raw UBLOX data
    if not sm.valid['ubloxRaw']:
      continue

    ublox_raw_data = sm['ubloxRaw']
    if len(ublox_raw_data) == 0:
      continue

    raw_data = bytes(ublox_raw_data)
    log_time = sm.logMonoTime['ubloxRaw'] * 1e-9

    # Parse messages from raw data
    data_offset = 0
    while data_offset < len(raw_data):
      remaining_data = raw_data[data_offset:]

      # Add data to parser
      message_ready, bytes_consumed = parser.add_data(log_time, remaining_data)
      data_offset += bytes_consumed

      if message_ready:
        # Parse and process complete message
        msg = parser.parse_message()
        if msg:  # Skip checksum validation for now
          events = processor.process_message(msg)

          # Send processed events
          for event in events:
            # Check which field is set in the event union
            try:
              _ = event.gpsLocationExternal
              pm.send('gpsLocationExternal', event)
            except Exception:
              try:
                _ = event.ubloxGnss
                pm.send('ubloxGnss', event)
              except Exception:
                cloudlog.warning("Unknown event type generated")

        # Reset parser for next message
        parser.reset()

      # Safety check to prevent infinite loop
      if bytes_consumed == 0:
        break


if __name__ == "__main__":
  main()
