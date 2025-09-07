#!/usr/bin/env python3
"""
Test suite for the Python ubloxd implementation.

Tests UBLOX message parsing, processing, and cereal message generation.
"""

import struct

from openpilot.system.ubloxd.ubloxd import (
  UBXMessageParser, UBXMessageProcessor, UBXMessage,
  UBLOX_PREAMBLE1, UBLOX_PREAMBLE2, UBXClass, UBXMessageID
)


class TestUBXMessageParser:
  """Test UBLOX message parser functionality"""

  def setup_method(self):
    self.parser = UBXMessageParser()

  def test_calculate_checksum(self):
    """Test UBLOX checksum calculation"""
    # Test message: UBX header + simple payload
    msg = bytearray([0xb5, 0x62, 0x01, 0x07, 0x04, 0x00, 0x01, 0x02, 0x03, 0x04])

    ck_a, ck_b = self.parser._calculate_checksum(msg, 2, 10)
    assert isinstance(ck_a, int)
    assert isinstance(ck_b, int)
    assert 0 <= ck_a <= 255
    assert 0 <= ck_b <= 255

  def test_simple_message_parsing(self):
    """Test parsing a complete simple message"""
    # Create a simple test message: UBX-NAV-PVT stub
    payload = b'\x00' * 92  # Minimum NAV-PVT payload size
    msg_data = struct.pack('<BBBBH', UBLOX_PREAMBLE1, UBLOX_PREAMBLE2,
                          UBXClass.NAV, UBXMessageID.NAV.PVT, len(payload))
    msg_data += payload

    # Add checksum
    ck_a = ck_b = 0
    for byte in msg_data[2:]:
      ck_a = (ck_a + byte) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    msg_data += bytes([ck_a, ck_b])

    # Parse the message
    complete, consumed = self.parser.add_data(123.456, msg_data)
    assert complete
    assert consumed == len(msg_data)

    # Extract the parsed message
    parsed_msg = self.parser.parse_message()
    assert parsed_msg is not None
    assert parsed_msg.msg_class == UBXClass.NAV
    assert parsed_msg.msg_id == UBXMessageID.NAV.PVT
    assert len(parsed_msg.payload) == 92
    assert parsed_msg.checksum_valid
    assert parsed_msg.log_time == 123.456

  def test_incremental_parsing(self):
    """Test parsing message received in multiple chunks"""
    # Create test message
    payload = b'\x11\x22\x33\x44'
    msg_data = struct.pack('<BBBBH', UBLOX_PREAMBLE1, UBLOX_PREAMBLE2,
                          0x01, 0x02, len(payload))
    msg_data += payload

    # Add checksum
    ck_a = ck_b = 0
    for byte in msg_data[2:]:
      ck_a = (ck_a + byte) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    msg_data += bytes([ck_a, ck_b])

    # Send message in chunks
    chunk1 = msg_data[:4]  # Header partial
    chunk2 = msg_data[4:8]  # Header complete + payload start
    chunk3 = msg_data[8:]   # Rest of payload + checksum

    # Process chunks
    complete1, consumed1 = self.parser.add_data(100.0, chunk1)
    assert not complete1
    assert consumed1 == len(chunk1)

    complete2, consumed2 = self.parser.add_data(100.0, chunk2)
    assert not complete2
    assert consumed2 == len(chunk2)

    complete3, consumed3 = self.parser.add_data(100.0, chunk3)
    assert complete3
    assert consumed3 == len(chunk3)

    # Verify parsed message
    parsed_msg = self.parser.parse_message()
    assert parsed_msg is not None
    assert parsed_msg.payload == payload

  def test_corrupted_data_recovery(self):
    """Test recovery from corrupted message data"""
    # Send some garbage data followed by valid message
    garbage = b'\x00\x11\x22\x33\x44\x55'

    # Valid message
    payload = b'\xaa\xbb'
    msg_data = struct.pack('<BBBBH', UBLOX_PREAMBLE1, UBLOX_PREAMBLE2,
                          0x01, 0x02, len(payload))
    msg_data += payload

    # Add checksum
    ck_a = ck_b = 0
    for byte in msg_data[2:]:
      ck_a = (ck_a + byte) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    msg_data += bytes([ck_a, ck_b])

    combined_data = garbage + msg_data

    # Parser should skip garbage and find valid message
    complete, consumed = self.parser.add_data(100.0, combined_data)
    assert complete

    parsed_msg = self.parser.parse_message()
    assert parsed_msg is not None
    assert parsed_msg.payload == payload

  def test_reset(self):
    """Test parser reset functionality"""
    # Add some data
    test_data = b'\xb5\x62\x01\x02'
    self.parser.add_data(100.0, test_data)
    assert len(self.parser.parse_buffer) > 0

    # Reset should clear buffer
    self.parser.reset()
    assert len(self.parser.parse_buffer) == 0


class TestUBXMessageProcessor:
  """Test UBLOX message processing and cereal event generation"""

  def setup_method(self):
    self.processor = UBXMessageProcessor()

  def test_process_nav_pvt(self, mocker):
    """Test processing of NAV-PVT messages"""
    # Mock cereal message
    mock_event = mocker.MagicMock()
    mock_location = mocker.MagicMock()
    mock_event.gpsLocationExternal = mock_location
    mock_new_message = mocker.patch('openpilot.system.ubloxd.ubloxd.messaging.new_message',
                                   return_value=mock_event)

    # Create NAV-PVT payload with test data
    fixType = 3   # 3D fix
    numSV = 12    # 12 satellites

    lon = int(-122.419416 * 1e7)  # San Francisco longitude
    lat = int(37.774929 * 1e7)    # San Francisco latitude
    height = 100000  # 100m above ellipsoid (in mm)
    hAcc = 5000      # 5m horizontal accuracy (in mm)
    vAcc = 10000     # 10m vertical accuracy (in mm)

    gSpeed = 2236    # ground speed ~2.24 m/s (in mm/s)
    headMot = 4500000  # 45 degrees heading (in 1e-5 deg)
    sAcc = 500       # 0.5 m/s speed accuracy (in mm/s)
    headAcc = 1800000  # 18 degree heading accuracy (in 1e-5 deg)

    # Pack the payload (first 92 bytes of NAV-PVT)
    payload = struct.pack('<IHBBBBBBIiBBBBiiiiIIiiiiiIIHHHHHHBBBB',
                         123000,    # iTOW
                         2024, 1, 15, 12, 30, 45,  # year, month, day, hour, min, sec
                         0x07,      # valid flags
                         1000,      # tAcc
                         0,         # nano
                         fixType,   # fixType (3D)
                         0x01,      # flags
                         0x00,      # flags2
                         numSV,     # numSV
                         lon, lat, height, 5000, hAcc, vAcc,
                         0, 0, 0, gSpeed, headMot, sAcc, headAcc, 100,
                         0, 0, 0, 0, 0, 0, 0, 0, 0)  # Reserved fields

    # Create UBX message
    msg = UBXMessage(
      msg_class=UBXClass.NAV,
      msg_id=UBXMessageID.NAV.PVT,
      payload=payload,
      checksum_valid=True,
      log_time=100.0
    )

    # Process message
    events = self.processor.process_message(msg)

    # Verify results
    assert len(events) == 1
    mock_new_message.assert_called_once_with('gpsLocationExternal', valid=True)

    # Check that location fields were set correctly
    assert mock_location.unixTimestampMillis == 100000
    assert abs(mock_location.latitude - 37.774929) < 1e-6
    assert abs(mock_location.longitude - (-122.419416)) < 1e-6
    assert abs(mock_location.altitude - 100.0) < 0.01
    assert abs(mock_location.speed - 2.236) < 0.001
    assert abs(mock_location.bearingDeg - 45.0) < 0.1
    assert abs(mock_location.horizontalAccuracy - 5.0) < 0.01
    assert mock_location.hasFix
    assert mock_location.satelliteCount == 12

  def test_process_invalid_message(self):
    """Test handling of invalid/corrupted messages"""
    # Create message with invalid payload (too short for NAV-PVT)
    msg = UBXMessage(
      msg_class=UBXClass.NAV,
      msg_id=UBXMessageID.NAV.PVT,
      payload=b'\x00\x01\x02',  # Too short
      checksum_valid=True,
      log_time=100.0
    )

    # Should handle gracefully without crashing
    events = self.processor.process_message(msg)
    assert len(events) == 0

  def test_process_unknown_message(self):
    """Test handling of unknown message types"""
    msg = UBXMessage(
      msg_class=0xFF,  # Unknown class
      msg_id=0xFF,     # Unknown ID
      payload=b'\x00\x01\x02\x03',
      checksum_valid=True,
      log_time=100.0
    )

    # Should handle gracefully
    events = self.processor.process_message(msg)
    assert len(events) == 0


class TestIntegration:
  """Integration tests for complete parsing pipeline"""

  def test_end_to_end_nav_pvt(self, mocker):
    """Test complete NAV-PVT message parsing and processing"""
    parser = UBXMessageParser()
    processor = UBXMessageProcessor()

    # Create realistic NAV-PVT message
    payload = struct.pack('<IHBBBBBBIiBBBBiiiiIIiiiiiIIHHHHHHBBBB',
                         123000,    # iTOW
                         2024, 1, 15, 12, 30, 45,  # year, month, day, hour, min, sec
                         0x07,      # valid flags
                         1000,      # tAcc
                         0,         # nano
                         3,         # fixType (3D)
                         0x01,      # flags
                         0x00,      # flags2
                         10,        # numSV
                         int(-122.0 * 1e7),  # lon
                         int(37.0 * 1e7),    # lat
                         10000,     # height (10m)
                         5000,      # hMSL (5m)
                         2000,      # hAcc (2m)
                         3000,      # vAcc (3m)
                         0, 0, 0,   # velN, velE, velD
                         0,         # gSpeed
                         0,         # headMot
                         0,         # sAcc
                         0,         # headAcc
                         100,       # pDOP
                         0, 0, 0, 0, 0, 0, 0, 0, 0)  # reserved

    # Build complete message with header and checksum
    msg_data = struct.pack('<BBBBH', UBLOX_PREAMBLE1, UBLOX_PREAMBLE2,
                          UBXClass.NAV, UBXMessageID.NAV.PVT, len(payload))
    msg_data += payload

    # Calculate and add checksum
    ck_a = ck_b = 0
    for byte in msg_data[2:]:
      ck_a = (ck_a + byte) & 0xFF
      ck_b = (ck_b + ck_a) & 0xFF
    msg_data += bytes([ck_a, ck_b])

    # Parse message
    mock_event = mocker.MagicMock()
    mock_location = mocker.MagicMock()
    mock_event.gpsLocationExternal = mock_location
    mock_new_message = mocker.patch('openpilot.system.ubloxd.ubloxd.messaging.new_message',
                                   return_value=mock_event)

    complete, consumed = parser.add_data(200.0, msg_data)
    assert complete

    parsed_msg = parser.parse_message()
    assert parsed_msg is not None
    assert parsed_msg.checksum_valid

    events = processor.process_message(parsed_msg)
    assert len(events) == 1

    # Verify location was processed correctly
    mock_new_message.assert_called_once_with('gpsLocationExternal', valid=True)
    assert mock_location.unixTimestampMillis == 200000
    assert abs(mock_location.latitude - 37.0) < 0.1
    assert abs(mock_location.longitude - (-122.0)) < 0.1
