#!/usr/bin/env python3
from asyncio import subprocess
import unittest
from unittest.mock import patch

from openpilot.system.timed import send_at_command, convert_modem_timezone_to_abbreviation, parse_modem_response, set_time

class TestModemScript(unittest.TestCase):

  @patch('subprocess.run')
  def test_send_at_command_success(self, mock_subprocess):
    mock_subprocess.return_value.stdout = 'Mock Response'
    result = send_at_command('AT+CLTS=2')
    self.assertEqual(result, 'Mock Response')

  @patch('subprocess.run')
  def test_send_at_command_failure(self, mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(returncode=1, cmd='AT+CLTS=2')
    result = send_at_command('AT+CLTS=2')
    self.assertIsNone(result)

  def test_convert_modem_timezone_to_abbreviation(self):
    timezone = "+32"
    abbreviation = convert_modem_timezone_to_abbreviation(timezone)
    self.assertEqual(abbreviation, "EST")

  def test_parse_modem_response_success(self):
    response = '2017/01/13,11:41:23+32,0'
    abbreviation, time_str = parse_modem_response(response)
    self.assertEqual(abbreviation, "EST")
    self.assertEqual(time_str, '2017/01/13 11:41:23 EST')

  def test_parse_modem_response_invalid_format(self):
    response = 'Invalid Response Format'
    abbreviation, time_str = parse_modem_response(response)
    self.assertIsNone(abbreviation)
    self.assertIsNone(time_str)

  @patch('subprocess.run')
  def test_set_system_time_success(self, mock_subprocess):
    set_time('2022/01/01 12:00:00+00')
    mock_subprocess.assert_called_with(['date', '-s', '2022/01/01 12:00:00+00'], check=True)

  @patch('subprocess.run')
  def test_set_system_time_failure(self, mock_subprocess):
    mock_subprocess.side_effect = subprocess.CalledProcessError(returncode=1, cmd='date -s')
    with self.assertLogs() as log:
      set_time('2022/01/01 12:00:00+00')
    self.assertIn('Error setting system time', log.output)
