#!/usr/bin/env python3
import unittest
from openpilot.system.timed import send_at_command, parse_modem_response, get_time_and_timezone_from_modem

class TestTimed(unittest.TestCase):

    def test_send_at_command(self):
        # Testing with a known AT command
        response = send_at_command('AT+QLTS=1')
        self.assertIsNotNone(response)

    def test_parse_modem_response(self):
        # Test with a valid response format
        response = '+CTZE: "+01",0,"2023/03/15,12:34:56"'
        timezone, time_str = parse_modem_response(response)
        self.assertEqual(timezone, '+01')
        self.assertEqual(time_str, '2023/03/15,12:34:56')

        # Test with an invalid response format
        response = 'Invalid response'
        timezone, time_str = parse_modem_response(response)
        self.assertIsNone(timezone)
        self.assertIsNone(time_str)

    def test_get_time_and_timezone_from_modem(self):
        timezone, modem_time = get_time_and_timezone_from_modem()

if __name__ == '__main__':
    unittest.main()
