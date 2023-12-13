#!/usr/bin/env python3
import subprocess
from openpilot.common.swaglog import cloudlog
from timezoned import main as timezoned

MODEM_DEVICE = 0

def send_at_command(command):
  """Send an AT command to the modem and return the response (str or None)."""
  try:
    result = subprocess.run(['mmcli', '-m', str(MODEM_DEVICE), '--command', command], capture_output=True, text=True, check=True)
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    cloudlog.error(f"Error sending AT command: {e}")
    return None

def convert_modem_timezone_to_abbreviation(timezone):
  """Convert modem-style timezone (str) to its abbreviation (str or None)."""
  try:
    sign = timezone[0]
    offset_quarters = int(timezone[1:])

    # Calculate the offset in minutes from UTC
    offset_minutes = offset_quarters * 15

    # Define a mapping of offsets to timezone abbreviations for U.S. timezones
    us_timezone_mapping = {
      "-300": "EST",
      "-360": "CST",
      "-420": "MST",
      "-480": "PST",
      "-540": "AKST",
      "-600": "HST"
    }

    offset_minutes *= -1 if sign == "-" else 1
    return us_timezone_mapping.get(str(offset_minutes), "Unknown")


  except Exception as e:
    cloudlog.error(f"Error converting modem timezone: {e}")

  return None

def parse_modem_response(response):
  """Parse modem response (str) and extract timezone abbreviation and time string (tuple or None)."""
  try:
    parts = response.split(',')
    if len(parts) >= 3:
      timezone = parts[2].strip('"')
      timezone_abbreviation = convert_modem_timezone_to_abbreviation(timezone)
      time_str = f"{parts[0]} {parts[1]} {timezone_abbreviation}"
      
      return timezone_abbreviation, time_str
    else:
      raise ValueError("Invalid modem response format")
  except Exception as e:
    cloudlog.error(f"Error parsing modem response: {e}")
    return None, None

def set_time(time_str):
  """Set system time using a time string in the format 'YYYY/MM/dd hh:mm:ss±zz'."""
  try:
    subprocess.run(['date', '-s', time_str], check=True)
  except subprocess.CalledProcessError as e:
    cloudlog.error(f"Error setting system time: {e}")
    
def get_time_from_modem():
  """Retrieve modem's time and timezone, returning timezone abbreviation and time string (tuple or None)."""
  response = send_at_command('AT+CLTS=2')
  if response:
    return parse_modem_response(response)
  return None, None

def main():
  timezone, modem_time = get_time_from_modem()
  if timezone and modem_time:
    set_time(modem_time)
    timezoned(timezone=timezone)
  else:
    timezoned()

if __name__ == "__main__":
  main()
