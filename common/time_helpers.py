import datetime
from pathlib import Path

MIN_DATE = datetime.datetime(year=2025, month=2, day=21)

def min_date():
  # on systemd systems, the default time is the systemd build time
  systemd_path = Path("/lib/systemd/systemd")
  if systemd_path.exists():
    d = datetime.datetime.fromtimestamp(systemd_path.stat().st_mtime)
    return max(MIN_DATE, d + datetime.timedelta(days=1))
  return MIN_DATE

def system_time_valid():
  return datetime.datetime.now() > min_date()
