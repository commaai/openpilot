import datetime

MIN_DATE = datetime.datetime(year=2023, month=6, day=1)

def system_time_valid():
  return datetime.datetime.now() > MIN_DATE