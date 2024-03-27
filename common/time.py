import datetime

MIN_DATE = datetime.datetime(year=2024, month=1, day=28)

def system_time_valid():
  return datetime.datetime.now() > MIN_DATE
