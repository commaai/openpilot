import datetime

MIN_DATE = datetime.datetime(year=2023, month=6, day=1)

def valid_datetime(d: datetime.datetime):
  return d > MIN_DATE