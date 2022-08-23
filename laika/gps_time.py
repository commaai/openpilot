import datetime


def datetime_to_tow(t):
    """
    Convert a Python datetime object to GPS Week and Time Of Week.
    Does *not* convert from UTC to GPST.
    Fractional seconds are supported.

    Parameters
    ----------
    t : datetime
      A time to be converted, on the GPST timescale.
    mod1024 : bool, optional
      If True (default), the week number will be output in 10-bit form.

    Returns
    -------
    week, tow : tuple (int, float)
      The GPS week number and time-of-week.
    """
    # DateTime to GPS week and TOW
    wk_ref = datetime.datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - datetime.timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow


def tow_to_datetime(tow, week):
    """
    Convert a GPS Week and Time Of Week to Python datetime object.
    Does *not* convert from GPST to UTC.
    Fractional seconds are supported.

    Parameters
    ----------
    tow : time of week in seconds

    weeks : gps week


    Returns
    -------
    t : datetime
      Python datetime
    """
    #  GPS week and TOW to DateTime
    t = datetime.datetime(1980, 1, 6, 0, 0, 0, 0, None)
    t += datetime.timedelta(seconds=tow)
    t += datetime.timedelta(weeks=week)
    return t


def get_leap_seconds(time):
  # TODO use library for this
  if time <= GPSTime.from_datetime(datetime.datetime(2006, 1, 1)):
    raise ValueError("Don't know how many leap seconds to use before 2006")
  elif time <= GPSTime.from_datetime(datetime.datetime(2009, 1, 1)):
    return 14
  elif time <= GPSTime.from_datetime(datetime.datetime(2012, 7, 1)):
    return 15
  elif time <= GPSTime.from_datetime(datetime.datetime(2015, 7, 1)):
    return 16
  # TODO is this correct?
  elif time <= GPSTime.from_datetime(datetime.datetime(2017, 7, 1)):
    return 17
  else:
    return 18


def gpst_to_utc(t_gpst):
    t_utc = t_gpst - get_leap_seconds(t_gpst)
    if utc_to_gpst(t_utc) - t_gpst != 0:
      return t_utc + 1
    else:
      return t_utc


def utc_to_gpst(t_utc):
    t_gpst = t_utc + get_leap_seconds(t_utc)
    return t_gpst


class GPSTime:
  """
  GPS time class to add and subtract [week, tow]
  """
  def __init__(self, week, tow):
    self.week = week
    self.tow = tow
    self.seconds_in_week = 604800

  @classmethod
  def from_datetime(cls, datetime):
    week, tow = datetime_to_tow(datetime)
    return cls(week, tow)

  @classmethod
  def from_glonass(cls, cycle, days, tow):
    # https://en.wikipedia.org/wiki/GLONASS
    # Day number (1 to 1461) within a four-year interval
    # starting on 1 January of the last leap year
    t = datetime.datetime(1992, 1, 1, 0, 0, 0, 0, None)
    t += datetime.timedelta(days=cycle*(365*4+1)+(days-1))
    # according to Moscow decree time.
    t -= datetime.timedelta(hours=3)
    t += datetime.timedelta(seconds=tow)
    ret = cls.from_datetime(t)
    return utc_to_gpst(ret)

  @classmethod
  def from_meas(cls, meas):
    return cls(meas[1], meas[2])

  def __sub__(self, other):
    if isinstance(other, type(self)):
      return (self.week - other.week)*self.seconds_in_week + self.tow - other.tow
    elif isinstance(other, float) or isinstance(other, int):
      new_week = self.week
      new_tow = self.tow - other
      while new_tow < 0:
        new_tow += self.seconds_in_week
        new_week -= 1
      return GPSTime(new_week, new_tow)
    raise NotImplementedError(f"subtracting {other} from {self}")

  def __add__(self, other):
    if isinstance(other, float) or isinstance(other, int):
      new_week = self.week
      new_tow = self.tow + other
      while new_tow >= self.seconds_in_week:
        new_tow -= self.seconds_in_week
        new_week += 1
      return GPSTime(new_week, new_tow)
    raise NotImplementedError(f"adding {other} from {self}")

  def __lt__(self, other):
    return self - other < 0

  def __gt__(self, other):
    return self - other > 0

  def __le__(self, other):
    return self - other <= 0

  def __ge__(self, other):
    return self - other >= 0

  def __eq__(self, other):
    return self - other == 0

  def as_datetime(self):
    return tow_to_datetime(self.tow, self.week)

  def as_unix_timestamp(self):
    return (gpst_to_utc(self).as_datetime() - datetime.datetime(1970, 1, 1)).total_seconds()

  @property
  def day(self):
    return int(self.tow/(24*3600))

  def __repr__(self):
    return f"GPSTime(week={self.week}, tow={self.tow})"


class TimeSyncer:
  """
  Converts logmonotime to gps_time and vice versa
  """
  def __init__(self, mono_time, gps_time):
    self.ref_mono_time = mono_time
    self.ref_gps_time = gps_time

  @classmethod
  def from_datetime(cls, datetime):
    week, tow = datetime_to_tow(datetime)
    return cls(week, tow)

  @classmethod
  def from_logs(cls, raw_qcom_measurement_report, clocks):
    #TODO
    #return cls(week, mono_time, gps_time)
    return None

  def mono2gps(self, mono_time):
    return self.ref_gps_time + mono_time - self.ref_mono_time

  def gps2mono(self, gps_time):
    return gps_time - self.ref_gps_time + self.ref_mono_time

  def __str__(self):
      return f"Reference mono time: {self.ref_mono_time} \n  Reference gps time: {self.ref_gps_time}"
