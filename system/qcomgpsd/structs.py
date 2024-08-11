from struct import unpack_from, calcsize

LOG_GNSS_POSITION_REPORT = 0x1476
LOG_GNSS_GPS_MEASUREMENT_REPORT = 0x1477
LOG_GNSS_CLOCK_REPORT = 0x1478
LOG_GNSS_GLONASS_MEASUREMENT_REPORT = 0x1480
LOG_GNSS_BDS_MEASUREMENT_REPORT = 0x1756
LOG_GNSS_GAL_MEASUREMENT_REPORT = 0x1886

LOG_GNSS_OEMDRE_MEASUREMENT_REPORT = 0x14DE
LOG_GNSS_OEMDRE_SVPOLY_REPORT = 0x14E1

LOG_GNSS_ME_DPO_STATUS = 0x1838
LOG_GNSS_CD_DB_REPORT = 0x147B
LOG_GNSS_PRX_RF_HW_STATUS_REPORT = 0x147E
LOG_CGPS_SLOW_CLOCK_CLIB_REPORT = 0x1488
LOG_GNSS_CONFIGURATION_STATE = 0x1516

oemdre_measurement_report = """
  uint8_t version;
  uint8_t reason;
  uint8_t sv_count;
  uint8_t seq_num;
  uint8_t seq_max;
  uint16_t rf_loss;

  uint8_t system_rtc_valid;
  uint32_t f_count;
  uint32_t clock_resets;
  uint64_t system_rtc_time;

  uint8_t gps_leap_seconds;
  uint8_t gps_leap_seconds_uncertainty;
  float gps_to_glonass_time_bias_milliseconds;
  float gps_to_glonass_time_bias_milliseconds_uncertainty;

  uint16_t gps_week;
  uint32_t gps_milliseconds;
  uint32_t gps_time_bias;
  uint32_t gps_clock_time_uncertainty;
  uint8_t gps_clock_source;

  uint8_t glonass_clock_source;
  uint8_t glonass_year;
  uint16_t glonass_day;
  uint32_t glonass_milliseconds;
  float glonass_time_bias;
  float glonass_clock_time_uncertainty;

  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t frequency_source;

  uint32_t cdma_clock_info[5];

  uint8_t source;
"""

oemdre_svpoly_report = """
  uint8_t version;
  uint16_t sv_id;
  int8_t frequency_index;
  uint8_t flags;
  uint16_t iode;
  double t0;
  double xyz0[3];
  double xyzN[9];
  float other[4];
  float position_uncertainty;
  float iono_delay;
  float iono_dot;
  float sbas_iono_delay;
  float sbas_iono_dot;
  float tropo_delay;
  float elevation;
  float elevation_dot;
  float elevation_uncertainty;
  double velocity_coeff[12];
"""


oemdre_measurement_report_sv = """
  uint8_t sv_id;
  uint8_t unkn;
  int8_t glonass_frequency_index;
  uint32_t observation_state;
  uint8_t observations;
  uint8_t good_observations;
  uint8_t filter_stages;
  uint8_t predetect_interval;
  uint8_t cycle_slip_count;
  uint16_t postdetections;

  uint32_t measurement_status;
  uint32_t measurement_status2;

  uint16_t carrier_noise;
  uint16_t rf_loss;
  int16_t latency;

  float filtered_measurement_fraction;
  uint32_t filtered_measurement_integral;
  float filtered_time_uncertainty;
  float filtered_speed;
  float filtered_speed_uncertainty;

  float unfiltered_measurement_fraction;
  uint32_t unfiltered_measurement_integral;
  float unfiltered_time_uncertainty;
  float unfiltered_speed;
  float unfiltered_speed_uncertainty;

  uint8_t multipath_estimate_valid;
  uint32_t multipath_estimate;
  uint8_t direction_valid;
  float azimuth;
  float elevation;
  float doppler_acceleration;
  float fine_speed;
  float fine_speed_uncertainty;

  uint64_t carrier_phase;
  uint32_t f_count;

  uint16_t parity_error_count;
  uint8_t good_parity;
"""

glonass_measurement_report = """
  uint8_t version;
  uint32_t f_count;
  uint8_t glonass_cycle_number;
  uint16_t glonass_number_of_days;
  uint32_t milliseconds;
  float time_bias;
  float clock_time_uncertainty;
  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t sv_count;
"""

glonass_measurement_report_sv = """
  uint8_t sv_id;
  int8_t frequency_index;
  uint8_t observation_state; // SVObservationStates
  uint8_t observations;
  uint8_t good_observations;
  uint8_t hemming_error_count;
  uint8_t filter_stages;
  uint16_t carrier_noise;
  int16_t latency;
  uint8_t predetect_interval;
  uint16_t postdetections;
  uint32_t unfiltered_measurement_integral;
  float unfiltered_measurement_fraction;
  float unfiltered_time_uncertainty;
  float unfiltered_speed;
  float unfiltered_speed_uncertainty;
  uint32_t measurement_status;
  uint8_t misc_status;
  uint32_t multipath_estimate;
  float azimuth;
  float elevation;
  int32_t carrier_phase_cycles_integral;
  uint16_t carrier_phase_cycles_fraction;
  float fine_speed;
  float fine_speed_uncertainty;
  uint8_t cycle_slip_count;
  uint32_t pad;
"""

gps_measurement_report = """
  uint8_t version;
  uint32_t f_count;
  uint16_t week;
  uint32_t milliseconds;
  float time_bias;
  float clock_time_uncertainty;
  float clock_frequency_bias;
  float clock_frequency_uncertainty;
  uint8_t sv_count;
"""

gps_measurement_report_sv = """
  uint8_t sv_id;                              // SV PRN
  uint8_t observation_state;                  // SV Observation state
  uint8_t observations;                       // Count of all observation (both success and failure)
  uint8_t good_observations;                  // Count of Good observations
  uint16_t parity_error_count;                // Carrier to Code filtering N count
  uint8_t filter_stages;                      // Pre-Detection (Coherent) Interval (msecs)
  uint16_t carrier_noise;                     // CNo. Units of 0.1 dB
  int16_t latency;                            // Age of the measurement in msecs (+ve meas Meas precedes ref time)
  uint8_t predetect_interval;                 // Pre-Detection (Coherent) Interval (msecs)
  uint16_t postdetections;                    // Num Post-Detections (uints of PreInts
  uint32_t unfiltered_measurement_integral;   // Range of 0 through (WEEK_MSECS-1) [msecs]
  float unfiltered_measurement_fraction;      // Range of 0 through 0.99999 [msecs]
  float unfiltered_time_uncertainty;          // Time uncertainty (msec)
  float unfiltered_speed;                     // Speed estimate (meters/sec)
  float unfiltered_speed_uncertainty;         // Speed uncertainty estimate (meters/sec)
  uint32_t measurement_status;
  uint8_t misc_status;
  uint32_t multipath_estimate;
  float azimuth;                              // Azimuth (radians)
  float elevation;                            // Elevation (radians)
  int32_t carrier_phase_cycles_integral;
  uint16_t carrier_phase_cycles_fraction;
  float fine_speed;                           // Carrier phase derived speed
  float fine_speed_uncertainty;               // Carrier phase derived speed UNC
  uint8_t cycle_slip_count;                   // Increments when a CSlip is detected
  uint32_t pad;
"""

position_report = """
  uint8       u_Version;                /* Version number of DM log */
  uint32      q_Fcount;                 /* Local millisecond counter */
  uint8       u_PosSource;              /* Source of position information */ /*  0: None 1: Weighted least-squares 2: Kalman filter 3: Externally injected 4: Internal database    */
  uint32      q_Reserved1;              /* Reserved memory field */
  uint16      w_PosVelFlag;             /* Position velocity bit field: (see DM log 0x1476 documentation) */
  uint32      q_PosVelFlag2;            /* Position velocity 2 bit field: (see DM log 0x1476 documentation) */
  uint8       u_FailureCode;            /* Failure code: (see DM log 0x1476 documentation) */
  uint16      w_FixEvents;              /* Fix events bit field: (see DM log 0x1476 documentation) */
  uint32 _fake_align_week_number;
  uint16      w_GpsWeekNumber;          /* GPS week number of position */
  uint32      q_GpsFixTimeMs;           /* GPS fix time of week of in milliseconds */
  uint8       u_GloNumFourYear;         /* Number of Glonass four year cycles */
  uint16      w_GloNumDaysInFourYear;   /* Glonass calendar day in four year cycle */
  uint32      q_GloFixTimeMs;           /* Glonass fix time of day in milliseconds */
  uint32      q_PosCount;               /* Integer count of the number of unique positions reported */
  uint64      t_DblFinalPosLatLon[2];   /* Final latitude and longitude of position in radians */
  uint32      q_FltFinalPosAlt;         /* Final height-above-ellipsoid altitude of position */
  uint32      q_FltHeadingRad;          /* User heading in radians */
  uint32      q_FltHeadingUncRad;       /* User heading uncertainty in radians */
  uint32      q_FltVelEnuMps[3];        /* User velocity in east, north, up coordinate frame. In meters per second. */
  uint32      q_FltVelSigmaMps[3];      /* Gaussian 1-sigma value for east, north, up components of user velocity */
  uint32      q_FltClockBiasMeters;     /* Receiver clock bias in meters */
  uint32      q_FltClockBiasSigmaMeters;  /* Gaussian 1-sigma value for receiver clock bias in meters */
  uint32      q_FltGGTBMeters;          /* GPS to Glonass time bias in meters */
  uint32      q_FltGGTBSigmaMeters;     /* Gaussian 1-sigma value for GPS to Glonass time bias uncertainty in meters */
  uint32      q_FltGBTBMeters;          /* GPS to BeiDou time bias in meters */
  uint32      q_FltGBTBSigmaMeters;     /* Gaussian 1-sigma value for GPS to BeiDou time bias uncertainty in meters */
  uint32      q_FltBGTBMeters;          /* BeiDou to Glonass time bias in meters */
  uint32      q_FltBGTBSigmaMeters;     /* Gaussian 1-sigma value for BeiDou to Glonass time bias uncertainty in meters */
  uint32      q_FltFiltGGTBMeters;      /* Filtered GPS to Glonass time bias in meters */
  uint32      q_FltFiltGGTBSigmaMeters; /* Filtered Gaussian 1-sigma value for GPS to Glonass time bias uncertainty in meters */
  uint32      q_FltFiltGBTBMeters;      /* Filtered GPS to BeiDou time bias in meters */
  uint32      q_FltFiltGBTBSigmaMeters; /* Filtered Gaussian 1-sigma value for GPS to BeiDou time bias uncertainty in meters */
  uint32      q_FltFiltBGTBMeters;      /* Filtered BeiDou to Glonass time bias in meters */
  uint32      q_FltFiltBGTBSigmaMeters; /* Filtered Gaussian 1-sigma value for BeiDou to Glonass time bias uncertainty in meters */
  uint32      q_FltSftOffsetSec;        /* SFT offset as computed by WLS in seconds */
  uint32      q_FltSftOffsetSigmaSec;   /* Gaussian 1-sigma value for SFT offset in seconds */
  uint32      q_FltClockDriftMps;       /* Clock drift (clock frequency bias) in meters per second */
  uint32      q_FltClockDriftSigmaMps;  /* Gaussian 1-sigma value for clock drift in meters per second */
  uint32      q_FltFilteredAlt;         /* Filtered height-above-ellipsoid altitude in meters as computed by WLS */
  uint32      q_FltFilteredAltSigma;    /* Gaussian 1-sigma value for filtered height-above-ellipsoid altitude in meters */
  uint32      q_FltRawAlt;              /* Raw height-above-ellipsoid altitude in meters as computed by WLS */
  uint32      q_FltRawAltSigma;         /* Gaussian 1-sigma value for raw height-above-ellipsoid altitude in meters */
  uint32   align_Flt[14];
  uint32      q_FltPdop;                /* 3D position dilution of precision as computed from the unweighted
  uint32      q_FltHdop;                /* Horizontal position dilution of precision as computed from the unweighted least-squares covariance matrix */
  uint32      q_FltVdop;                /* Vertical position dilution of precision as computed from the unweighted least-squares covariance matrix */
  uint8       u_EllipseConfidence;      /* Statistical measure of the confidence (percentage) associated with the uncertainty ellipse values */
  uint32      q_FltEllipseAngle;        /* Angle of semimajor axis with respect to true North, with increasing angles moving clockwise from North. In units of degrees. */
  uint32      q_FltEllipseSemimajorAxis;  /* Semimajor axis of final horizontal position uncertainty error ellipse.  In units of meters. */
  uint32      q_FltEllipseSemiminorAxis;  /* Semiminor axis of final horizontal position uncertainty error ellipse.  In units of meters. */
  uint32      q_FltPosSigmaVertical;    /* Gaussian 1-sigma value for final position height-above-ellipsoid altitude in meters */
  uint8       u_HorizontalReliability;  /* Horizontal position reliability 0: Not set 1: Very Low 2: Low 3: Medium 4: High    */
  uint8       u_VerticalReliability;    /* Vertical position reliability */
  uint16      w_Reserved2;              /* Reserved memory field */
  uint32      q_FltGnssHeadingRad;      /* User heading in radians derived from GNSS only solution  */
  uint32      q_FltGnssHeadingUncRad;   /* User heading uncertainty in radians derived from GNSS only solution  */
  uint32      q_SensorDataUsageMask;    /* Denotes which additional sensor data were used to compute this position fix.  BIT[0] 0x00000001 <96> Accelerometer BIT[1] 0x00000002 <96> Gyro 0x0000FFFC - Reserved A bit set to 1 indicates that certain fields as defined by the SENSOR_AIDING_MASK were aided with sensor data*/
  uint32      q_SensorAidMask;         /* Denotes which component of the position report was assisted with additional sensors defined in SENSOR_DATA_USAGE_MASK BIT[0] 0x00000001 <96> Heading aided with sensor data BIT[1] 0x00000002 <96> Speed aided with sensor data BIT[2] 0x00000004 <96> Position aided with sensor data BIT[3] 0x00000008 <96> Velocity aided with sensor data 0xFFFFFFF0 <96> Reserved */
  uint8       u_NumGpsSvsUsed;          /* The number of GPS SVs used in the fix */
  uint8       u_TotalGpsSvs;            /* Total number of GPS SVs detected by searcher, including ones not used in position calculation */
  uint8       u_NumGloSvsUsed;          /* The number of Glonass SVs used in the fix */
  uint8       u_TotalGloSvs;            /* Total number of Glonass SVs detected by searcher, including ones not used in position calculation */
  uint8       u_NumBdsSvsUsed;          /* The number of BeiDou SVs used in the fix */
  uint8       u_TotalBdsSvs;            /* Total number of BeiDou SVs detected by searcher, including ones not used in position calculation */
""" # noqa: E501

def name_to_camelcase(nam):
  ret = []
  i = 0
  while i < len(nam):
    if nam[i] == "_":
      ret.append(nam[i+1].upper())
      i += 2
    else:
      ret.append(nam[i])
      i += 1
  return ''.join(ret)

def parse_struct(ss):
  st = "<"
  nams = []
  for l in ss.strip().split("\n"):
    if len(l.strip()) == 0:
      continue
    typ, nam = l.split(";")[0].split()
    #print(typ, nam)
    if typ == "float" or '_Flt' in nam:
      st += "f"
    elif typ == "double" or '_Dbl' in nam:
      st += "d"
    elif typ in ["uint8", "uint8_t"]:
      st += "B"
    elif typ in ["int8", "int8_t"]:
      st += "b"
    elif typ in ["uint32", "uint32_t"]:
      st += "I"
    elif typ in ["int32", "int32_t"]:
      st += "i"
    elif typ in ["uint16", "uint16_t"]:
      st += "H"
    elif typ in ["int16", "int16_t"]:
      st += "h"
    elif typ in ["uint64", "uint64_t"]:
      st += "Q"
    else:
      raise RuntimeError(f"unknown type {typ}")
    if '[' in nam:
      cnt = int(nam.split("[")[1].split("]")[0])
      st += st[-1]*(cnt-1)
      for i in range(cnt):
        nams.append("%s[%d]" % (nam.split("[")[0], i))
    else:
      nams.append(nam)
  return st, nams

def dict_unpacker(ss, camelcase = False):
  st, nams = parse_struct(ss)
  if camelcase:
    nams = [name_to_camelcase(x) for x in nams]
  sz = calcsize(st)
  return lambda x: dict(zip(nams, unpack_from(st, x), strict=True)), sz

def relist(dat):
  list_keys = set()
  for key in dat.keys():
    if '[' in key:
      list_keys.add(key.split('[')[0])
  list_dict = {}
  for list_key in list_keys:
    list_dict[list_key] = []
    i = 0
    while True:
      key = list_key + f'[{i}]'
      if key not in dat:
        break
      list_dict[list_key].append(dat[key])
      del dat[key]
      i += 1
  return {**dat, **list_dict}
