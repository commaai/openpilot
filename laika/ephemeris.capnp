@0xb3ca6d2462778bb1;

struct Ephemeris {
  # This is according to the rinex (2?) format
  svId @0 :UInt16;
  year @1 :UInt16;
  month @2 :UInt16;
  day @3 :UInt16;
  hour @4 :UInt16;
  minute @5 :UInt16;
  second @6 :Float32;
  af0 @7 :Float64;
  af1 @8 :Float64;
  af2 @9 :Float64;

  iode @10 :Float64;
  crs @11 :Float64;
  deltaN @12 :Float64;
  m0 @13 :Float64;

  cuc @14 :Float64;
  ecc @15 :Float64;
  cus @16 :Float64;
  a @17 :Float64; # note that this is not the root!!

  toe @18 :Float64;
  cic @19 :Float64;
  omega0 @20 :Float64;
  cis @21 :Float64;

  i0 @22 :Float64;
  crc @23 :Float64;
  omega @24 :Float64;
  omegaDot @25 :Float64;

  iDot @26 :Float64;
  codesL2 @27 :Float64;
  gpsWeekDEPRECATED @28 :Float64;
  l2 @29 :Float64;

  svAcc @30 :Float64;
  svHealth @31 :Float64;
  tgd @32 :Float64;
  iodc @33 :Float64;

  transmissionTime @34 :Float64;
  fitInterval @35 :Float64;

  toc @36 :Float64;

  ionoCoeffsValid @37 :Bool;
  ionoAlpha @38 :List(Float64);
  ionoBeta @39 :List(Float64);

  towCount @40 :UInt32;
  toeWeek @41 :UInt16;
  tocWeek @42 :UInt16;
}

struct GlonassEphemeris {
  svId @0 :UInt16;
  year @1 :UInt16;
  dayInYear @2 :UInt16;
  hour @3 :UInt16;
  minute @4 :UInt16;
  second @5 :Float32;

  x @6 :Float64;
  xVel @7 :Float64;
  xAccel @8 :Float64;
  y @9 :Float64;
  yVel @10 :Float64;
  yAccel @11 :Float64;
  z @12 :Float64;
  zVel @13 :Float64;
  zAccel @14 :Float64;

  svType @15 :UInt8;
  svURA @16 :Float32;
  age @17 :UInt8;

  svHealth @18 :UInt8;
  tkDEPRECATED @19 :UInt16;
  tb @20 :UInt16;

  tauN @21 :Float64;
  deltaTauN @22 :Float64;
  gammaN @23 :Float64;

  p1 @24 :UInt8;
  p2 @25 :UInt8;
  p3 @26 :UInt8;
  p4 @27 :UInt8;

  freqNumDEPRECATED @28 :UInt32;

  n4 @29 :UInt8;
  nt @30 :UInt16;
  freqNum @31 :Int16;
  tkSeconds @32 :UInt32;
}

struct EphemerisCache {
  gpsEphemerides @0 :List(Ephemeris);
  glonassEphemerides @1 :List(GlonassEphemeris);
}