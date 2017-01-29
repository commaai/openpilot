using Cxx = import "c++.capnp";
$Cxx.namespace("cereal");

using Car = import "car.capnp";

@0xf3b1f17e25a4285b;

const logVersion :Int32 = 1;

struct InitData {
  kernelArgs @0 :List(Text);
  gctx @1 :Text;
  dongleId @2 :Text;
}

struct FrameData {
  frameId @0 :UInt32;
  encodeId @1 :UInt32; # DEPRECATED
  timestampEof @2 :UInt64;
  frameLength @3 :Int32;
  integLines @4 :Int32;
  globalGain @5 :Int32;
  image @6 :Data;
}

struct GPSNMEAData {
  timestamp @0 :Int64;
  localWallTime @1 :UInt64;
  nmea @2 :Text;
}

# android sensor_event_t
struct SensorEventData {
  version @0 :Int32;
  sensor @1 :Int32;
  type @2 :Int32;
  timestamp @3 :Int64;
  union {
    acceleration @4 :SensorVec;
    magnetic @5 :SensorVec;
    orientation @6 :SensorVec;
    gyro @7 :SensorVec;
  }
  source @8 :SensorSource;

  struct SensorVec {
    v @0 :List(Float32);
    status @1 :Int8;
  }

  enum SensorSource {
    android @0;
    iOS @1;
    fiber @2;
    velodyne @3;  # Velodyne IMU
  }
}

# android struct GpsLocation
struct GpsLocationData {
  # Contains GpsLocationFlags bits.
  flags @0 :UInt16;

  # Represents latitude in degrees.
  latitude @1 :Float64;

  # Represents longitude in degrees.
  longitude @2 :Float64;

  # Represents altitude in meters above the WGS 84 reference ellipsoid.
  altitude @3 :Float64;

  # Represents speed in meters per second.
  speed @4 :Float32;

  # Represents heading in degrees.
  bearing @5 :Float32;

  # Represents expected accuracy in meters.
  accuracy @6 :Float32;

  # Timestamp for the location fix.
  # Milliseconds since January 1, 1970.
  timestamp @7 :Int64;
}

struct CanData {
  address @0 :UInt32;
  busTime @1 :UInt16;
  dat     @2 :Data;
  src     @3 :Int8;
}

struct ThermalData {
  cpu0 @0 :UInt16;
  cpu1 @1 :UInt16;
  cpu2 @2 :UInt16;
  cpu3 @3 :UInt16;
  mem @4 :UInt16;
  gpu @5 :UInt16;
  bat @6 :UInt32;

  # not thermal
  freeSpace @7 :Float32;
  batteryPercent @8 :Int16;
}

struct HealthData {
  # from can health
  voltage @0 :UInt32;
  current @1 :UInt32;
  started @2 :Bool;
  controlsAllowed @3 :Bool;
  gasInterceptorDetected @4 :Bool;
  startedSignalDetected @5 :Bool;
}

struct LiveUI {
  rearViewCam @0 :Bool;
  alertText1 @1 :Text; 
  alertText2 @2 :Text; 
  awarenessStatus @3 :Float32;
}

struct Live20Data {
  canMonoTimes @10 :List(UInt64);
  mdMonoTime @6 :UInt64;
  ftMonoTime @7 :UInt64;

  # all deprecated
  warpMatrixDEPRECATED @0 :List(Float32);
  angleOffsetDEPRECATED @1 :Float32;
  calStatusDEPRECATED @2 :Int8;
  calCycleDEPRECATED @8 :Int32;
  calPercDEPRECATED @9 :Int8;

  leadOne @3 :LeadData;
  leadTwo @4 :LeadData;
  cumLagMs @5 :Float32;

  struct LeadData {
    dRel @0 :Float32;
    yRel @1 :Float32;
    vRel @2 :Float32;
    aRel @3 :Float32;
    vLead @4 :Float32;
    aLead @5 :Float32;
    dPath @6 :Float32;
    vLat @7 :Float32;
    vLeadK @8 :Float32;
    aLeadK @9 :Float32;
    fcw @10 :Bool;
    status @11 :Bool;
  }
}

struct LiveCalibrationData {
  warpMatrix @0 :List(Float32);
  calStatus @1 :Int8;
  calCycle @2 :Int32;
  calPerc @3 :Int8;
}

struct LiveTracks {
  trackId @0 :Int32;
  dRel @1 :Float32;
  yRel @2 :Float32;
  vRel @3 :Float32;
  aRel @4 :Float32;
  timeStamp @5 :Float32;
  status @6 :Float32;
  currentTime @7 :Float32;
  stationary @8 :Bool;
  oncoming @9 :Bool;
}

struct Live100Data {
  canMonoTime @16 :UInt64;
  canMonoTimes @21 :List(UInt64);
  l20MonoTime @17 :UInt64;
  mdMonoTime @18 :UInt64;

  vEgo @0 :Float32;
  aEgoDEPRECATED @1 :Float32;
  vPid @2 :Float32;
  vTargetLead @3 :Float32;
  upAccelCmd @4 :Float32;
  uiAccelCmd @5 :Float32;
  yActual @6 :Float32;
  yDes @7 :Float32;
  upSteer @8 :Float32;
  uiSteer @9 :Float32;
  aTargetMin @10 :Float32;
  aTargetMax @11 :Float32;
  jerkFactor @12 :Float32;
  angleSteers @13 :Float32;
  hudLeadDEPRECATED @14 :Int32;
  cumLagMs @15 :Float32;

  enabled @19: Bool;
  steerOverride @20: Bool;

  vCruise @22: Float32;

  rearViewCam @23 :Bool;
  alertText1 @24 :Text; 
  alertText2 @25 :Text; 
  awarenessStatus @26 :Float32;
}

struct LiveEventData {
  name @0 :Text;
  value @1 :Int32;
}

struct ModelData {
  frameId @0 :UInt32;

  path @1 :PathData;
  leftLane @2 :PathData;
  rightLane @3 :PathData;
  lead @4 :LeadData;

  settings @5 :ModelSettings;

  struct PathData {
    points @0 :List(Float32);
    prob @1 :Float32;
    std @2 :Float32;
  }

  struct LeadData {
    dist @0 :Float32;
    prob @1 :Float32;
    std @2 :Float32;
  }

  struct ModelSettings {
    bigBoxX @0 :UInt16;
    bigBoxY @1 :UInt16;
    bigBoxWidth @2 :UInt16;
    bigBoxHeight @3 :UInt16;
    boxProjection @4 :List(Float32);
    yuvCorrection @5 :List(Float32);
  }
}

struct CalibrationFeatures {
  frameId @0 :UInt32;

  p0 @1 :List(Float32);
  p1 @2 :List(Float32);
  status @3 :List(Int8);
}

struct EncodeIndex {
  # picture from camera 
  frameId @0 :UInt32;
  type @1 :Type;
  # index of encoder from start of route
  encodeId @2 :UInt32;
  # minute long segment this frame is in
  segmentNum @3 :Int32;
  # index into camera file in segment from 0
  segmentId @4 :UInt32;

  enum Type {
    bigBoxLossless @0;   # rcamera.mkv
    fullHEVC @1;         # fcamera.hevc
    bigBoxHEVC @2;       # bcamera.hevc
  }
}

struct AndroidLogEntry {
  id @0 :UInt8;
  ts @1 :UInt64;
  priority @2 :UInt8;
  pid @3 :Int32;
  tid @4 :Int32;
  tag @5 :Text;
  message @6 :Text;
}

struct LogRotate {
  segmentNum @0 :Int32;
  path @1 :Text;
}


struct Event {
  logMonoTime @0 :UInt64;

  union {
    initData @1 :InitData;
    frame @2 :FrameData;
    gpsNMEA @3 :GPSNMEAData;
    sensorEventDEPRECATED @4 :SensorEventData;
    can @5 :List(CanData);
    thermal @6 :ThermalData;
    live100 @7 :Live100Data;
    liveEventDEPRECATED @8 :List(LiveEventData);
    model @9 :ModelData;
    features @10 :CalibrationFeatures;
    sensorEvents @11 :List(SensorEventData);
    health @12 : HealthData;
    live20 @13 :Live20Data;
    liveUIDEPRECATED @14 :LiveUI;
    encodeIdx @15 :EncodeIndex;
    liveTracks @16 :List(LiveTracks);
    sendcan @17 :List(CanData);
    logMessage @18 :Text;
    liveCalibration @19 :LiveCalibrationData;
    androidLogEntry @20 :AndroidLogEntry;
    gpsLocation @21 :GpsLocationData;
    carState @22 :Car.CarState;
  }
}
