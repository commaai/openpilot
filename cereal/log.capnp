using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Java = import "./include/java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("Log");

using Car = import "car.capnp";

@0xf3b1f17e25a4285b;

const logVersion :Int32 = 1;

struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}

struct InitData {
  kernelArgs @0 :List(Text);
  kernelVersion @15 :Text;

  gctx @1 :Text;
  dongleId @2 :Text;

  deviceType @3 :DeviceType;
  version @4 :Text;
  gitCommit @10 :Text;
  gitBranch @11 :Text;
  gitRemote @13 :Text;

  androidBuildInfo @5 :AndroidBuildInfo;
  androidSensors @6 :List(AndroidSensor);
  androidProperties @16 :Map(Text, Text);
  chffrAndroidExtra @7 :ChffrAndroidExtra;
  iosBuildInfo @14 :IosBuildInfo;

  pandaInfo @8 :PandaInfo;

  dirty @9 :Bool;
  passive @12 :Bool;
  params @17 :Map(Text, Text);

  enum DeviceType {
    unknown @0;
    neo @1;
    chffrAndroid @2;
    chffrIos @3;
  }

  struct AndroidBuildInfo {
    board @0 :Text;
    bootloader @1 :Text;
    brand @2 :Text;
    device @3 :Text;
    display @4 :Text;
    fingerprint @5 :Text;
    hardware @6 :Text;
    host @7 :Text;
    id @8 :Text;
    manufacturer @9 :Text;
    model @10 :Text;
    product @11 :Text;
    radioVersion @12 :Text;
    serial @13 :Text;
    supportedAbis @14 :List(Text);
    tags @15 :Text;
    time @16 :Int64;
    type @17 :Text;
    user @18 :Text;

    versionCodename @19 :Text;
    versionRelease @20 :Text;
    versionSdk @21 :Int32;
    versionSecurityPatch @22 :Text;
  }

  struct AndroidSensor {
    id @0 :Int32;
    name @1 :Text;
    vendor @2 :Text;
    version @3 :Int32;
    handle @4 :Int32;
    type @5 :Int32;
    maxRange @6 :Float32;
    resolution @7 :Float32;
    power @8 :Float32;
    minDelay @9 :Int32;
    fifoReservedEventCount @10 :UInt32;
    fifoMaxEventCount @11 :UInt32;
    stringType @12 :Text;
    maxDelay @13 :Int32;
  }

  struct ChffrAndroidExtra {
    allCameraCharacteristics @0 :Map(Text, Text);
  }

  struct IosBuildInfo {
    appVersion @0 :Text;
    appBuild @1 :UInt32;
    osVersion @2 :Text;
    deviceModel @3 :Text;
  }

  struct PandaInfo {
    hasPanda @0 :Bool;
    dongleId @1 :Text;
    stVersion @2 :Text;
    espVersion @3 :Text;
  }
}

struct FrameData {
  frameId @0 :UInt32;
  encodeId @1 :UInt32; # DEPRECATED
  timestampEof @2 :UInt64;
  frameLength @3 :Int32;
  integLines @4 :Int32;
  globalGain @5 :Int32;
  lensPos @11 :Int32;
  lensSag @12 :Float32;
  lensErr @13 :Float32;
  lensTruePos @14 :Float32;
  image @6 :Data;
  gainFrac @15 :Float32;

  frameType @7 :FrameType;
  timestampSof @8 :UInt64;
  transform @10 :List(Float32);

  androidCaptureResult @9 :AndroidCaptureResult;

  enum FrameType {
    unknown @0;
    neo @1;
    chffrAndroid @2;
    front @3;
  }

  struct AndroidCaptureResult {
    sensitivity @0 :Int32;
    frameDuration @1 :Int64;
    exposureTime @2 :Int64;
    rollingShutterSkew @3 :UInt64;
    colorCorrectionTransform @4 :List(Int32);
    colorCorrectionGains @5 :List(Float32);
    displayRotation @6 :Int8;
  }
}

struct Thumbnail {
  frameId @0 :UInt32;
  timestampEof @1 :UInt64;
  thumbnail @2 :Data;
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
  uncalibratedDEPRECATED @10 :Bool;

  union {
    acceleration @4 :SensorVec;
    magnetic @5 :SensorVec;
    orientation @6 :SensorVec;
    gyro @7 :SensorVec;
    pressure @9 :SensorVec;
    magneticUncalibrated @11 :SensorVec;
    gyroUncalibrated @12 :SensorVec;
    proximity @13: Float32;
    light @14: Float32;
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
    bno055 @4;    # Bosch accelerometer
    lsm6ds3 @5;   # accelerometer (c2)
    bmp280 @6;    # barometer (c2)
    mmc3416x @7;  # magnetometer (c2)
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

  # Represents expected accuracy in meters. (presumably 1 sigma?)
  accuracy @6 :Float32;

  # Timestamp for the location fix.
  # Milliseconds since January 1, 1970.
  timestamp @7 :Int64;

  source @8 :SensorSource;

  # Represents NED velocity in m/s.
  vNED @9 :List(Float32);

  # Represents expected vertical accuracy in meters. (presumably 1 sigma?)
  verticalAccuracy @10 :Float32;

  # Represents bearing accuracy in degrees. (presumably 1 sigma?)
  bearingAccuracy @11 :Float32;

  # Represents velocity accuracy in m/s. (presumably 1 sigma?)
  speedAccuracy @12 :Float32;

  enum SensorSource {
    android @0;
    iOS @1;
    car @2;
    velodyne @3;  # Velodyne IMU
    fusion @4;
    external @5;
    ublox @6;
    trimble @7;
  }
}

struct CanData {
  address @0 :UInt32;
  busTime @1 :UInt16;
  dat     @2 :Data;
  src     @3 :UInt8;
}

struct ThermalData {
  cpu0 @0 :UInt16;
  cpu1 @1 :UInt16;
  cpu2 @2 :UInt16;
  cpu3 @3 :UInt16;
  mem @4 :UInt16;
  gpu @5 :UInt16;
  bat @6 :UInt32;
  pa0 @21 :UInt16;

  # not thermal
  freeSpace @7 :Float32;
  batteryPercent @8 :Int16;
  batteryStatus @9 :Text;
  batteryCurrent @15 :Int32;
  batteryVoltage @16 :Int32;
  usbOnline @12 :Bool;

  fanSpeed @10 :UInt16;
  started @11 :Bool;
  startedTs @13 :UInt64;

  thermalStatus @14 :ThermalStatus;
  chargingError @17 :Bool;
  chargingDisabled @18 :Bool;

  memUsedPercent @19 :Int8;
  cpuPerc @20 :Int8;

  enum ThermalStatus {
    green @0;   # all processes run
    yellow @1;  # critical processes run (kill uploader), engage still allowed
    red @2;     # no engage, will disengage
    danger @3;  # immediate process shutdown
  }
}

struct HealthData {
  # from can health
  voltage @0 :UInt32;
  current @1 :UInt32;
  ignitionLine @2 :Bool;
  controlsAllowed @3 :Bool;
  gasInterceptorDetected @4 :Bool;
  startedSignalDetectedDeprecated @5 :Bool;
  hasGps @6 :Bool;
  canSendErrs @7 :UInt32;
  canFwdErrs @8 :UInt32;
  canRxErrs @19 :UInt32;
  gmlanSendErrs @9 :UInt32;
  hwType @10 :HwType;
  fanSpeedRpm @11 :UInt16;
  usbPowerMode @12 :UsbPowerMode;
  ignitionCan @13 :Bool;
  safetyModel @14 :Car.CarParams.SafetyModel;
  faultStatus @15 :FaultStatus;
  powerSaveEnabled @16 :Bool;
  uptime @17 :UInt32;
  faults @18 :List(FaultType);

  enum FaultStatus {
    none @0;
    faultTemp @1;
    faultPerm @2;
  }

  enum FaultType {
    relayMalfunction @0;
  }

  enum HwType {
    unknown @0;
    whitePanda @1;
    greyPanda @2;
    blackPanda @3;
    pedal @4;
    uno @5;
  }

  enum UsbPowerMode {
    none @0;
    client @1;
    cdp @2;
    dcp @3;
  }
}

struct LiveUI {
  rearViewCam @0 :Bool;
  alertText1 @1 :Text;
  alertText2 @2 :Text;
  awarenessStatus @3 :Float32;
}

struct RadarState @0x9a185389d6fdd05f {
  canMonoTimes @10 :List(UInt64);
  mdMonoTime @6 :UInt64;
  ftMonoTimeDEPRECATED @7 :UInt64;
  controlsStateMonoTime @11 :UInt64;
  radarErrors @12 :List(Car.RadarData.Error);

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
    aLeadDEPRECATED @5 :Float32;
    dPath @6 :Float32;
    vLat @7 :Float32;
    vLeadK @8 :Float32;
    aLeadK @9 :Float32;
    fcw @10 :Bool;
    status @11 :Bool;
    aLeadTau @12 :Float32;
    modelProb @13 :Float32;
    radar @14 :Bool;
  }
}

struct LiveCalibrationData {
  # deprecated
  warpMatrix @0 :List(Float32);
  # camera_frame_from_model_frame
  warpMatrix2 @5 :List(Float32);
  warpMatrixBig @6 :List(Float32);
  calStatus @1 :Int8;
  calCycle @2 :Int32;
  calPerc @3 :Int8;

  # view_frame_from_road_frame
  # ui's is inversed needs new
  extrinsicMatrix @4 :List(Float32);
  # the direction of travel vector in device frame
  rpyCalib @7 :List(Float32);
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

struct ControlsState @0x97ff69c53601abf1 {
  canMonoTimeDEPRECATED @16 :UInt64;
  canMonoTimes @21 :List(UInt64);
  radarStateMonoTimeDEPRECATED @17 :UInt64;
  mdMonoTimeDEPRECATED @18 :UInt64;
  planMonoTime @28 :UInt64;
  pathPlanMonoTime @50 :UInt64;

  state @31 :OpenpilotState;
  vEgo @0 :Float32;
  vEgoRaw @32 :Float32;
  aEgoDEPRECATED @1 :Float32;
  longControlState @30 :LongControlState;
  vPid @2 :Float32;
  vTargetLead @3 :Float32;
  upAccelCmd @4 :Float32;
  uiAccelCmd @5 :Float32;
  ufAccelCmd @33 :Float32;
  yActualDEPRECATED @6 :Float32;
  yDesDEPRECATED @7 :Float32;
  upSteerDEPRECATED @8 :Float32;
  uiSteerDEPRECATED @9 :Float32;
  ufSteerDEPRECATED @34 :Float32;
  aTargetMinDEPRECATED @10 :Float32;
  aTargetMaxDEPRECATED @11 :Float32;
  aTarget @35 :Float32;
  jerkFactor @12 :Float32;
  angleSteers @13 :Float32;     # Steering angle in degrees.
  angleSteersDes @29 :Float32;
  curvature @37 :Float32;       # path curvature from vehicle model
  hudLeadDEPRECATED @14 :Int32;
  cumLagMs @15 :Float32;
  startMonoTime @48 :UInt64;
  mapValid @49 :Bool;
  forceDecel @51 :Bool;

  enabled @19 :Bool;
  active @36 :Bool;
  steerOverride @20 :Bool;

  vCruise @22 :Float32;

  rearViewCam @23 :Bool;
  alertText1 @24 :Text;
  alertText2 @25 :Text;
  alertStatus @38 :AlertStatus;
  alertSize @39 :AlertSize;
  alertBlinkingRate @42 :Float32;
  alertType @44 :Text;
  alertSoundDEPRECATED @45 :Text;
  alertSound @56 :Car.CarControl.HUDControl.AudibleAlert;
  awarenessStatus @26 :Float32;
  angleModelBiasDEPRECATED @27 :Float32;
  gpsPlannerActive @40 :Bool;
  engageable @41 :Bool;  # can OP be engaged?
  driverMonitoringOn @43 :Bool;

  # maps
  vCurvature @46 :Float32;
  decelForTurn @47 :Bool;

  decelForModel @54 :Bool;
  canErrorCounter @57 :UInt32;

  lateralControlState :union {
    indiState @52 :LateralINDIState;
    pidState @53 :LateralPIDState;
    lqrState @55 :LateralLQRState;
  }

  enum OpenpilotState @0xdbe58b96d2d1ac61 {
    disabled @0;
    preEnabled @1;
    enabled @2;
    softDisabling @3;
  }

  enum LongControlState {
    off @0;
    pid @1;
    stopping @2;
    starting @3;
  }

  enum AlertStatus {
    normal @0;       # low priority alert for user's convenience
    userPrompt @1;   # mid piority alert that might require user intervention
    critical @2;     # high priority alert that needs immediate user intervention
  }

  enum AlertSize {
    none @0;    # don't display the alert
    small @1;   # small box
    mid @2;     # mid screen
    full @3;    # full screen
  }

  struct LateralINDIState {
    active @0 :Bool;
    steerAngle @1 :Float32;
    steerRate @2 :Float32;
    steerAccel @3 :Float32;
    rateSetPoint @4 :Float32;
    accelSetPoint @5 :Float32;
    accelError @6 :Float32;
    delayedOutput @7 :Float32;
    delta @8 :Float32;
    output @9 :Float32;
    saturated @10 :Bool;
  }

  struct LateralPIDState {
    active @0 :Bool;
    steerAngle @1 :Float32;
    steerRate @2 :Float32;
    angleError @3 :Float32;
    p @4 :Float32;
    i @5 :Float32;
    f @6 :Float32;
    output @7 :Float32;
    saturated @8 :Bool;
   }

  struct LateralLQRState {
    active @0 :Bool;
    steerAngle @1 :Float32;
    i @2 :Float32;
    output @3 :Float32;
    lqrOutput @4 :Float32;
    saturated @5 :Bool;
   }


}

struct LiveEventData {
  name @0 :Text;
  value @1 :Int32;
}

struct ModelData {
  frameId @0 :UInt32;
  timestampEof @9 :UInt64;

  path @1 :PathData;
  leftLane @2 :PathData;
  rightLane @3 :PathData;
  lead @4 :LeadData;
  freePath @6 :List(Float32);

  settings @5 :ModelSettings;
  leadFuture @7 :LeadData;
  speed @8 :List(Float32);
  meta @10 :MetaData;
  longitudinal @11 :LongitudinalData;

  struct PathData {
    points @0 :List(Float32);
    prob @1 :Float32;
    std @2 :Float32;
    stds @3 :List(Float32);
    poly @4 :List(Float32);
  }

  struct LeadData {
    dist @0 :Float32;
    prob @1 :Float32;
    std @2 :Float32;
    relVel @3 :Float32;
    relVelStd @4 :Float32;
    relY @5 :Float32;
    relYStd @6 :Float32;
    relA @7 :Float32;
    relAStd @8 :Float32;
  }

  struct ModelSettings {
    bigBoxX @0 :UInt16;
    bigBoxY @1 :UInt16;
    bigBoxWidth @2 :UInt16;
    bigBoxHeight @3 :UInt16;
    boxProjection @4 :List(Float32);
    yuvCorrection @5 :List(Float32);
    inputTransform @6 :List(Float32);
  }

  struct MetaData {
    engagedProb @0 :Float32;
    desirePrediction @1 :List(Float32);
    brakeDisengageProb @2 :Float32;
    gasDisengageProb @3 :Float32;
    steerOverrideProb @4 :Float32;
  }

  struct LongitudinalData {
    speeds @0 :List(Float32);
    accelerations @1 :List(Float32);
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
  # index into camera file in segment in presentation order
  segmentId @4 :UInt32;
  # index into camera file in segment in encode order
  segmentIdEncode @5 :UInt32;

  enum Type {
    bigBoxLossless @0;   # rcamera.mkv
    fullHEVC @1;         # fcamera.hevc
    bigBoxHEVC @2;       # bcamera.hevc
    chffrAndroidH264 @3; # acamera
    fullLosslessClip @4; # prcamera.mkv
    front @5;            # dcamera.hevc
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

struct Plan {
  mdMonoTime @9 :UInt64;
  radarStateMonoTime @10 :UInt64;
  commIssue @31 :Bool;

  eventsDEPRECATED @13 :List(Car.CarEvent);

  # lateral, 3rd order polynomial
  lateralValidDEPRECATED @0 :Bool;
  dPolyDEPRECATED @1 :List(Float32);
  laneWidthDEPRECATED @11 :Float32;

  # longitudinal
  longitudinalValidDEPRECATED @2 :Bool;
  vCruise @16 :Float32;
  aCruise @17 :Float32;
  vTarget @3 :Float32;
  vTargetFuture @14 :Float32;
  vMax @20 :Float32;
  aTargetMinDEPRECATED @4 :Float32;
  aTargetMaxDEPRECATED @5 :Float32;
  aTarget @18 :Float32;

  vStart @26 :Float32;
  aStart @27 :Float32;

  jerkFactor @6 :Float32;
  hasLead @7 :Bool;
  hasLeftLaneDEPRECATED @23 :Bool;
  hasRightLaneDEPRECATED @24 :Bool;
  fcw @8 :Bool;
  longitudinalPlanSource @15 :LongitudinalPlanSource;

  # gps trajectory in car frame
  gpsTrajectory @12 :GpsTrajectory;

  gpsPlannerActive @19 :Bool;

  # maps
  vCurvature @21 :Float32;
  decelForTurn @22 :Bool;
  mapValid @25 :Bool;
  radarValid @28 :Bool;
  radarCanError @30 :Bool;

  processingDelay @29 :Float32;


  struct GpsTrajectory {
    x @0 :List(Float32);
    y @1 :List(Float32);
  }

  enum LongitudinalPlanSource {
    cruise @0;
    mpc1 @1;
    mpc2 @2;
    mpc3 @3;
    model @4;
  }
}

struct PathPlan {
  laneWidth @0 :Float32;

  dPoly @1 :List(Float32);
  cPoly @2 :List(Float32);
  cProb @3 :Float32;
  lPoly @4 :List(Float32);
  lProb @5 :Float32;
  rPoly @6 :List(Float32);
  rProb @7 :Float32;

  angleSteers @8 :Float32; # deg
  rateSteers @13 :Float32; # deg/s
  mpcSolutionValid @9 :Bool;
  paramsValid @10 :Bool;
  modelValidDEPRECATED @12 :Bool;
  angleOffset @11 :Float32;
  sensorValid @14 :Bool;
  commIssue @15 :Bool;
  posenetValid @16 :Bool;
  desire @17 :Desire;
  laneChangeState @18 :LaneChangeState;
  laneChangeDirection @19 :LaneChangeDirection;

  enum Desire {
    none @0;
    turnLeft @1;
    turnRight @2;
    laneChangeLeft @3;
    laneChangeRight @4;
    keepLeft @5;
    keepRight @6;
  }

  enum LaneChangeState {
    off @0;
    preLaneChange @1;
    laneChangeStarting @2;
    laneChangeFinishing @3;
  }

  enum LaneChangeDirection {
    none @0;
    left @1;
    right @2;
  }
}

struct LiveLocationData {
  status @0 :UInt8;

  # 3D fix
  lat @1 :Float64;
  lon @2 :Float64;
  alt @3 :Float32;     # m

  # speed
  speed @4 :Float32;   # m/s

  # NED velocity components
  vNED @5 :List(Float32);

  # roll, pitch, heading (x,y,z)
  roll @6 :Float32;     # WRT to center of earth?
  pitch @7 :Float32;    # WRT to center of earth?
  heading @8 :Float32;  # WRT to north?

  # what are these?
  wanderAngle @9 :Float32;
  trackAngle @10 :Float32;

  # car frame -- https://upload.wikimedia.org/wikipedia/commons/f/f5/RPY_angles_of_cars.png

  # gyro, in car frame, deg/s
  gyro @11 :List(Float32);

  # accel, in car frame, m/s^2
  accel @12 :List(Float32);

  accuracy @13 :Accuracy;

  source @14 :SensorSource;
  # if we are fixing a location in the past
  fixMonoTime @15 :UInt64;

  gpsWeek @16 :Int32;
  timeOfWeek @17 :Float64;

  positionECEF @18 :List(Float64);
  poseQuatECEF @19 :List(Float32);
  pitchCalibration @20 :Float32;
  yawCalibration @21 :Float32;
  imuFrame @22 :List(Float32);

  struct Accuracy {
    pNEDError @0 :List(Float32);
    vNEDError @1 :List(Float32);
    rollError @2 :Float32;
    pitchError @3 :Float32;
    headingError @4 :Float32;
    ellipsoidSemiMajorError @5 :Float32;
    ellipsoidSemiMinorError @6 :Float32;
    ellipsoidOrientationError @7 :Float32;
  }

  enum SensorSource {
    applanix @0;
    kalman @1;
    orbslam @2;
    timing @3;
    dummy @4;
  }
}

struct EthernetPacket {
  pkt @0 :Data;
  ts @1 :Float32;
}

struct NavUpdate {
  isNavigating @0 :Bool;
  curSegment @1 :Int32;
  segments @2 :List(Segment);

  struct LatLng {
    lat @0 :Float64;
    lng @1 :Float64;
  }

  struct Segment {
    from @0 :LatLng;
    to @1 :LatLng;
    updateTime @2 :Int32;
    distance @3 :Int32;
    crossTime @4 :Int32;
    exitNo @5 :Int32;
    instruction @6 :Instruction;

    parts @7 :List(LatLng);

    enum Instruction {
      turnLeft @0;
      turnRight @1;
      keepLeft @2;
      keepRight @3;
      straight @4;
      roundaboutExitNumber @5;
      roundaboutExit @6;
      roundaboutTurnLeft @7;
      unkn8 @8;
      roundaboutStraight @9;
      unkn10 @10;
      roundaboutTurnRight @11;
      unkn12 @12;
      roundaboutUturn @13;
      unkn14 @14;
      arrive @15;
      exitLeft @16;
      exitRight @17;
      unkn18 @18;
      uturn @19;
      # ...
    }
  }
}

struct NavStatus {
  isNavigating @0 :Bool;
  currentAddress @1 :Address;

  struct Address {
    title @0 :Text;
    lat @1 :Float64;
    lng @2 :Float64;
    house @3 :Text;
    address @4 :Text;
    street @5 :Text;
    city @6 :Text;
    state @7 :Text;
    country @8 :Text;
  }
}

struct CellInfo {
  timestamp @0 :UInt64;
  repr @1 :Text; # android toString() for now
}

struct WifiScan {
  bssid @0 :Text;
  ssid @1 :Text;
  capabilities @2 :Text;
  frequency @3 :Int32;
  level @4 :Int32;
  timestamp @5 :Int64;

  centerFreq0 @6 :Int32;
  centerFreq1 @7 :Int32;
  channelWidth @8 :ChannelWidth;
  operatorFriendlyName @9 :Text;
  venueName @10 :Text;
  is80211mcResponder @11 :Bool;
  passpoint @12 :Bool;

  distanceCm @13 :Int32;
  distanceSdCm @14 :Int32;

  enum ChannelWidth {
    w20Mhz @0;
    w40Mhz @1;
    w80Mhz @2;
    w160Mhz @3;
    w80Plus80Mhz @4;
  }
}

struct AndroidGnss {
  union {
    measurements @0 :Measurements;
    navigationMessage @1 :NavigationMessage;
  }

  struct Measurements {
    clock @0 :Clock;
    measurements @1 :List(Measurement);

    struct Clock {
      timeNanos @0 :Int64;
      hardwareClockDiscontinuityCount @1 :Int32;

      hasTimeUncertaintyNanos @2 :Bool;
      timeUncertaintyNanos @3 :Float64;

      hasLeapSecond @4 :Bool;
      leapSecond @5 :Int32;

      hasFullBiasNanos @6 :Bool;
      fullBiasNanos @7 :Int64;

      hasBiasNanos @8 :Bool;
      biasNanos @9 :Float64;

      hasBiasUncertaintyNanos @10 :Bool;
      biasUncertaintyNanos @11 :Float64;

      hasDriftNanosPerSecond @12 :Bool;
      driftNanosPerSecond @13 :Float64;

      hasDriftUncertaintyNanosPerSecond @14 :Bool;
      driftUncertaintyNanosPerSecond @15 :Float64;
    }

    struct Measurement {
      svId @0 :Int32;
      constellation @1 :Constellation;

      timeOffsetNanos @2 :Float64;
      state @3 :Int32;
      receivedSvTimeNanos @4 :Int64;
      receivedSvTimeUncertaintyNanos @5 :Int64;
      cn0DbHz @6 :Float64;
      pseudorangeRateMetersPerSecond @7 :Float64;
      pseudorangeRateUncertaintyMetersPerSecond @8 :Float64;
      accumulatedDeltaRangeState @9 :Int32;
      accumulatedDeltaRangeMeters @10 :Float64;
      accumulatedDeltaRangeUncertaintyMeters @11 :Float64;

      hasCarrierFrequencyHz @12 :Bool;
      carrierFrequencyHz @13 :Float32;
      hasCarrierCycles @14 :Bool;
      carrierCycles @15 :Int64;
      hasCarrierPhase @16 :Bool;
      carrierPhase @17 :Float64;
      hasCarrierPhaseUncertainty @18 :Bool;
      carrierPhaseUncertainty @19 :Float64;
      hasSnrInDb @20 :Bool;
      snrInDb @21 :Float64;

      multipathIndicator @22 :MultipathIndicator;

      enum Constellation {
        unknown @0;
        gps @1;
        sbas @2;
        glonass @3;
        qzss @4;
        beidou @5;
        galileo @6;
      }

      enum State {
        unknown @0;
        codeLock @1;
        bitSync @2;
        subframeSync @3;
        towDecoded @4;
        msecAmbiguous @5;
        symbolSync @6;
        gloStringSync @7;
        gloTodDecoded @8;
        bdsD2BitSync @9;
        bdsD2SubframeSync @10;
        galE1bcCodeLock @11;
        galE1c2ndCodeLock @12;
        galE1bPageSync @13;
        sbasSync @14;
      }

      enum MultipathIndicator {
        unknown @0;
        detected @1;
        notDetected @2;
      }
    }
  }

  struct NavigationMessage {
    type @0 :Int32;
    svId @1 :Int32;
    messageId @2 :Int32;
    submessageId @3 :Int32;
    data @4 :Data;
    status @5 :Status;

    enum Status {
      unknown @0;
      parityPassed @1;
      parityRebuilt @2;
    }
  }
}

struct QcomGnss {
  logTs @0 :UInt64;
  union {
    measurementReport @1 :MeasurementReport;
    clockReport @2 :ClockReport;
    drMeasurementReport @3 :DrMeasurementReport;
    drSvPoly @4 :DrSvPolyReport;
    rawLog @5 :Data;
  }

  enum MeasurementSource @0xd71a12b6faada7ee {
    gps @0;
    glonass @1;
    beidou @2;
  }

  enum SVObservationState @0xe81e829a0d6c83e9 {
    idle @0;
    search @1;
    searchVerify @2;
    bitEdge @3;
    trackVerify @4;
    track @5;
    restart @6;
    dpo @7;
    glo10msBe @8;
    glo10msAt @9;
  }

  struct MeasurementStatus @0xe501010e1bcae83b {
    subMillisecondIsValid @0 :Bool;
    subBitTimeIsKnown @1 :Bool;
    satelliteTimeIsKnown @2 :Bool;
    bitEdgeConfirmedFromSignal @3 :Bool;
    measuredVelocity @4 :Bool;
    fineOrCoarseVelocity @5 :Bool;
    lockPointValid @6 :Bool;
    lockPointPositive @7 :Bool;
    lastUpdateFromDifference @8 :Bool;
    lastUpdateFromVelocityDifference @9 :Bool;
    strongIndicationOfCrossCorelation @10 :Bool;
    tentativeMeasurement @11 :Bool;
    measurementNotUsable @12 :Bool;
    sirCheckIsNeeded @13 :Bool;
    probationMode @14 :Bool;

    glonassMeanderBitEdgeValid @15 :Bool;
    glonassTimeMarkValid @16 :Bool;

    gpsRoundRobinRxDiversity @17 :Bool;
    gpsRxDiversity @18 :Bool;
    gpsLowBandwidthRxDiversityCombined @19 :Bool;
    gpsHighBandwidthNu4 @20 :Bool;
    gpsHighBandwidthNu8 @21 :Bool;
    gpsHighBandwidthUniform @22 :Bool;
    multipathIndicator @23 :Bool;

    imdJammingIndicator @24 :Bool;
    lteB13TxJammingIndicator @25 :Bool;
    freshMeasurementIndicator @26 :Bool;

    multipathEstimateIsValid @27 :Bool;
    directionIsValid @28 :Bool;
  }

  struct MeasurementReport {
    source @0 :MeasurementSource;

    fCount @1 :UInt32;

    gpsWeek @2 :UInt16;
    glonassCycleNumber @3 :UInt8;
    glonassNumberOfDays @4 :UInt16;

    milliseconds @5 :UInt32;
    timeBias @6 :Float32;
    clockTimeUncertainty @7 :Float32;
    clockFrequencyBias @8 :Float32;
    clockFrequencyUncertainty @9 :Float32;

    sv @10 :List(SV);

    struct SV {
      svId @0 :UInt8;
      observationState @2 :SVObservationState;
      observations @3 :UInt8;
      goodObservations @4 :UInt8;
      gpsParityErrorCount @5 :UInt16;
      glonassFrequencyIndex @1 :Int8;
      glonassHemmingErrorCount @6 :UInt8;
      filterStages @7 :UInt8;
      carrierNoise @8 :UInt16;
      latency @9 :Int16;
      predetectInterval @10 :UInt8;
      postdetections @11 :UInt16;

      unfilteredMeasurementIntegral @12 :UInt32;
      unfilteredMeasurementFraction @13 :Float32;
      unfilteredTimeUncertainty @14 :Float32;
      unfilteredSpeed @15 :Float32;
      unfilteredSpeedUncertainty @16 :Float32;
      measurementStatus @17 :MeasurementStatus;
      multipathEstimate @18 :UInt32;
      azimuth @19 :Float32;
      elevation @20 :Float32;
      carrierPhaseCyclesIntegral @21 :Int32;
      carrierPhaseCyclesFraction @22 :UInt16;
      fineSpeed @23 :Float32;
      fineSpeedUncertainty @24 :Float32;
      cycleSlipCount @25 :UInt8;
    }

  }

  struct ClockReport {
    hasFCount @0 :Bool;
    fCount @1 :UInt32;

    hasGpsWeek @2 :Bool;
    gpsWeek @3 :UInt16;
    hasGpsMilliseconds @4 :Bool;
    gpsMilliseconds @5 :UInt32;
    gpsTimeBias @6 :Float32;
    gpsClockTimeUncertainty @7 :Float32;
    gpsClockSource @8 :UInt8;

    hasGlonassYear @9 :Bool;
    glonassYear @10 :UInt8;
    hasGlonassDay @11 :Bool;
    glonassDay @12 :UInt16;
    hasGlonassMilliseconds @13 :Bool;
    glonassMilliseconds @14 :UInt32;
    glonassTimeBias @15 :Float32;
    glonassClockTimeUncertainty @16 :Float32;
    glonassClockSource @17 :UInt8;

    bdsWeek @18 :UInt16;
    bdsMilliseconds @19 :UInt32;
    bdsTimeBias @20 :Float32;
    bdsClockTimeUncertainty @21 :Float32;
    bdsClockSource @22 :UInt8;

    galWeek @23 :UInt16;
    galMilliseconds @24 :UInt32;
    galTimeBias @25 :Float32;
    galClockTimeUncertainty @26 :Float32;
    galClockSource @27 :UInt8;

    clockFrequencyBias @28 :Float32;
    clockFrequencyUncertainty @29 :Float32;
    frequencySource @30 :UInt8;
    gpsLeapSeconds @31 :UInt8;
    gpsLeapSecondsUncertainty @32 :UInt8;
    gpsLeapSecondsSource @33 :UInt8;

    gpsToGlonassTimeBiasMilliseconds @34 :Float32;
    gpsToGlonassTimeBiasMillisecondsUncertainty @35 :Float32;
    gpsToBdsTimeBiasMilliseconds @36 :Float32;
    gpsToBdsTimeBiasMillisecondsUncertainty @37 :Float32;
    bdsToGloTimeBiasMilliseconds @38 :Float32;
    bdsToGloTimeBiasMillisecondsUncertainty @39 :Float32;
    gpsToGalTimeBiasMilliseconds @40 :Float32;
    gpsToGalTimeBiasMillisecondsUncertainty @41 :Float32;
    galToGloTimeBiasMilliseconds @42 :Float32;
    galToGloTimeBiasMillisecondsUncertainty @43 :Float32;
    galToBdsTimeBiasMilliseconds @44 :Float32;
    galToBdsTimeBiasMillisecondsUncertainty @45 :Float32;

    hasRtcTime @46 :Bool;
    systemRtcTime @47 :UInt32;
    fCountOffset @48 :UInt32;
    lpmRtcCount @49 :UInt32;
    clockResets @50 :UInt32;
  }

  struct DrMeasurementReport {

    reason @0 :UInt8;
    seqNum @1 :UInt8;
    seqMax @2 :UInt8;
    rfLoss @3 :UInt16;

    systemRtcValid @4 :Bool;
    fCount @5 :UInt32;
    clockResets @6 :UInt32;
    systemRtcTime @7 :UInt64;

    gpsLeapSeconds @8 :UInt8;
    gpsLeapSecondsUncertainty @9 :UInt8;
    gpsToGlonassTimeBiasMilliseconds @10 :Float32;
    gpsToGlonassTimeBiasMillisecondsUncertainty @11 :Float32;

    gpsWeek @12 :UInt16;
    gpsMilliseconds @13 :UInt32;
    gpsTimeBiasMs @14 :UInt32;
    gpsClockTimeUncertaintyMs @15 :UInt32;
    gpsClockSource @16 :UInt8;

    glonassClockSource @17 :UInt8;
    glonassYear @18 :UInt8;
    glonassDay @19 :UInt16;
    glonassMilliseconds @20 :UInt32;
    glonassTimeBias @21 :Float32;
    glonassClockTimeUncertainty @22 :Float32;

    clockFrequencyBias @23 :Float32;
    clockFrequencyUncertainty @24 :Float32;
    frequencySource @25 :UInt8;

    source @26 :MeasurementSource;

    sv @27 :List(SV);

    struct SV {
      svId @0 :UInt8;
      glonassFrequencyIndex @1 :Int8;
      observationState @2 :SVObservationState;
      observations @3 :UInt8;
      goodObservations @4 :UInt8;
      filterStages @5 :UInt8;
      predetectInterval @6 :UInt8;
      cycleSlipCount @7 :UInt8;
      postdetections @8 :UInt16;

      measurementStatus @9 :MeasurementStatus;

      carrierNoise @10 :UInt16;
      rfLoss @11 :UInt16;
      latency @12 :Int16;

      filteredMeasurementFraction @13 :Float32;
      filteredMeasurementIntegral @14 :UInt32;
      filteredTimeUncertainty @15 :Float32;
      filteredSpeed @16 :Float32;
      filteredSpeedUncertainty @17 :Float32;

      unfilteredMeasurementFraction @18 :Float32;
      unfilteredMeasurementIntegral @19 :UInt32;
      unfilteredTimeUncertainty @20 :Float32;
      unfilteredSpeed @21 :Float32;
      unfilteredSpeedUncertainty @22 :Float32;

      multipathEstimate @23 :UInt32;
      azimuth @24 :Float32;
      elevation @25 :Float32;
      dopplerAcceleration @26 :Float32;
      fineSpeed @27 :Float32;
      fineSpeedUncertainty @28 :Float32;

      carrierPhase @29 :Float64;
      fCount @30 :UInt32;

      parityErrorCount @31 :UInt16;
      goodParity @32 :Bool;
    }
  }

  struct DrSvPolyReport {
    svId @0 :UInt16;
    frequencyIndex @1 :Int8;

    hasPosition @2 :Bool;
    hasIono @3 :Bool;
    hasTropo @4 :Bool;
    hasElevation @5 :Bool;
    polyFromXtra @6 :Bool;
    hasSbasIono @7 :Bool;

    iode @8 :UInt16;
    t0 @9 :Float64;
    xyz0 @10 :List(Float64);
    xyzN @11 :List(Float64);
    other @12 :List(Float32);

    positionUncertainty @13 :Float32;
    ionoDelay @14 :Float32;
    ionoDot @15 :Float32;
    sbasIonoDelay @16 :Float32;
    sbasIonoDot @17 :Float32;
    tropoDelay @18 :Float32;
    elevation @19 :Float32;
    elevationDot @20 :Float32;
    elevationUncertainty @21 :Float32;

    velocityCoeff @22 :List(Float64);

  }
}

struct LidarPts {
  r @0 :List(UInt16);        # uint16   m*500.0
  theta @1 :List(UInt16);    # uint16 deg*100.0
  reflect @2 :List(UInt8);   # uint8      0-255

  # For storing out of file.
  idx @3 :UInt64;

  # For storing in file
  pkt @4 :Data;
}

struct ProcLog {
  cpuTimes @0 :List(CPUTimes);
  mem @1 :Mem;
  procs @2 :List(Process);

  struct Process {
    pid @0 :Int32;
    name @1 :Text;
    state @2 :UInt8;
    ppid @3 :Int32;

    cpuUser @4 :Float32;
    cpuSystem @5 :Float32;
    cpuChildrenUser @6 :Float32;
    cpuChildrenSystem @7 :Float32;
    priority @8 :Int64;
    nice @9 :Int32;
    numThreads @10 :Int32;
    startTime @11 :Float64;

    memVms @12 :UInt64;
    memRss @13 :UInt64;

    processor @14 :Int32;

    cmdline @15 :List(Text);
    exe @16 :Text;
  }

  struct CPUTimes {
    cpuNum @0 :Int64;
    user @1 :Float32;
    nice @2 :Float32;
    system @3 :Float32;
    idle @4 :Float32;
    iowait @5 :Float32;
    irq @6 :Float32;
    softirq @7 :Float32;
  }

  struct Mem {
    total @0 :UInt64;
    free @1 :UInt64;
    available @2 :UInt64;
    buffers @3 :UInt64;
    cached @4 :UInt64;
    active @5 :UInt64;
    inactive @6 :UInt64;
    shared @7 :UInt64;
  }

}

struct UbloxGnss {
  union {
    measurementReport @0 :MeasurementReport;
    ephemeris @1 :Ephemeris;
    ionoData @2 :IonoData;
  }

  struct MeasurementReport {
    #received time of week in gps time in seconds and gps week
    rcvTow @0 :Float64;
    gpsWeek @1 :UInt16;
    # leap seconds in seconds
    leapSeconds @2 :UInt16;
    # receiver status
    receiverStatus @3 :ReceiverStatus;
    # num of measurements to follow
    numMeas @4 :UInt8;
    measurements @5 :List(Measurement);

    struct ReceiverStatus {
      # leap seconds have been determined
      leapSecValid @0 :Bool;
      # Clock reset applied
      clkReset @1 :Bool;
    }

    struct Measurement {
      svId @0 :UInt8;
      trackingStatus @1 :TrackingStatus;
      # pseudorange in meters
      pseudorange @2 :Float64;
      # carrier phase measurement in cycles
      carrierCycles @3 :Float64;
      # doppler measurement in Hz
      doppler @4 :Float32;
      # GNSS id, 0 is gps
      gnssId @5 :UInt8;
      glonassFrequencyIndex @6 :UInt8;
      # carrier phase locktime counter in ms
      locktime @7 :UInt16;
      # Carrier-to-noise density ratio (signal strength) in dBHz
      cno @8 :UInt8;
      # pseudorange standard deviation in meters
      pseudorangeStdev @9 :Float32;
      # carrier phase standard deviation in cycles
      carrierPhaseStdev @10 :Float32;
      # doppler standard deviation in Hz
      dopplerStdev @11 :Float32;
      sigId @12 :UInt8;

      struct TrackingStatus {
        # pseudorange valid
        pseudorangeValid @0 :Bool;
        # carrier phase valid
        carrierPhaseValid @1 :Bool;
        # half cycle valid
        halfCycleValid @2 :Bool;
        # half sycle subtracted from phase
        halfCycleSubtracted @3 :Bool;
      }
    }
  }

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
    gpsWeek @28 :Float64;
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

  }

  struct IonoData {
    svHealth @0 :UInt32;
    tow  @1 :Float64;
    gpsWeek @2 :Float64;

    ionoAlpha @3 :List(Float64);
    ionoBeta @4 :List(Float64);

    healthValid @5 :Bool;
    ionoCoeffsValid @6 :Bool;
  }
}


struct Clocks {
  bootTimeNanos @0 :UInt64;
  monotonicNanos @1 :UInt64;
  monotonicRawNanos @2 :UInt64;
  wallTimeNanos @3 :UInt64;
  modemUptimeMillis @4 :UInt64;
}

struct LiveMpcData {
  x @0 :List(Float32);
  y @1 :List(Float32);
  psi @2 :List(Float32);
  delta @3 :List(Float32);
  qpIterations @4 :UInt32;
  calculationTime @5 :UInt64;
  cost @6 :Float64;
}

struct LiveLongitudinalMpcData {
  xEgo @0 :List(Float32);
  vEgo @1 :List(Float32);
  aEgo @2 :List(Float32);
  xLead @3 :List(Float32);
  vLead @4 :List(Float32);
  aLead @5 :List(Float32);
  aLeadTau @6 :Float32;    # lead accel time constant
  qpIterations @7 :UInt32;
  mpcId @8 :UInt32;
  calculationTime @9 :UInt64;
  cost @10 :Float64;
}


struct ECEFPointDEPRECATED @0xe10e21168db0c7f7 {
  x @0 :Float32;
  y @1 :Float32;
  z @2 :Float32;
}

struct ECEFPoint @0xc25bbbd524983447 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

struct GPSPlannerPoints {
  curPosDEPRECATED @0 :ECEFPointDEPRECATED;
  pointsDEPRECATED @1 :List(ECEFPointDEPRECATED);
  curPos @6 :ECEFPoint;
  points @7 :List(ECEFPoint);
  valid @2 :Bool;
  trackName @3 :Text;
  speedLimit @4 :Float32;
  accelTarget @5 :Float32;
}

struct GPSPlannerPlan {
  valid @0 :Bool;
  poly @1 :List(Float32);
  trackName @2 :Text;
  speed @3 :Float32;
  acceleration @4 :Float32;
  pointsDEPRECATED @5 :List(ECEFPointDEPRECATED);
  points @6 :List(ECEFPoint);
  xLookahead @7 :Float32;
}

struct TrafficEvent @0xacfa74a094e62626 {
  type @0 :Type;
  distance @1 :Float32;
  action @2 :Action;
  resuming @3 :Bool;

  enum Type {
    stopSign @0;
    lightRed @1;
    lightYellow @2;
    lightGreen @3;
    stopLight @4;
  }

  enum Action {
    none @0;
    yield @1;
    stop @2;
    resumeReady @3;
  }

}

struct OrbslamCorrection {
  correctionMonoTime @0 :UInt64;
  prePositionECEF @1 :List(Float64);
  postPositionECEF @2 :List(Float64);
  prePoseQuatECEF @3 :List(Float32);
  postPoseQuatECEF @4 :List(Float32);
  numInliers @5 :UInt32;
}

struct OrbObservation {
  observationMonoTime @0 :UInt64;
  normalizedCoordinates @1 :List(Float32);
  locationECEF @2 :List(Float64);
  matchDistance @3: UInt32;
}

struct UiNavigationEvent {
  type @0: Type;
  status @1: Status;
  distanceTo @2: Float32;
  endRoadPointDEPRECATED @3: ECEFPointDEPRECATED;
  endRoadPoint @4: ECEFPoint;

  enum Type {
    none @0;
    laneChangeLeft @1;
    laneChangeRight @2;
    mergeLeft @3;
    mergeRight @4;
    turnLeft @5;
    turnRight @6;
  }

  enum Status {
    none @0;
    passive @1;
    approaching @2;
    active @3;
  }
}

struct UiLayoutState {
  activeApp @0 :App;
  sidebarCollapsed @1 :Bool;
  mapEnabled @2 :Bool;

  enum App {
    home @0;
    music @1;
    nav @2;
    settings @3;
  }
}

struct Joystick {
  # convenient for debug and live tuning
  axes @0: List(Float32);
  buttons @1: List(Bool);
}

struct OrbOdometry {
  # timing first
  startMonoTime @0 :UInt64;
  endMonoTime @1 :UInt64;

  # fundamental matrix and error
  f @2: List(Float64);
  err @3: Float64;

  # number of inlier points
  inliers @4: Int32;

  # for debug only
  # indexed by endMonoTime features
  # value is startMonoTime feature match
  # -1 if no match
  matches @5: List(Int16);
}

struct OrbFeatures {
  timestampEof @0 :UInt64;
  # transposed arrays of normalized image coordinates
  # len(xs) == len(ys) == len(descriptors) * 32
  xs @1 :List(Float32);
  ys @2 :List(Float32);
  descriptors @3 :Data;
  octaves @4 :List(Int8);

  # match index to last OrbFeatures
  # -1 if no match
  timestampLastEof @5 :UInt64;
  matches @6: List(Int16);
}

struct OrbFeaturesSummary {
  timestampEof @0 :UInt64;
  timestampLastEof @1 :UInt64;

  featureCount @2 :UInt16;
  matchCount @3 :UInt16;
  computeNs @4 :UInt64;
}

struct OrbKeyFrame {
  # this is a globally unique id for the KeyFrame
  id @0: UInt64;

  # this is the location of the KeyFrame
  pos @1: ECEFPoint;

  # these are the features in the world
  # len(dpos) == len(descriptors) * 32
  dpos @2 :List(ECEFPoint);
  descriptors @3 :Data;
}

struct DriverMonitoring {
  frameId @0 :UInt32;
  descriptorDEPRECATED @1 :List(Float32);
  stdDEPRECATED @2 :Float32;
  faceOrientation @3 :List(Float32);
  facePosition @4 :List(Float32);
  faceProb @5 :Float32;
  leftEyeProb @6 :Float32;
  rightEyeProb @7 :Float32;
  leftBlinkProb @8 :Float32;
  rightBlinkProb @9 :Float32;
  irPwrDEPRECATED @10 :Float32;
  faceOrientationStd @11 :List(Float32);
  facePositionStd @12 :List(Float32);
}

struct Boot {
  wallTimeNanos @0 :UInt64;
  lastKmsg @1 :Data;
  lastPmsg @2 :Data;
}

struct LiveParametersData {
  valid @0 :Bool;
  gyroBias @1 :Float32;
  angleOffset @2 :Float32;
  angleOffsetAverage @3 :Float32;
  stiffnessFactor @4 :Float32;
  steerRatio @5 :Float32;
  sensorValid @6 :Bool;
  yawRate @7 :Float32;
  posenetSpeed @8 :Float32;
  posenetValid @9 :Bool;
}

struct LiveMapData {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedAdvisoryValid @12 :Bool;
  speedAdvisory @13 :Float32;
  speedLimitAheadValid @14 :Bool;
  speedLimitAhead @15 :Float32;
  speedLimitAheadDistance @16 :Float32;
  curvatureValid @2 :Bool;
  curvature @3 :Float32;
  wayId @4 :UInt64;
  roadX @5 :List(Float32);
  roadY @6 :List(Float32);
  lastGps @7: GpsLocationData;
  roadCurvatureX @8 :List(Float32);
  roadCurvature @9 :List(Float32);
  distToTurn @10 :Float32;
  mapValid @11 :Bool;
}

struct CameraOdometry {
  frameId @4 :UInt32;
  timestampEof @5 :UInt64;
  trans @0 :List(Float32); # m/s in device frame
  rot @1 :List(Float32); # rad/s in device frame
  transStd @2 :List(Float32); # std m/s in device frame
  rotStd @3 :List(Float32); # std rad/s in device frame
}

struct KalmanOdometry {
  trans @0 :List(Float32); # m/s in device frame
  rot @1 :List(Float32); # rad/s in device frame
  transStd @2 :List(Float32); # std m/s in device frame
  rotStd @3 :List(Float32); # std rad/s in device frame
}

struct Event {
  # in nanoseconds?
  logMonoTime @0 :UInt64;
  valid @67 :Bool = true;

  union {
    initData @1 :InitData;
    frame @2 :FrameData;
    gpsNMEA @3 :GPSNMEAData;
    sensorEventDEPRECATED @4 :SensorEventData;
    can @5 :List(CanData);
    thermal @6 :ThermalData;
    controlsState @7 :ControlsState;
    liveEventDEPRECATED @8 :List(LiveEventData);
    model @9 :ModelData;
    features @10 :CalibrationFeatures;
    sensorEvents @11 :List(SensorEventData);
    health @12 :HealthData;
    radarState @13 :RadarState;
    liveUIDEPRECATED @14 :LiveUI;
    encodeIdx @15 :EncodeIndex;
    liveTracks @16 :List(LiveTracks);
    sendcan @17 :List(CanData);
    logMessage @18 :Text;
    liveCalibration @19 :LiveCalibrationData;
    androidLogEntry @20 :AndroidLogEntry;
    gpsLocation @21 :GpsLocationData;
    carState @22 :Car.CarState;
    carControl @23 :Car.CarControl;
    plan @24 :Plan;
    liveLocation @25 :LiveLocationData;
    ethernetData @26 :List(EthernetPacket);
    navUpdate @27 :NavUpdate;
    cellInfo @28 :List(CellInfo);
    wifiScan @29 :List(WifiScan);
    androidGnss @30 :AndroidGnss;
    qcomGnss @31 :QcomGnss;
    lidarPts @32 :LidarPts;
    procLog @33 :ProcLog;
    ubloxGnss @34 :UbloxGnss;
    clocks @35 :Clocks;
    liveMpc @36 :LiveMpcData;
    liveLongitudinalMpc @37 :LiveLongitudinalMpcData;
    navStatus @38 :NavStatus;
    ubloxRaw @39 :Data;
    gpsPlannerPoints @40 :GPSPlannerPoints;
    gpsPlannerPlan @41 :GPSPlannerPlan;
    applanixRaw @42 :Data;
    trafficEvents @43 :List(TrafficEvent);
    liveLocationTiming @44 :LiveLocationData;
    orbslamCorrectionDEPRECATED @45 :OrbslamCorrection;
    liveLocationCorrected @46 :LiveLocationData;
    orbObservation @47 :List(OrbObservation);
    gpsLocationExternal @48 :GpsLocationData;
    location @49 :LiveLocationData;
    uiNavigationEvent @50 :UiNavigationEvent;
    liveLocationKalman @51 :LiveLocationData;
    testJoystick @52 :Joystick;
    orbOdometry @53 :OrbOdometry;
    orbFeatures @54 :OrbFeatures;
    applanixLocation @55 :LiveLocationData;
    orbKeyFrame @56 :OrbKeyFrame;
    uiLayoutState @57 :UiLayoutState;
    orbFeaturesSummary @58 :OrbFeaturesSummary;
    driverMonitoring @59 :DriverMonitoring;
    boot @60 :Boot;
    liveParameters @61 :LiveParametersData;
    liveMapData @62 :LiveMapData;
    cameraOdometry @63 :CameraOdometry;
    pathPlan @64 :PathPlan;
    kalmanOdometry @65 :KalmanOdometry;
    thumbnail @66: Thumbnail;
    carEvents @68: List(Car.CarEvent);
    carParams @69: Car.CarParams;
    frontFrame @70: FrameData;
  }
}
