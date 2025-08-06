using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Car = import "car.capnp";
using Legacy = import "legacy.capnp";
using Custom = import "custom.capnp";

@0xf3b1f17e25a4285b;

const logVersion :Int32 = 1;

struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}

struct OnroadEvent @0xc4fa6047f024e718 {
  name @0 :EventName;

  # event types
  enable @1 :Bool;
  noEntry @2 :Bool;
  warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
  userDisable @4 :Bool;
  softDisable @5 :Bool;
  immediateDisable @6 :Bool;
  preEnable @7 :Bool;
  permanent @8 :Bool; # alerts presented regardless of openpilot state
  overrideLateral @10 :Bool;
  overrideLongitudinal @9 :Bool;

  enum EventName @0x91f1992a1f77fb03 {
    canError @0;
    steerUnavailable @1;
    wrongGear @2;
    doorOpen @3;
    seatbeltNotLatched @4;
    espDisabled @5;
    wrongCarMode @6;
    steerTempUnavailable @7;
    reverseGear @8;
    buttonCancel @9;
    buttonEnable @10;
    pedalPressed @11;  # exits active state
    preEnableStandstill @12;  # added during pre-enable state with brake
    gasPressedOverride @13;  # added when user is pressing gas with no disengage on gas
    steerOverride @14;
    steerDisengage @94;  # exits active state
    cruiseDisabled @15;
    speedTooLow @16;
    outOfSpace @17;
    overheat @18;
    calibrationIncomplete @19;
    calibrationInvalid @20;
    calibrationRecalibrating @21;
    controlsMismatch @22;
    pcmEnable @23;
    pcmDisable @24;
    radarFault @25;
    radarTempUnavailable @93;
    brakeHold @26;
    parkBrake @27;
    manualRestart @28;
    joystickDebug @29;
    longitudinalManeuver @30;
    steerTempUnavailableSilent @31;
    resumeRequired @32;
    preDriverDistracted @33;
    promptDriverDistracted @34;
    driverDistracted @35;
    preDriverUnresponsive @36;
    promptDriverUnresponsive @37;
    driverUnresponsive @38;
    belowSteerSpeed @39;
    lowBattery @40;
    accFaulted @41;
    sensorDataInvalid @42;
    commIssue @43;
    commIssueAvgFreq @44;
    tooDistracted @45;
    posenetInvalid @46;
    preLaneChangeLeft @48;
    preLaneChangeRight @49;
    laneChange @50;
    lowMemory @51;
    stockAeb @52;
    ldw @53;
    carUnrecognized @54;
    invalidLkasSetting @55;
    speedTooHigh @56;
    laneChangeBlocked @57;
    relayMalfunction @58;
    stockFcw @59;
    startup @60;
    startupNoCar @61;
    startupNoControl @62;
    startupNoSecOcKey @63;
    startupMaster @64;
    fcw @65;
    steerSaturated @66;
    belowEngageSpeed @67;
    noGps @68;
    wrongCruiseMode @69;
    modeldLagging @70;
    deviceFalling @71;
    fanMalfunction @72;
    cameraMalfunction @73;
    cameraFrameRate @74;
    processNotRunning @75;
    dashcamMode @76;
    selfdriveInitializing @77;
    usbError @78;
    cruiseMismatch @79;
    canBusMissing @80;
    selfdrivedLagging @81;
    resumeBlocked @82;
    steerTimeLimit @83;
    vehicleSensorsInvalid @84;
    locationdTemporaryError @85;
    locationdPermanentError @86;
    paramsdTemporaryError @87;
    paramsdPermanentError @88;
    actuatorsApiUnavailable @89;
    espActive @90;
    personalityChanged @91;
    aeb @92;
    userBookmark @95;
    excessiveActuation @96;
    audioFeedback @97;

    soundsUnavailableDEPRECATED @47;
  }
}

enum LongitudinalPersonality {
  aggressive @0;
  standard @1;
  relaxed @2;
}

struct InitData {
  kernelArgs @0 :List(Text);
  kernelVersion @15 :Text;
  osVersion @18 :Text;

  dongleId @2 :Text;
  bootlogId @22 :Text;

  deviceType @3 :DeviceType;
  version @4 :Text;
  gitCommit @10 :Text;
  gitCommitDate @21 :Text;
  gitBranch @11 :Text;
  gitRemote @13 :Text;

  # this is source commit for prebuilt branches
  gitSrcCommit @23 :Text;
  gitSrcCommitDate @24 :Text;

  androidProperties @16 :Map(Text, Text);

  pandaInfo @8 :PandaInfo;

  dirty @9 :Bool;
  passive @12 :Bool;
  params @17 :Map(Text, Data);

  commands @19 :Map(Text, Data);

  wallTimeNanos @20 :UInt64;

  enum DeviceType {
    unknown @0;
    neo @1;
    chffrAndroid @2;
    chffrIos @3;
    tici @4;
    pc @5;
    tizi @6;
    mici @7;
  }

  struct PandaInfo {
    hasPanda @0 :Bool;
    dongleId @1 :Text;
    stVersion @2 :Text;
    espVersion @3 :Text;
  }

  # ***** deprecated stuff *****
  gctxDEPRECATED @1 :Text;
  androidBuildInfo @5 :AndroidBuildInfo;
  androidSensorsDEPRECATED @6 :List(AndroidSensor);
  chffrAndroidExtraDEPRECATED @7 :ChffrAndroidExtra;
  iosBuildInfoDEPRECATED @14 :IosBuildInfo;

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
}

struct FrameData {
  frameId @0 :UInt32;
  frameIdSensor @25 :UInt32;
  requestId @28 :UInt32;
  encodeId @1 :UInt32;

  # Timestamps
  timestampEof @2 :UInt64;
  timestampSof @8 :UInt64;
  processingTime @23 :Float32;

  # Exposure
  integLines @4 :Int32;
  highConversionGain @20 :Bool;
  gain @15 :Float32; # This includes highConversionGain if enabled
  measuredGreyFraction @21 :Float32;
  targetGreyFraction @22 :Float32;
  exposureValPercent @27 :Float32;

  transform @10 :List(Float32);

  image @6 :Data;

  temperaturesC @24 :List(Float32);

  enum FrameTypeDEPRECATED {
    unknown @0;
    neo @1;
    chffrAndroid @2;
    front @3;
  }

  sensor @26 :ImageSensor;
  enum ImageSensor {
    unknown @0;
    ar0231 @1;
    ox03c10 @2;
    os04c10 @3;
  }

  frameLengthDEPRECATED @3 :Int32;
  globalGainDEPRECATED @5 :Int32;
  frameTypeDEPRECATED @7 :FrameTypeDEPRECATED;
  androidCaptureResultDEPRECATED @9 :AndroidCaptureResult;
  lensPosDEPRECATED @11 :Int32;
  lensSagDEPRECATED @12 :Float32;
  lensErrDEPRECATED @13 :Float32;
  lensTruePosDEPRECATED @14 :Float32;
  focusValDEPRECATED @16 :List(Int16);
  focusConfDEPRECATED @17 :List(UInt8);
  sharpnessScoreDEPRECATED @18 :List(UInt16);
  recoverStateDEPRECATED @19 :Int32;
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
  encoding @3 :Encoding;

  enum Encoding {
    unknown @0;
    jpeg @1;
    keyframe @2;
  }
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
    temperature @15: Float32;
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
    lsm6ds3 @5;   # includes LSM6DS3 and LSM6DS3TR, TR = tape reel
    bmp280 @6;    # barometer
    mmc3416x @7;  # magnetometer
    bmx055 @8;
    rpr0521 @9;
    lsm6ds3trc @10;
    mmc5603nj @11;
  }
}

# android struct GpsLocation
struct GpsLocationData {
  # Contains module-specific flags.
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
  bearingDeg @5 :Float32;

  # Represents expected horizontal accuracy in meters.
  horizontalAccuracy @6 :Float32;

  unixTimestampMillis @7 :Int64;

  source @8 :SensorSource;

  # Represents NED velocity in m/s.
  vNED @9 :List(Float32);

  # Represents expected vertical accuracy in meters. (presumably 1 sigma?)
  verticalAccuracy @10 :Float32;

  # Represents bearing accuracy in degrees. (presumably 1 sigma?)
  bearingAccuracyDeg @11 :Float32;

  # Represents velocity accuracy in m/s. (presumably 1 sigma?)
  speedAccuracy @12 :Float32;

  hasFix @13 :Bool;
  satelliteCount @14 :Int8;

  enum SensorSource {
    android @0;
    iOS @1;
    car @2;
    velodyne @3;  # Velodyne IMU
    fusion @4;
    external @5;
    ublox @6;
    trimble @7;
    qcomdiag @8;
    unicore @9;
  }
}

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

struct CanData {
  address @0 :UInt32;
  dat     @2 :Data;
  src     @3 :UInt8;
  busTimeDEPRECATED @1 :UInt16;
}

struct DeviceState @0xa4d8b5af2aa492eb {
  deviceType @45 :InitData.DeviceType;

  networkType @22 :NetworkType;
  networkInfo @31 :NetworkInfo;
  networkStrength @24 :NetworkStrength;
  networkStats @43 :NetworkStats;
  networkMetered @41 :Bool;
  lastAthenaPingTime @32 :UInt64;

  started @11 :Bool;
  startedMonoTime @13 :UInt64;

  # system utilization
  freeSpacePercent @7 :Float32;
  memoryUsagePercent @19 :Int8;
  gpuUsagePercent @33 :Int8;
  cpuUsagePercent @34 :List(Int8);  # per-core cpu usage

  # power
  offroadPowerUsageUwh @23 :UInt32;
  carBatteryCapacityUwh @25 :UInt32;
  powerDrawW @40 :Float32;
  somPowerDrawW @42 :Float32;

  # device thermals
  cpuTempC @26 :List(Float32);
  gpuTempC @27 :List(Float32);
  dspTempC @49 :Float32;
  memoryTempC @28 :Float32;
  modemTempC @36 :List(Float32);
  pmicTempC @39 :List(Float32);
  intakeTempC @46 :Float32;
  exhaustTempC @47 :Float32;
  caseTempC @48 :Float32;
  maxTempC @44 :Float32;  # max of other temps, used to control fan
  thermalZones @38 :List(ThermalZone);
  thermalStatus @14 :ThermalStatus;

  fanSpeedPercentDesired @10 :UInt16;
  screenBrightnessPercent @37 :Int8;

  struct ThermalZone {
    name @0 :Text;
    temp @1 :Float32;
  }

  enum ThermalStatus {
    green @0;
    yellow @1;
    red @2;
    danger @3;
  }

  enum NetworkType {
    none @0;
    wifi @1;
    cell2G @2;
    cell3G @3;
    cell4G @4;
    cell5G @5;
    ethernet @6;
  }

  enum NetworkStrength {
    unknown @0;
    poor @1;
    moderate @2;
    good @3;
    great @4;
  }

  struct NetworkInfo {
    technology @0 :Text;
    operator @1 :Text;
    band @2 :Text;
    channel @3 :UInt16;
    extra @4 :Text;
    state @5 :Text;
  }

  struct NetworkStats {
    wwanTx @0 :Int64;
    wwanRx @1 :Int64;
  }

  # deprecated
  cpu0DEPRECATED @0 :UInt16;
  cpu1DEPRECATED @1 :UInt16;
  cpu2DEPRECATED @2 :UInt16;
  cpu3DEPRECATED @3 :UInt16;
  memDEPRECATED @4 :UInt16;
  gpuDEPRECATED @5 :UInt16;
  batDEPRECATED @6 :UInt32;
  pa0DEPRECATED @21 :UInt16;
  cpuUsagePercentDEPRECATED @20 :Int8;
  batteryStatusDEPRECATED @9 :Text;
  batteryVoltageDEPRECATED @16 :Int32;
  batteryTempCDEPRECATED @29 :Float32;
  batteryPercentDEPRECATED @8 :Int16;
  batteryCurrentDEPRECATED @15 :Int32;
  chargingErrorDEPRECATED @17 :Bool;
  chargingDisabledDEPRECATED @18 :Bool;
  usbOnlineDEPRECATED @12 :Bool;
  ambientTempCDEPRECATED @30 :Float32;
  nvmeTempCDEPRECATED @35 :List(Float32);
}

struct PandaState @0xa7649e2575e4591e {
  ignitionLine @2 :Bool;
  rxBufferOverflow @7 :UInt32;
  txBufferOverflow @8 :UInt32;
  pandaType @10 :PandaType;
  ignitionCan @13 :Bool;
  faultStatus @15 :FaultStatus;
  powerSaveEnabled @16 :Bool;
  uptime @17 :UInt32;
  faults @18 :List(FaultType);
  heartbeatLost @22 :Bool;
  interruptLoad @25 :Float32;
  fanPower @28 :UInt8;
  fanStallCount @34 :UInt8;

  spiErrorCount @33 :UInt16;

  harnessStatus @21 :HarnessStatus;
  sbu1Voltage @35 :Float32;
  sbu2Voltage @36 :Float32;

  # can health
  canState0 @29 :PandaCanState;
  canState1 @30 :PandaCanState;
  canState2 @31 :PandaCanState;

  # safety stuff
  controlsAllowed @3 :Bool;
  safetyRxInvalid @19 :UInt32;
  safetyTxBlocked @24 :UInt32;
  safetyModel @14 :Car.CarParams.SafetyModel;
  safetyParam @27 :UInt16;
  alternativeExperience @23 :Int16;
  safetyRxChecksInvalid @32 :Bool;

  voltage @0 :UInt32;
  current @1 :UInt32;

  enum FaultStatus {
    none @0;
    faultTemp @1;
    faultPerm @2;
  }

  enum FaultType {
    relayMalfunction @0;
    unusedInterruptHandled @1;
    interruptRateCan1 @2;
    interruptRateCan2 @3;
    interruptRateCan3 @4;
    interruptRateTach @5;
    interruptRateGmlanDEPRECATED @6;
    interruptRateInterrupts @7;
    interruptRateSpiDma @8;
    interruptRateSpiCs @9;
    interruptRateUart1 @10;
    interruptRateUart2 @11;
    interruptRateUart3 @12;
    interruptRateUart5 @13;
    interruptRateUartDma @14;
    interruptRateUsb @15;
    interruptRateTim1 @16;
    interruptRateTim3 @17;
    registerDivergent @18;
    interruptRateKlineInit @19;
    interruptRateClockSource @20;
    interruptRateTick @21;
    interruptRateExti @22;
    interruptRateSpi @23;
    interruptRateUart7 @24;
    sirenMalfunction @25;
    heartbeatLoopWatchdog @26;
    # Update max fault type in boardd when adding faults
  }

  enum PandaType @0x8a58adf93e5b3751 {
    unknown @0;
    whitePanda @1;
    greyPanda @2;
    blackPanda @3;
    pedal @4;
    uno @5;
    dos @6;
    redPanda @7;
    redPandaV2 @8;
    tres @9;
    cuatro @10;
  }

  enum HarnessStatus {
    notConnected @0;
    normal @1;
    flipped @2;
  }

  struct PandaCanState {
    busOff @0 :Bool;
    busOffCnt @1 :UInt32;
    errorWarning @2 :Bool;
    errorPassive @3 :Bool;
    lastError @4 :LecErrorCode;
    lastStoredError @5 :LecErrorCode;
    lastDataError @6 :LecErrorCode;
    lastDataStoredError @7 :LecErrorCode;
    receiveErrorCnt @8 :UInt8;
    transmitErrorCnt @9 :UInt8;
    totalErrorCnt @10 :UInt32;
    totalTxLostCnt @11 :UInt32;
    totalRxLostCnt @12 :UInt32;
    totalTxCnt @13 :UInt32;
    totalRxCnt @14 :UInt32;
    totalFwdCnt @15 :UInt32;
    canSpeed @16 :UInt16;
    canDataSpeed @17 :UInt16;
    canfdEnabled @18 :Bool;
    brsEnabled @19 :Bool;
    canfdNonIso @20 :Bool;
    irq0CallRate @21 :UInt32;
    irq1CallRate @22 :UInt32;
    irq2CallRate @23 :UInt32;
    canCoreResetCnt @24 :UInt32;

    enum LecErrorCode {
      noError @0;
      stuffError @1;
      formError @2;
      ackError @3;
      bit1Error @4;
      bit0Error @5;
      crcError @6;
      noChange @7;
    }
  }

  gasInterceptorDetectedDEPRECATED @4 :Bool;
  startedSignalDetectedDEPRECATED @5 :Bool;
  hasGpsDEPRECATED @6 :Bool;
  gmlanSendErrsDEPRECATED @9 :UInt32;
  fanSpeedRpmDEPRECATED @11 :UInt16;
  usbPowerModeDEPRECATED @12 :PeripheralState.UsbPowerModeDEPRECATED;
  safetyParamDEPRECATED @20 :Int16;
  safetyParam2DEPRECATED @26 :UInt32;
}

struct PeripheralState {
  pandaType @0 :PandaState.PandaType;
  voltage @1 :UInt32;
  current @2 :UInt32;
  fanSpeedRpm @3 :UInt16;

  usbPowerModeDEPRECATED @4 :UsbPowerModeDEPRECATED;
  enum UsbPowerModeDEPRECATED @0xa8883583b32c9877 {
    none @0;
    client @1;
    cdp @2;
    dcp @3;
  }
}

struct RadarState @0x9a185389d6fdd05f {
  mdMonoTime @6 :UInt64;
  carStateMonoTime @11 :UInt64;
  radarErrors @13 :Car.RadarData.Error;

  leadOne @3 :LeadData;
  leadTwo @4 :LeadData;

  struct LeadData {
    dRel @0 :Float32;
    yRel @1 :Float32;
    vRel @2 :Float32;
    aRel @3 :Float32;
    vLead @4 :Float32;
    dPath @6 :Float32;
    vLat @7 :Float32;
    vLeadK @8 :Float32;
    aLeadK @9 :Float32;
    fcw @10 :Bool;
    status @11 :Bool;
    aLeadTau @12 :Float32;
    modelProb @13 :Float32;
    radar @14 :Bool;
    radarTrackId @15 :Int32 = -1;

    aLeadDEPRECATED @5 :Float32;
  }

  # deprecated
  ftMonoTimeDEPRECATED @7 :UInt64;
  warpMatrixDEPRECATED @0 :List(Float32);
  angleOffsetDEPRECATED @1 :Float32;
  calStatusDEPRECATED @2 :Int8;
  calCycleDEPRECATED @8 :Int32;
  calPercDEPRECATED @9 :Int8;
  canMonoTimesDEPRECATED @10 :List(UInt64);
  cumLagMsDEPRECATED @5 :Float32;
  radarErrorsDEPRECATED @12 :List(Car.RadarData.ErrorDEPRECATED);
}

struct LiveCalibrationData {
  calStatus @11 :Status;
  calCycle @2 :Int32;
  calPerc @3 :Int8;
  validBlocks @9 :Int32;

  # view_frame_from_road_frame
  # ui's is inversed needs new
  extrinsicMatrix @4 :List(Float32);
  # the direction of travel vector in device frame
  rpyCalib @7 :List(Float32);
  rpyCalibSpread @8 :List(Float32);
  wideFromDeviceEuler @10 :List(Float32);
  height @12 :List(Float32);

  warpMatrixDEPRECATED @0 :List(Float32);
  calStatusDEPRECATED @1 :Int8;
  warpMatrix2DEPRECATED @5 :List(Float32);
  warpMatrixBigDEPRECATED @6 :List(Float32);

  enum Status {
    uncalibrated @0;
    calibrated @1;
    invalid @2;
    recalibrating @3;
  }
}

struct LiveTracksDEPRECATED {
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

struct SelfdriveState {
  # high level system state
  state @0 :OpenpilotState;
  enabled @1 :Bool;
  active @2 :Bool;
  engageable @9 :Bool;  # can OP be engaged?

  # UI alerts
  alertText1 @3 :Text;
  alertText2 @4 :Text;
  alertStatus @5 :AlertStatus;
  alertSize @6 :AlertSize;
  alertType @7 :Text;
  alertSound @8 :Car.CarControl.HUDControl.AudibleAlert;
  alertHudVisual @12 :Car.CarControl.HUDControl.VisualAlert;

  # configurable driving settings
  experimentalMode @10 :Bool;
  personality @11 :LongitudinalPersonality;

  enum OpenpilotState @0xdbe58b96d2d1ac61 {
    disabled @0;
    preEnabled @1;
    enabled @2;
    softDisabling @3;
    overriding @4;  # superset of overriding with steering or accelerator
  }

  enum AlertStatus @0xa0d0dcd113193c62 {
    normal @0;
    userPrompt @1;
    critical @2;
  }

  enum AlertSize @0xe98bb99d6e985f64 {
    none @0;
    small @1;
    mid @2;
    full @3;
  }
}

struct ControlsState @0x97ff69c53601abf1 {
  longitudinalPlanMonoTime @28 :UInt64;
  lateralPlanMonoTime @50 :UInt64;

  longControlState @30 :Car.CarControl.Actuators.LongControlState;
  upAccelCmd @4 :Float32;
  uiAccelCmd @5 :Float32;
  ufAccelCmd @33 :Float32;
  curvature @37 :Float32;  # path curvature from vehicle model
  desiredCurvature @61 :Float32;  # lag adjusted curvatures used by lateral controllers
  forceDecel @51 :Bool;

  lateralControlState :union {
    pidState @53 :LateralPIDState;
    angleState @58 :LateralAngleState;
    debugState @59 :LateralDebugState;
    torqueState @60 :LateralTorqueState;

    curvatureStateDEPRECATED @65 :LateralCurvatureState;
    lqrStateDEPRECATED @55 :LateralLQRState;
    indiStateDEPRECATED @52 :LateralINDIState;
  }

  struct LateralINDIState {
    active @0 :Bool;
    steeringAngleDeg @1 :Float32;
    steeringRateDeg @2 :Float32;
    steeringAccelDeg @3 :Float32;
    rateSetPoint @4 :Float32;
    accelSetPoint @5 :Float32;
    accelError @6 :Float32;
    delayedOutput @7 :Float32;
    delta @8 :Float32;
    output @9 :Float32;
    saturated @10 :Bool;
    steeringAngleDesiredDeg @11 :Float32;
    steeringRateDesiredDeg @12 :Float32;
  }

  struct LateralPIDState {
    active @0 :Bool;
    steeringAngleDeg @1 :Float32;
    steeringRateDeg @2 :Float32;
    angleError @3 :Float32;
    p @4 :Float32;
    i @5 :Float32;
    f @6 :Float32;
    output @7 :Float32;
    saturated @8 :Bool;
    steeringAngleDesiredDeg @9 :Float32;
   }

  struct LateralTorqueState {
    active @0 :Bool;
    error @1 :Float32;
    errorRate @8 :Float32;
    p @2 :Float32;
    i @3 :Float32;
    d @4 :Float32;
    f @5 :Float32;
    output @6 :Float32;
    saturated @7 :Bool;
    actualLateralAccel @9 :Float32;
    desiredLateralAccel @10 :Float32;
   }

  struct LateralLQRState {
    active @0 :Bool;
    steeringAngleDeg @1 :Float32;
    i @2 :Float32;
    output @3 :Float32;
    lqrOutput @4 :Float32;
    saturated @5 :Bool;
    steeringAngleDesiredDeg @6 :Float32;
  }

  struct LateralAngleState {
    active @0 :Bool;
    steeringAngleDeg @1 :Float32;
    output @2 :Float32;
    saturated @3 :Bool;
    steeringAngleDesiredDeg @4 :Float32;
  }

  struct LateralCurvatureState {
    active @0 :Bool;
    actualCurvature @1 :Float32;
    desiredCurvature @2 :Float32;
    error @3 :Float32;
    p @4 :Float32;
    i @5 :Float32;
    f @6 :Float32;
    output @7 :Float32;
    saturated @8 :Bool;
  }

  struct LateralDebugState {
    active @0 :Bool;
    steeringAngleDeg @1 :Float32;
    output @2 :Float32;
    saturated @3 :Bool;
  }

  # deprecated
  vEgoDEPRECATED @0 :Float32;
  vEgoRawDEPRECATED @32 :Float32;
  aEgoDEPRECATED @1 :Float32;
  canMonoTimeDEPRECATED @16 :UInt64;
  radarStateMonoTimeDEPRECATED @17 :UInt64;
  mdMonoTimeDEPRECATED @18 :UInt64;
  yActualDEPRECATED @6 :Float32;
  yDesDEPRECATED @7 :Float32;
  upSteerDEPRECATED @8 :Float32;
  uiSteerDEPRECATED @9 :Float32;
  ufSteerDEPRECATED @34 :Float32;
  aTargetMinDEPRECATED @10 :Float32;
  aTargetMaxDEPRECATED @11 :Float32;
  rearViewCamDEPRECATED @23 :Bool;
  driverMonitoringOnDEPRECATED @43 :Bool;
  hudLeadDEPRECATED @14 :Int32;
  alertSoundDEPRECATED @45 :Text;
  angleModelBiasDEPRECATED @27 :Float32;
  gpsPlannerActiveDEPRECATED @40 :Bool;
  decelForTurnDEPRECATED @47 :Bool;
  decelForModelDEPRECATED @54 :Bool;
  awarenessStatusDEPRECATED @26 :Float32;
  angleSteersDEPRECATED @13 :Float32;
  vCurvatureDEPRECATED @46 :Float32;
  mapValidDEPRECATED @49 :Bool;
  jerkFactorDEPRECATED @12 :Float32;
  steerOverrideDEPRECATED @20 :Bool;
  steeringAngleDesiredDegDEPRECATED @29 :Float32;
  canMonoTimesDEPRECATED @21 :List(UInt64);
  desiredCurvatureRateDEPRECATED @62 :Float32;
  canErrorCounterDEPRECATED @57 :UInt32;
  vPidDEPRECATED @2 :Float32;
  alertBlinkingRateDEPRECATED @42 :Float32;
  alertText1DEPRECATED @24 :Text;
  alertText2DEPRECATED @25 :Text;
  alertStatusDEPRECATED @38 :SelfdriveState.AlertStatus;
  alertSizeDEPRECATED @39 :SelfdriveState.AlertSize;
  alertTypeDEPRECATED @44 :Text;
  alertSound2DEPRECATED @56 :Car.CarControl.HUDControl.AudibleAlert;
  engageableDEPRECATED @41 :Bool;  # can OP be engaged?
  stateDEPRECATED @31 :SelfdriveState.OpenpilotState;
  enabledDEPRECATED @19 :Bool;
  activeDEPRECATED @36 :Bool;
  experimentalModeDEPRECATED @64 :Bool;
  personalityDEPRECATED @66 :LongitudinalPersonality;
  vCruiseDEPRECATED @22 :Float32;  # actual set speed
  vCruiseClusterDEPRECATED @63 :Float32;  # set speed to display in the UI
  startMonoTimeDEPRECATED @48 :UInt64;
  cumLagMsDEPRECATED @15 :Float32;
  aTargetDEPRECATED @35 :Float32;
  vTargetLeadDEPRECATED @3 :Float32;
}

struct DrivingModelData {
  frameId @0 :UInt32;
  frameIdExtra @1 :UInt32;
  frameDropPerc @6 :Float32;
  modelExecutionTime @7 :Float32;

  action @2 :ModelDataV2.Action;

  laneLineMeta @3 :LaneLineMeta;
  meta @4 :MetaData;

  path @5 :PolyPath;

  struct PolyPath {
    xCoefficients @0 :List(Float32);
    yCoefficients @1 :List(Float32);
    zCoefficients @2 :List(Float32);
  }

  struct LaneLineMeta {
    leftY @0 :Float32;
    rightY @1 :Float32;
    leftProb @2 :Float32;
    rightProb @3 :Float32;
  }

  struct MetaData {
    laneChangeState @0 :LaneChangeState;
    laneChangeDirection @1 :LaneChangeDirection;
  }
}

# All SI units and in device frame
struct XYZTData @0xc3cbae1fd505ae80 {
  x @0 :List(Float32);
  y @1 :List(Float32);
  z @2 :List(Float32);
  t @3 :List(Float32);
  xStd @4 :List(Float32);
  yStd @5 :List(Float32);
  zStd @6 :List(Float32);
}

struct ModelDataV2 {
  frameId @0 :UInt32;
  frameIdExtra @20 :UInt32;
  frameAge @1 :UInt32;
  frameDropPerc @2 :Float32;
  timestampEof @3 :UInt64;
  modelExecutionTime @15 :Float32;
  rawPredictions @16 :Data;

  # predicted future position, orientation, etc..
  position @4 :XYZTData;
  orientation @5 :XYZTData;
  velocity @6 :XYZTData;
  orientationRate @7 :XYZTData;
  acceleration @19 :XYZTData;

  # prediction lanelines and road edges
  laneLines @8 :List(XYZTData);
  laneLineProbs @9 :List(Float32);
  laneLineStds @13 :List(Float32);
  roadEdges @10 :List(XYZTData);
  roadEdgeStds @14 :List(Float32);

  # predicted lead cars
  leads @11 :List(LeadDataV2);
  leadsV3 @18 :List(LeadDataV3);

  meta @12 :MetaData;
  confidence @23: ConfidenceClass;

  # Model perceived motion
  temporalPoseDEPRECATED @21 :Pose;

  # e2e lateral planner
  action @26: Action;

  gpuExecutionTimeDEPRECATED @17 :Float32;
  navEnabledDEPRECATED @22 :Bool;
  locationMonoTimeDEPRECATED @24 :UInt64;
  lateralPlannerSolutionDEPRECATED @25: LateralPlannerSolution;

  struct LeadDataV2 {
    prob @0 :Float32; # probability that car is your lead at time t
    t @1 :Float32;

    # x and y are relative position in device frame
    # v is norm relative speed
    # a is norm relative acceleration
    xyva @2 :List(Float32);
    xyvaStd @3 :List(Float32);
  }

  struct LeadDataV3 {
    prob @0 :Float32; # probability that car is your lead at time t
    probTime @1 :Float32;
    t @2 :List(Float32);

    # x and y are relative position in device frame
    # v absolute norm speed
    # a is derivative of v
    x @3 :List(Float32);
    xStd @4 :List(Float32);
    y @5 :List(Float32);
    yStd @6 :List(Float32);
    v @7 :List(Float32);
    vStd @8 :List(Float32);
    a @9 :List(Float32);
    aStd @10 :List(Float32);
  }


  struct MetaData {
    engagedProb @0 :Float32;
    desirePrediction @1 :List(Float32);
    desireState @5 :List(Float32);
    disengagePredictions @6 :DisengagePredictions;
    hardBrakePredicted @7 :Bool;
    laneChangeState @8 :LaneChangeState;
    laneChangeDirection @9 :LaneChangeDirection;


    # deprecated
    brakeDisengageProbDEPRECATED @2 :Float32;
    gasDisengageProbDEPRECATED @3 :Float32;
    steerOverrideProbDEPRECATED @4 :Float32;
  }

  enum ConfidenceClass {
    red @0;
    yellow @1;
    green @2;
  }

  struct DisengagePredictions {
    t @0 :List(Float32);
    brakeDisengageProbs @1 :List(Float32);
    gasDisengageProbs @2 :List(Float32);
    steerOverrideProbs @3 :List(Float32);
    brake3MetersPerSecondSquaredProbs @4 :List(Float32);
    brake4MetersPerSecondSquaredProbs @5 :List(Float32);
    brake5MetersPerSecondSquaredProbs @6 :List(Float32);
    gasPressProbs @7 :List(Float32);
    brakePressProbs @8 :List(Float32);
  }

  struct Pose {
    trans @0 :List(Float32); # m/s in device frame
    rot @1 :List(Float32); # rad/s in device frame
    transStd @2 :List(Float32); # std m/s in device frame
    rotStd @3 :List(Float32); # std rad/s in device frame
  }

  struct LateralPlannerSolution {
    x @0 :List(Float32);
    y @1 :List(Float32);
    yaw @2 :List(Float32);
    yawRate @3 :List(Float32);
    xStd @4 :List(Float32);
    yStd @5 :List(Float32);
    yawStd @6 :List(Float32);
    yawRateStd @7 :List(Float32);
  }

  struct Action {
    desiredCurvature @0 :Float32;
    desiredAcceleration @1 :Float32;
    shouldStop @2 :Bool;
  }
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
  timestampSof @6 :UInt64;
  timestampEof @7 :UInt64;

  # encoder metadata
  flags @8 :UInt32;
  len @9 :UInt32;

  enum Type {
    bigBoxLossless @0;
    fullHEVC @1;
    qcameraH264 @6;
    livestreamH264 @7;

    # deprecated
    bigBoxHEVCDEPRECATED @2;
    chffrAndroidH264DEPRECATED @3;
    fullLosslessClipDEPRECATED @4;
    frontDEPRECATED @5;

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

struct DriverAssistance {
  # Lane Departure Warnings
  leftLaneDeparture @0 :Bool;
  rightLaneDeparture @1 :Bool;

  # FCW, AEB, etc. will go here
}

struct LongitudinalPlan @0xe00b5b3eba12876c {
  modelMonoTime @9 :UInt64;
  hasLead @7 :Bool;
  fcw @8 :Bool;
  longitudinalPlanSource @15 :LongitudinalPlanSource;
  processingDelay @29 :Float32;

  # desired speed/accel/jerk over next 2.5s
  accels @32 :List(Float32);
  speeds @33 :List(Float32);
  jerks @34 :List(Float32);
  aTarget @18 :Float32;
  shouldStop @37: Bool;
  allowThrottle @38: Bool;
  allowBrake @39: Bool;


  solverExecutionTime @35 :Float32;

  enum LongitudinalPlanSource {
    cruise @0;
    lead0 @1;
    lead1 @2;
    lead2 @3;
    e2e @4;
  }

  # deprecated
  vCruiseDEPRECATED @16 :Float32;
  aCruiseDEPRECATED @17 :Float32;
  vTargetDEPRECATED @3 :Float32;
  vTargetFutureDEPRECATED @14 :Float32;
  vStartDEPRECATED @26 :Float32;
  aStartDEPRECATED @27 :Float32;
  vMaxDEPRECATED @20 :Float32;
  radarStateMonoTimeDEPRECATED @10 :UInt64;
  jerkFactorDEPRECATED @6 :Float32;
  hasLeftLaneDEPRECATED @23 :Bool;
  hasRightLaneDEPRECATED @24 :Bool;
  aTargetMinDEPRECATED @4 :Float32;
  aTargetMaxDEPRECATED @5 :Float32;
  lateralValidDEPRECATED @0 :Bool;
  longitudinalValidDEPRECATED @2 :Bool;
  dPolyDEPRECATED @1 :List(Float32);
  laneWidthDEPRECATED @11 :Float32;
  vCurvatureDEPRECATED @21 :Float32;
  decelForTurnDEPRECATED @22 :Bool;
  mapValidDEPRECATED @25 :Bool;
  radarValidDEPRECATED @28 :Bool;
  radarCanErrorDEPRECATED @30 :Bool;
  commIssueDEPRECATED @31 :Bool;
  eventsDEPRECATED @13 :List(Car.OnroadEventDEPRECATED);
  gpsTrajectoryDEPRECATED @12 :GpsTrajectory;
  gpsPlannerActiveDEPRECATED @19 :Bool;
  personalityDEPRECATED @36 :LongitudinalPersonality;

  struct GpsTrajectory {
    x @0 :List(Float32);
    y @1 :List(Float32);
  }
}
struct UiPlan {
  frameId @2 :UInt32;
  position @0 :XYZTData;
  accel @1 :List(Float32);
}

struct LateralPlan @0xe1e9318e2ae8b51e {
  modelMonoTime @31 :UInt64;
  laneWidthDEPRECATED @0 :Float32;
  lProbDEPRECATED @5 :Float32;
  rProbDEPRECATED @7 :Float32;
  dPathPoints @20 :List(Float32);
  dProbDEPRECATED @21 :Float32;

  mpcSolutionValid @9 :Bool;
  desire @17 :Desire;
  laneChangeState @18 :LaneChangeState;
  laneChangeDirection @19 :LaneChangeDirection;
  useLaneLines @29 :Bool;

  # desired curvatures over next 2.5s in rad/m
  psis @26 :List(Float32);
  curvatures @27 :List(Float32);
  curvatureRates @28 :List(Float32);

  solverExecutionTime @30 :Float32;
  solverCost @32 :Float32;
  solverState @33 :SolverState;

  struct SolverState {
    x @0 :List(List(Float32));
    u @1 :List(Float32);
  }

  # deprecated
  curvatureDEPRECATED @22 :Float32;
  curvatureRateDEPRECATED @23 :Float32;
  rawCurvatureDEPRECATED @24 :Float32;
  rawCurvatureRateDEPRECATED @25 :Float32;
  cProbDEPRECATED @3 :Float32;
  dPolyDEPRECATED @1 :List(Float32);
  cPolyDEPRECATED @2 :List(Float32);
  lPolyDEPRECATED @4 :List(Float32);
  rPolyDEPRECATED @6 :List(Float32);
  modelValidDEPRECATED @12 :Bool;
  commIssueDEPRECATED @15 :Bool;
  posenetValidDEPRECATED @16 :Bool;
  sensorValidDEPRECATED @14 :Bool;
  paramsValidDEPRECATED @10 :Bool;
  steeringAngleDegDEPRECATED @8 :Float32; # deg
  steeringRateDegDEPRECATED @13 :Float32; # deg/s
  angleOffsetDegDEPRECATED @11 :Float32;
}

struct LiveLocationKalman {

  # More info on reference frames:
  # https://github.com/commaai/openpilot/tree/master/common/transformations

  positionECEF @0 : Measurement;
  positionGeodetic @1 : Measurement;
  velocityECEF @2 : Measurement;
  velocityNED @3 : Measurement;
  velocityDevice @4 : Measurement;
  accelerationDevice @5: Measurement;


  # These angles are all eulers and roll, pitch, yaw
  # orientationECEF transforms to rot matrix: ecef_from_device
  orientationECEF @6 : Measurement;
  calibratedOrientationECEF @20 : Measurement;
  orientationNED @7 : Measurement;
  angularVelocityDevice @8 : Measurement;

  # orientationNEDCalibrated transforms to rot matrix: NED_from_calibrated
  calibratedOrientationNED @9 : Measurement;

  # Calibrated frame is simply device frame
  # aligned with the vehicle
  velocityCalibrated @10 : Measurement;
  accelerationCalibrated @11 : Measurement;
  angularVelocityCalibrated @12 : Measurement;

  gpsWeek @13 :Int32;
  gpsTimeOfWeek @14 :Float64;
  status @15 :Status;
  unixTimestampMillis @16 :Int64;
  inputsOK @17 :Bool = true;
  posenetOK @18 :Bool = true;
  gpsOK @19 :Bool = true;
  sensorsOK @21 :Bool = true;
  deviceStable @22 :Bool = true;
  timeSinceReset @23 :Float64;
  excessiveResets @24 :Bool;
  timeToFirstFix @25 :Float32;

  filterState @26 : Measurement;

  enum Status {
    uninitialized @0;
    uncalibrated @1;
    valid @2;
  }

  struct Measurement {
    value @0 : List(Float64);
    std @1 : List(Float64);
    valid @2 : Bool;
  }
}


struct LivePose {
  # More info on reference frames:
  # https://github.com/commaai/openpilot/tree/master/common/transformations
  orientationNED @0 :XYZMeasurement;
  velocityDevice @1 :XYZMeasurement;
  accelerationDevice @2 :XYZMeasurement;
  angularVelocityDevice @3 :XYZMeasurement;

  inputsOK @4 :Bool = false;
  posenetOK @5 :Bool = false;
  sensorsOK @6 :Bool = false;

  debugFilterState @7 :FilterState;

  struct XYZMeasurement {
    x @0 :Float32;
    y @1 :Float32;
    z @2 :Float32;
    xStd @3 :Float32;
    yStd @4 :Float32;
    zStd @5 :Float32;
    valid @6 :Bool;
  }

  struct FilterState {
    value @0 : List(Float64);
    std @1 : List(Float64);
    valid @2 : Bool;

    observations @3 :List(Observation);

    struct Observation {
      kind @0 :Int32;
      value @1 :List(Float32);
      error @2 :List(Float32);
    }
  }
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

struct GnssMeasurements {
  measTime @0 :UInt64;
  gpsWeek @1 :Int16;
  gpsTimeOfWeek @2 :Float64;

  correctedMeasurements @3 :List(CorrectedMeasurement);
  ephemerisStatuses @9 :List(EphemerisStatus);

  kalmanPositionECEF @4 :LiveLocationKalman.Measurement;
  kalmanVelocityECEF @5 :LiveLocationKalman.Measurement;
  positionECEF @6 :LiveLocationKalman.Measurement;
  velocityECEF @7 :LiveLocationKalman.Measurement;
  timeToFirstFix @8 :Float32;
  # Todo sync this with timing pulse of ublox

  struct EphemerisStatus {
    constellationId @0 :ConstellationId;
    svId @1 :UInt8;
    type @2 :EphemerisType;
    source @3 :EphemerisSource;
    gpsWeek @4 : UInt16;
    tow @5 :Float64;
  }

  struct CorrectedMeasurement {
    constellationId @0 :ConstellationId;
    svId @1 :UInt8;
    # Is 0 when not Glonass constellation.
    glonassFrequency @2 :Int8;
    pseudorange @3 :Float64;
    pseudorangeStd @4 :Float64;
    pseudorangeRate @5 :Float64;
    pseudorangeRateStd @6 :Float64;
    # Satellite position and velocity [x,y,z]
    satPos @7 :List(Float64);
    satVel @8 :List(Float64);
    ephemerisSourceDEPRECATED @9 :EphemerisSourceDEPRECATED;
  }

  struct EphemerisSourceDEPRECATED {
    type @0 :EphemerisType;
    # first epoch in file:
    gpsWeek @1 :Int16; # -1 if Nav
    gpsTimeOfWeek @2 :Int32; # -1 if Nav. Integer for seconds is good enough for logs.
  }

  enum ConstellationId {
    # Satellite Constellation using the Ublox gnssid as index
    gps @0;
    sbas @1;
    galileo @2;
    beidou @3;
    imes @4;
    qznss @5;
    glonass @6;
  }

  enum EphemerisType {
    nav @0;
    # Different ultra-rapid files:
    nasaUltraRapid @1;
    glonassIacUltraRapid @2;
    qcom @3;
  }

  enum EphemerisSource {
    gnssChip @0;
    internet @1;
    cache @2;
    unknown @3;
  }
}

struct UbloxGnss {
  union {
    measurementReport @0 :MeasurementReport;
    ephemeris @1 :Ephemeris;
    ionoData @2 :IonoData;
    hwStatus @3 :HwStatus;
    hwStatus2 @4 :HwStatus2;
    glonassEphemeris @5 :GlonassEphemeris;
    satReport @6 :SatReport;
  }

  struct SatReport {
    #received time of week in gps time in seconds and gps week
    iTow @0 :UInt32;
    svs @1 :List(SatInfo);

    struct SatInfo {
      svId @0 :UInt8;
      gnssId @1 :UInt8;
      flagsBitfield @2 :UInt32;
      cno @3 :UInt8;
      elevationDeg @4 :Int8;
      azimuthDeg @5 :Int16;
      pseudorangeResidual @6 :Float32;
    }
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
        # half cycle subtracted from phase
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

  struct IonoData {
    svHealth @0 :UInt32;
    tow  @1 :Float64;
    gpsWeek @2 :Float64;

    ionoAlpha @3 :List(Float64);
    ionoBeta @4 :List(Float64);

    healthValid @5 :Bool;
    ionoCoeffsValid @6 :Bool;
  }

  struct HwStatus {
    noisePerMS @0 :UInt16;
    agcCnt @1 :UInt16;
    aStatus @2 :AntennaSupervisorState;
    aPower @3 :AntennaPowerStatus;
    jamInd @4 :UInt8;
    flags @5 :UInt8;

    enum AntennaSupervisorState {
      init @0;
      dontknow @1;
      ok @2;
      short @3;
      open @4;
    }

    enum AntennaPowerStatus {
      off @0;
      on @1;
      dontknow @2;
    }
  }

  struct HwStatus2 {
    ofsI @0 :Int8;
    magI @1 :UInt8;
    ofsQ @2 :Int8;
    magQ @3 :UInt8;
    cfgSource @4 :ConfigSource;
    lowLevCfg @5 :UInt32;
    postStatus @6 :UInt32;

    enum ConfigSource {
      undefined @0;
      rom @1;
      otp @2;
      configpins @3;
      flash @4;
    }
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
}

struct QcomGnss @0xde94674b07ae51c1 {
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
    unknown3 @3;
    unknown4 @4;
    unknown5 @5;
    sbas @6;
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

  struct MeasurementReport @0xf580d7d86b7b8692 {
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

    struct SV @0xf10c595ae7bb2c27 {
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

  struct ClockReport @0xca965e4add8f4f0b {
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

  struct DrMeasurementReport @0x8053c39445c6c75c {

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

    struct SV @0xf08b81df8cbf459c {
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

  struct DrSvPolyReport @0xb1fb80811a673270 {
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

    gpsWeek @23 :UInt16;
    gpsTow @24 :Float64;
  }
}

struct Clocks {
  wallTimeNanos @3 :UInt64;  # unix epoch time

  bootTimeNanosDEPRECATED @0 :UInt64;
  monotonicNanosDEPRECATED @1 :UInt64;
  monotonicRawNanosDEPRECATD @2 :UInt64;
  modemUptimeMillisDEPRECATED @4 :UInt64;
}

struct LiveMpcData {
  x @0 :List(Float32);
  y @1 :List(Float32);
  psi @2 :List(Float32);
  curvature @3 :List(Float32);
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

struct Joystick {
  # convenient for debug and live tuning
  axes @0: List(Float32);
  buttons @1: List(Bool);
}

struct DriverStateV2 {
  frameId @0 :UInt32;
  modelExecutionTime @1 :Float32;
  dspExecutionTimeDEPRECATED @2 :Float32;
  gpuExecutionTime @8 :Float32;
  rawPredictions @3 :Data;

  poorVisionProb @4 :Float32;
  wheelOnRightProb @5 :Float32;

  leftDriverData @6 :DriverData;
  rightDriverData @7 :DriverData;

  struct DriverData {
    faceOrientation @0 :List(Float32);
    faceOrientationStd @1 :List(Float32);
    facePosition @2 :List(Float32);
    facePositionStd @3 :List(Float32);
    faceProb @4 :Float32;
    leftEyeProb @5 :Float32;
    rightEyeProb @6 :Float32;
    leftBlinkProb @7 :Float32;
    rightBlinkProb @8 :Float32;
    sunglassesProb @9 :Float32;
    occludedProb @10 :Float32;
    readyProb @11 :List(Float32);
    notReadyProb @12 :List(Float32);
  }
}

struct DriverStateDEPRECATED @0xb83c6cc593ed0a00 {
  frameId @0 :UInt32;
  modelExecutionTime @14 :Float32;
  dspExecutionTime @16 :Float32;
  rawPredictions @15 :Data;

  faceOrientation @3 :List(Float32);
  facePosition @4 :List(Float32);
  faceProb @5 :Float32;
  leftEyeProb @6 :Float32;
  rightEyeProb @7 :Float32;
  leftBlinkProb @8 :Float32;
  rightBlinkProb @9 :Float32;
  faceOrientationStd @11 :List(Float32);
  facePositionStd @12 :List(Float32);
  sunglassesProb @13 :Float32;
  poorVision @17 :Float32;
  partialFace @18 :Float32;
  distractedPose @19 :Float32;
  distractedEyes @20 :Float32;
  eyesOnRoad @21 :Float32;
  phoneUse @22 :Float32;
  occludedProb @23 :Float32;

  readyProb @24 :List(Float32);
  notReadyProb @25 :List(Float32);

  irPwrDEPRECATED @10 :Float32;
  descriptorDEPRECATED @1 :List(Float32);
  stdDEPRECATED @2 :Float32;
}

struct DriverMonitoringState @0xb83cda094a1da284 {
  events @18 :List(OnroadEvent);
  faceDetected @1 :Bool;
  isDistracted @2 :Bool;
  distractedType @17 :UInt32;
  awarenessStatus @3 :Float32;
  posePitchOffset @6 :Float32;
  posePitchValidCount @7 :UInt32;
  poseYawOffset @8 :Float32;
  poseYawValidCount @9 :UInt32;
  stepChange @10 :Float32;
  awarenessActive @11 :Float32;
  awarenessPassive @12 :Float32;
  isLowStd @13 :Bool;
  hiStdCount @14 :UInt32;
  isActiveMode @16 :Bool;
  isRHD @4 :Bool;

  isPreviewDEPRECATED @15 :Bool;
  rhdCheckedDEPRECATED @5 :Bool;
  eventsDEPRECATED @0 :List(Car.OnroadEventDEPRECATED);
}

struct Boot {
  wallTimeNanos @0 :UInt64;
  pstore @4 :Map(Text, Data);
  commands @5 :Map(Text, Data);
  launchLog @3 :Text;

  lastKmsgDEPRECATED @1 :Data;
  lastPmsgDEPRECATED @2 :Data;
}

struct LiveParametersData {
  valid @0 :Bool;
  gyroBias @1 :Float32;
  angleOffsetDeg @2 :Float32;
  angleOffsetAverageDeg @3 :Float32;
  stiffnessFactor @4 :Float32;
  steerRatio @5 :Float32;
  sensorValid @6 :Bool;
  posenetSpeed @8 :Float32;
  posenetValid @9 :Bool;
  angleOffsetFastStd @10 :Float32;
  angleOffsetAverageStd @11 :Float32;
  stiffnessFactorStd @12 :Float32;
  steerRatioStd @13 :Float32;
  roll @14 :Float32;
  debugFilterState @16 :FilterState;

  angleOffsetValid @17 :Bool = true;
  angleOffsetAverageValid @18 :Bool = true;
  steerRatioValid @19 :Bool = true;
  stiffnessFactorValid @20 :Bool = true;

  yawRateDEPRECATED @7 :Float32;
  filterStateDEPRECATED @15 :LiveLocationKalman.Measurement;

  struct FilterState {
    value @0 : List(Float64);
    std @1 : List(Float64);
  }
}

struct LiveTorqueParametersData {
  liveValid @0 :Bool;
  latAccelFactorRaw @1 :Float32;
  latAccelOffsetRaw @2 :Float32;
  frictionCoefficientRaw @3 :Float32;
  latAccelFactorFiltered @4 :Float32;
  latAccelOffsetFiltered @5 :Float32;
  frictionCoefficientFiltered @6 :Float32;
  totalBucketPoints @7 :Float32;
  decay @8 :Float32;
  maxResets @9 :Float32;
  points @10 :List(List(Float32));
  version @11 :Int32;
  useParams @12 :Bool;
  calPerc @13 :Int8;
}

struct LiveDelayData {
  lateralDelay @0 :Float32;
  validBlocks @1 :Int32;
  status @2 :Status;

  lateralDelayEstimate @3 :Float32;
  lateralDelayEstimateStd @5 :Float32;
  points @4 :List(Float32);
  calPerc @6 :Int8;

  enum Status {
    unestimated @0;
    estimated @1;
    invalid @2;
  }
}

struct LiveMapDataDEPRECATED {
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
  wideFromDeviceEuler @6 :List(Float32);
  wideFromDeviceEulerStd @7 :List(Float32);
  roadTransformTrans @8 :List(Float32);
  roadTransformTransStd @9 :List(Float32);
}

struct Sentinel {
  enum SentinelType {
    endOfSegment @0;
    endOfRoute @1;
    startOfSegment @2;
    startOfRoute @3;
  }
  type @0 :SentinelType;
  signal @1 :Int32;
}

struct UIDebug {
  drawTimeMillis @0 :Float32;
}

struct ManagerState {
  processes @0 :List(ProcessState);

  struct ProcessState {
    name @0 :Text;
    pid @1 :Int32;
    running @2 :Bool;
    shouldBeRunning @4 :Bool;
    exitCode @3 :Int32;
  }
}

struct UploaderState {
  immediateQueueSize @0 :UInt32;
  immediateQueueCount @1 :UInt32;
  rawQueueSize @2 :UInt32;
  rawQueueCount @3 :UInt32;

  # stats for last successfully uploaded file
  lastTime @4 :Float32;  # s
  lastSpeed @5 :Float32; # MB/s
  lastFilename @6 :Text;
}

struct NavInstruction {
  maneuverPrimaryText @0 :Text;
  maneuverSecondaryText @1 :Text;
  maneuverDistance @2 :Float32;  # m
  maneuverType @3 :Text; # TODO: Make Enum
  maneuverModifier @4 :Text; # TODO: Make Enum

  distanceRemaining @5 :Float32; # m
  timeRemaining @6 :Float32; # s
  timeRemainingTypical @7 :Float32; # s

  lanes @8 :List(Lane);
  showFull @9 :Bool;

  speedLimit @10 :Float32; # m/s
  speedLimitSign @11 :SpeedLimitSign;

  allManeuvers @12 :List(Maneuver);

  struct Lane {
    directions @0 :List(Direction);
    active @1 :Bool;
    activeDirection @2 :Direction;
  }

  enum Direction {
    none @0;
    left @1;
    right @2;
    straight @3;
    slightLeft @4;
    slightRight @5;
  }

  enum SpeedLimitSign {
    mutcd @0; # US Style
    vienna @1; # EU Style
  }

  struct Maneuver {
    distance @0 :Float32;
    type @1 :Text;
    modifier @2 :Text;
  }
}

struct NavRoute {
  coordinates @0 :List(Coordinate);

  struct Coordinate {
    latitude @0 :Float32;
    longitude @1 :Float32;
  }
}

struct MapRenderState {
  locationMonoTime @0 :UInt64;
  renderTime @1 :Float32;
  frameId @2: UInt32;
}

struct NavModelData {
  frameId @0 :UInt32;
  locationMonoTime @6 :UInt64;
  modelExecutionTime @1 :Float32;
  dspExecutionTime @2 :Float32;
  features @3 :List(Float32);
  # predicted future position
  position @4 :XYData;
  desirePrediction @5 :List(Float32);

  # All SI units and in device frame
  struct XYData {
    x @0 :List(Float32);
    y @1 :List(Float32);
    xStd @2 :List(Float32);
    yStd @3 :List(Float32);
  }
}

struct EncodeData {
  idx @0 :EncodeIndex;
  data @1 :Data;
  header @2 :Data;
  unixTimestampNanos @3 :UInt64;
  width @4 :UInt32;
  height @5 :UInt32;
}

struct DebugAlert {
  alertText1 @0 :Text;
  alertText2 @1 :Text;
}

struct UserBookmark @0xfe346a9de48d9b50 {
}

struct SoundPressure @0xdc24138990726023 {
  soundPressure @0 :Float32;

  # uncalibrated, A-weighted
  soundPressureWeighted @3 :Float32;
  soundPressureWeightedDb @1 :Float32;

  filteredSoundPressureWeightedDbDEPRECATED @2 :Float32;
}

struct AudioData {
  data @0 :Data;
  sampleRate @1 :UInt32;
}

struct AudioFeedback {
  audio @0 :AudioData;
  blockNum @1 :UInt16;
}

struct Touch {
  sec @0 :Int64;
  usec @1 :Int64;
  type @2 :UInt8;
  code @3 :Int32;
  value @4 :Int32;
}

struct Event {
  logMonoTime @0 :UInt64;  # nanoseconds
  valid @67 :Bool = true;

  union {
    # *********** log metadata ***********
    initData @1 :InitData;
    sentinel @73 :Sentinel;

    # *********** bootlog ***********
    boot @60 :Boot;

    # ********** openpilot daemon msgs **********
    gpsNMEA @3 :GPSNMEAData;
    can @5 :List(CanData);
    controlsState @7 :ControlsState;
    selfdriveState @130 :SelfdriveState;
    gyroscope @99 :SensorEventData;
    gyroscope2 @100 :SensorEventData;
    accelerometer @98 :SensorEventData;
    accelerometer2 @101 :SensorEventData;
    magnetometer @95 :SensorEventData;
    lightSensor @96 :SensorEventData;
    temperatureSensor @97 :SensorEventData;
    temperatureSensor2 @123 :SensorEventData;
    pandaStates @81 :List(PandaState);
    peripheralState @80 :PeripheralState;
    radarState @13 :RadarState;
    liveTracks @131 :Car.RadarData;
    sendcan @17 :List(CanData);
    liveCalibration @19 :LiveCalibrationData;
    carState @22 :Car.CarState;
    carControl @23 :Car.CarControl;
    carOutput @127 :Car.CarOutput;
    longitudinalPlan @24 :LongitudinalPlan;
    driverAssistance @132 :DriverAssistance;
    ubloxGnss @34 :UbloxGnss;
    ubloxRaw @39 :Data;
    qcomGnss @31 :QcomGnss;
    gpsLocationExternal @48 :GpsLocationData;
    gpsLocation @21 :GpsLocationData;
    gnssMeasurements @91 :GnssMeasurements;
    liveParameters @61 :LiveParametersData;
    liveTorqueParameters @94 :LiveTorqueParametersData;
    liveDelay @146 : LiveDelayData;
    cameraOdometry @63 :CameraOdometry;
    thumbnail @66: Thumbnail;
    onroadEvents @134: List(OnroadEvent);
    carParams @69: Car.CarParams;
    driverMonitoringState @71: DriverMonitoringState;
    livePose @129 :LivePose;
    modelV2 @75 :ModelDataV2;
    drivingModelData @128 :DrivingModelData;
    driverStateV2 @92 :DriverStateV2;

    # camera stuff, each camera state has a matching encode idx
    roadCameraState @2 :FrameData;
    driverCameraState @70: FrameData;
    wideRoadCameraState @74: FrameData;
    roadEncodeIdx @15 :EncodeIndex;
    driverEncodeIdx @76 :EncodeIndex;
    wideRoadEncodeIdx @77 :EncodeIndex;
    qRoadEncodeIdx @90 :EncodeIndex;

    livestreamRoadEncodeIdx @117 :EncodeIndex;
    livestreamWideRoadEncodeIdx @118 :EncodeIndex;
    livestreamDriverEncodeIdx @119 :EncodeIndex;

    # microphone data
    soundPressure @103 :SoundPressure;
    rawAudioData @147 :AudioData;

    # systems stuff
    androidLog @20 :AndroidLogEntry;
    managerState @78 :ManagerState;
    uploaderState @79 :UploaderState;
    procLog @33 :ProcLog;
    clocks @35 :Clocks;
    deviceState @6 :DeviceState;
    logMessage @18 :Text;
    errorLogMessage @85 :Text;

    # touch frame
    touch @135 :List(Touch);

    # navigation
    navInstruction @82 :NavInstruction;
    navRoute @83 :NavRoute;
    navThumbnail @84: Thumbnail;
    mapRenderState @105: MapRenderState;

    # UI services
    uiDebug @102 :UIDebug;

    # driving feedback
    userBookmark @93 :UserBookmark;
    bookmarkButton @148 :UserBookmark;
    audioFeedback @149 :AudioFeedback;

    # *********** debug ***********
    testJoystick @52 :Joystick;
    roadEncodeData @86 :EncodeData;
    driverEncodeData @87 :EncodeData;
    wideRoadEncodeData @88 :EncodeData;
    qRoadEncodeData @89 :EncodeData;
    alertDebug @133 :DebugAlert;

    livestreamRoadEncodeData @120 :EncodeData;
    livestreamWideRoadEncodeData @121 :EncodeData;
    livestreamDriverEncodeData @122 :EncodeData;

    # *********** Custom: reserved for forks ***********

    # DO change the name of the field
    # DON'T change anything after the "@"
    customReservedRawData0 @124 :Data;
    customReservedRawData1 @125 :Data;
    customReservedRawData2 @126 :Data;

    # DO change the name of the field and struct
    # DON'T change the ID (e.g. @107)
    # DON'T change which struct it points to
    customReserved0 @107 :Custom.CustomReserved0;
    customReserved1 @108 :Custom.CustomReserved1;
    customReserved2 @109 :Custom.CustomReserved2;
    customReserved3 @110 :Custom.CustomReserved3;
    customReserved4 @111 :Custom.CustomReserved4;
    customReserved5 @112 :Custom.CustomReserved5;
    customReserved6 @113 :Custom.CustomReserved6;
    customReserved7 @114 :Custom.CustomReserved7;
    customReserved8 @115 :Custom.CustomReserved8;
    customReserved9 @116 :Custom.CustomReserved9;
    customReserved10 @136 :Custom.CustomReserved10;
    customReserved11 @137 :Custom.CustomReserved11;
    customReserved12 @138 :Custom.CustomReserved12;
    customReserved13 @139 :Custom.CustomReserved13;
    customReserved14 @140 :Custom.CustomReserved14;
    customReserved15 @141 :Custom.CustomReserved15;
    customReserved16 @142 :Custom.CustomReserved16;
    customReserved17 @143 :Custom.CustomReserved17;
    customReserved18 @144 :Custom.CustomReserved18;
    customReserved19 @145 :Custom.CustomReserved19;

    # *********** legacy + deprecated ***********
    model @9 :Legacy.ModelData; # TODO: rename modelV2 and mark this as deprecated
    liveMpcDEPRECATED @36 :LiveMpcData;
    liveLongitudinalMpcDEPRECATED @37 :LiveLongitudinalMpcData;
    liveLocationKalmanLegacyDEPRECATED @51 :Legacy.LiveLocationData;
    orbslamCorrectionDEPRECATED @45 :Legacy.OrbslamCorrection;
    liveUIDEPRECATED @14 :Legacy.LiveUI;
    sensorEventDEPRECATED @4 :SensorEventData;
    liveEventDEPRECATED @8 :List(Legacy.LiveEventData);
    liveLocationDEPRECATED @25 :Legacy.LiveLocationData;
    ethernetDataDEPRECATED @26 :List(Legacy.EthernetPacket);
    cellInfoDEPRECATED @28 :List(Legacy.CellInfo);
    wifiScanDEPRECATED @29 :List(Legacy.WifiScan);
    uiNavigationEventDEPRECATED @50 :Legacy.UiNavigationEvent;
    liveMapDataDEPRECATED @62 :LiveMapDataDEPRECATED;
    gpsPlannerPointsDEPRECATED @40 :Legacy.GPSPlannerPoints;
    gpsPlannerPlanDEPRECATED @41 :Legacy.GPSPlannerPlan;
    applanixRawDEPRECATED @42 :Data;
    androidGnssDEPRECATED @30 :Legacy.AndroidGnss;
    lidarPtsDEPRECATED @32 :Legacy.LidarPts;
    navStatusDEPRECATED @38 :Legacy.NavStatus;
    trafficEventsDEPRECATED @43 :List(Legacy.TrafficEvent);
    liveLocationTimingDEPRECATED @44 :Legacy.LiveLocationData;
    liveLocationCorrectedDEPRECATED @46 :Legacy.LiveLocationData;
    navUpdateDEPRECATED @27 :Legacy.NavUpdate;
    orbObservationDEPRECATED @47 :List(Legacy.OrbObservation);
    locationDEPRECATED @49 :Legacy.LiveLocationData;
    orbOdometryDEPRECATED @53 :Legacy.OrbOdometry;
    orbFeaturesDEPRECATED @54 :Legacy.OrbFeatures;
    applanixLocationDEPRECATED @55 :Legacy.LiveLocationData;
    orbKeyFrameDEPRECATED @56 :Legacy.OrbKeyFrame;
    orbFeaturesSummaryDEPRECATED @58 :Legacy.OrbFeaturesSummary;
    featuresDEPRECATED @10 :Legacy.CalibrationFeatures;
    kalmanOdometryDEPRECATED @65 :Legacy.KalmanOdometry;
    uiLayoutStateDEPRECATED @57 :Legacy.UiLayoutState;
    pandaStateDEPRECATED @12 :PandaState;
    driverStateDEPRECATED @59 :DriverStateDEPRECATED;
    sensorEventsDEPRECATED @11 :List(SensorEventData);
    lateralPlanDEPRECATED @64 :LateralPlan;
    navModelDEPRECATED @104 :NavModelData;
    uiPlanDEPRECATED @106 :UiPlan;
    liveLocationKalmanDEPRECATED @72 :LiveLocationKalman;
    liveTracksDEPRECATED @16 :List(LiveTracksDEPRECATED);
    onroadEventsDEPRECATED @68: List(Car.OnroadEventDEPRECATED);
  }
}
