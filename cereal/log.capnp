using Cxx = import "c++.capnp";
$Cxx.namespace("cereal");

using Java = import "java.capnp";
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
  gctx @1 :Text;
  dongleId @2 :Text;

  deviceType @3 :DeviceType;
  version @4 :Text;

  androidBuildInfo @5 :AndroidBuildInfo;
  androidSensors @6 :List(AndroidSensor);
  chffrAndroidExtra @7 :ChffrAndroidExtra;

  pandaInfo @8 :PandaInfo;

  dirty @9 :Bool;

  enum DeviceType {
    unknown @0;
    neo @1;
    chffrAndroid @2;
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

  struct PandaInfo {
    hasPanda @0: Bool;
    dongleId @1: Text;
    stVersion @2: Text;
    espVersion @3: Text;
  }
}

struct FrameData {
  frameId @0 :UInt32;
  encodeId @1 :UInt32; # DEPRECATED
  timestampEof @2 :UInt64;
  frameLength @3 :Int32;
  integLines @4 :Int32;
  globalGain @5 :Int32;
  image @6 :Data;

  frameType @7 :FrameType;
  timestampSof @8 :UInt64;

  androidCaptureResult @9 :AndroidCaptureResult;
  
  enum FrameType {
    unknown @0;
    neo @1;
    chffrAndroid @2;
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

  # not thermal
  freeSpace @7 :Float32;
  batteryPercent @8 :Int16;
  batteryStatus @9: Text;
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
  ftMonoTimeDEPRECATED @7 :UInt64;
  l100MonoTime @11 :UInt64;

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

  # Maps car space to normalized image space.
  extrinsicMatrix @4 :List(Float32);
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
  canMonoTimeDEPRECATED @16 :UInt64;
  canMonoTimes @21 :List(UInt64);
  l20MonoTimeDEPRECATED @17 :UInt64;
  mdMonoTimeDEPRECATED @18 :UInt64;
  planMonoTime @28 :UInt64;

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
  angleSteers @13 :Float32;  # Steering angle in degrees.
  hudLeadDEPRECATED @14 :Int32;
  cumLagMs @15 :Float32;

  enabled @19: Bool;
  steerOverride @20: Bool;

  vCruise @22: Float32;

  rearViewCam @23 :Bool;
  alertText1 @24 :Text; 
  alertText2 @25 :Text; 
  awarenessStatus @26 :Float32;

  angleOffset @27 :Float32;
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
  l20MonoTime @10 :UInt64;

  # lateral, 3rd order polynomial
  lateralValid @0: Bool;
  dPoly @1 :List(Float32);

  # longitudinal
  longitudinalValid @2: Bool;
  vTarget @3 :Float32;
  aTargetMin @4 :Float32;
  aTargetMax @5 :Float32;
  jerkFactor @6 :Float32;
  hasLead @7 :Bool;
  fcw @8 :Bool;
}

struct LiveLocationData {
  status @0: UInt8;

  # 3D fix 
  lat @1: Float64;
  lon @2: Float64;
  alt @3: Float32;     # m

  # speed
  speed @4: Float32;   # m/s

  # NED velocity components
  vNED @5: List(Float32);

  # roll, pitch, heading (x,y,z)
  roll @6: Float32;     # WRT to center of earth?
  pitch @7: Float32;    # WRT to center of earth?
  heading @8: Float32;  # WRT to north?

  # what are these?
  wanderAngle @9: Float32;
  trackAngle @10: Float32;

  # car frame -- https://upload.wikimedia.org/wikipedia/commons/f/f5/RPY_angles_of_cars.png

  # gyro, in car frame, deg/s
  gyro @11: List(Float32);

  # accel, in car frame, m/s^2
  accel @12: List(Float32);

  accuracy @13: Accuracy;

  struct Accuracy {
    pNEDError @0: List(Float32);
    vNEDError @1: List(Float32);
    rollError @2: Float32;
    pitchError @3: Float32;
    headingError @4: Float32;
    ellipsoidSemiMajorError @5: Float32;
    ellipsoidSemiMinorError @6: Float32;
    ellipsoidOrientationError @7: Float32;
  }
}

struct EthernetPacket {
  pkt @0 :Data;
  ts @1: Float32;
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
  }

  struct MeasurementReport {
    source @0 :Source;

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

    enum Source {
      gps @0;
      glonass @1;
    }

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
      predetectIntegration @10 :UInt8;
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

      struct MeasurementStatus {
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
        gpsMultipathIndicator @23 :Bool;
        
        imdJammingIndicator @24 :Bool;
        lteB13TxJammingIndicator @25 :Bool;
        freshMeasurementIndicator @26 :Bool;

        multipathEstimateIsValid @27 :Bool;
        directionIsValid @28 :Bool;
      }

      enum SVObservationState {
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
    }

  }

  struct ClockReport {
    hasFCount @0 :Bool;
    fCount @1 :UInt32;

    hasGpsWeekNumber @2 :Bool;
    gpsWeekNumber @3 :UInt16;
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
      clkReset @1 : Bool;
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
      cno @8 : UInt8;
      # pseudorange standard deviation in meters
      pseudorangeStdev @9 :Float32;
      # carrier phase standard deviation in cycles
      carrierPhaseStdev @10 :Float32;
      # doppler standard deviation in Hz
      dopplerStdev @11 :Float32;

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
  }
}
struct Event {
  # in nanoseconds?
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
  }
}
