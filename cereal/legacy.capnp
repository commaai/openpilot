using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0x80ef1ec4889c2a63;

# legacy.capnp: a home for deprecated structs

struct LogRotate @0x9811e1f38f62f2d1 {
  segmentNum @0 :Int32;
  path @1 :Text;
}

struct LiveUI @0xc08240f996aefced {
  rearViewCam @0 :Bool;
  alertText1 @1 :Text;
  alertText2 @2 :Text;
  awarenessStatus @3 :Float32;
}

struct UiLayoutState @0x88dcce08ad29dda0 {
  activeApp @0 :App;
  sidebarCollapsed @1 :Bool;
  mapEnabled @2 :Bool;
  mockEngaged @3 :Bool;

  enum App @0x9917470acf94d285 {
    home @0;
    music @1;
    nav @2;
    settings @3;
    none @4;
  }
}

struct OrbslamCorrection @0x8afd33dc9b35e1aa {
  correctionMonoTime @0 :UInt64;
  prePositionECEF @1 :List(Float64);
  postPositionECEF @2 :List(Float64);
  prePoseQuatECEF @3 :List(Float32);
  postPoseQuatECEF @4 :List(Float32);
  numInliers @5 :UInt32;
}

struct EthernetPacket @0xa99a9d5b33cf5859 {
  pkt @0 :Data;
  ts @1 :Float32;
}

struct CellInfo @0xcff7566681c277ce {
  timestamp @0 :UInt64;
  repr @1 :Text; # android toString() for now
}

struct WifiScan @0xd4df5a192382ba0b {
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

  enum ChannelWidth @0xcb6a279f015f6b51 {
    w20Mhz @0;
    w40Mhz @1;
    w80Mhz @2;
    w160Mhz @3;
    w80Plus80Mhz @4;
  }
}

struct LiveEventData @0x94b7baa90c5c321e {
  name @0 :Text;
  value @1 :Int32;
}

struct ModelData @0xb8aad62cffef28a9 {
  frameId @0 :UInt32;
  frameAge @12 :UInt32;
  frameDropPerc @13 :Float32;
  timestampEof @9 :UInt64;
  modelExecutionTime @14 :Float32;
  gpuExecutionTime @16 :Float32;
  rawPred @15 :Data;

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

  struct PathData @0x8817eeea389e9f08 {
    points @0 :List(Float32);
    prob @1 :Float32;
    std @2 :Float32;
    stds @3 :List(Float32);
    poly @4 :List(Float32);
    validLen @5 :Float32;
  }

  struct LeadData @0xd1c9bef96d26fa91 {
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

  struct ModelSettings @0xa26e3710efd3e914 {
    bigBoxX @0 :UInt16;
    bigBoxY @1 :UInt16;
    bigBoxWidth @2 :UInt16;
    bigBoxHeight @3 :UInt16;
    boxProjection @4 :List(Float32);
    yuvCorrection @5 :List(Float32);
    inputTransform @6 :List(Float32);
  }

  struct MetaData @0x9744f25fb60f2bf8 {
    engagedProb @0 :Float32;
    desirePrediction @1 :List(Float32);
    brakeDisengageProb @2 :Float32;
    gasDisengageProb @3 :Float32;
    steerOverrideProb @4 :Float32;
    desireState @5 :List(Float32);
  }

  struct LongitudinalData @0xf98f999c6a071122 {
    distances @2 :List(Float32);
    speeds @0 :List(Float32);
    accelerations @1 :List(Float32);
  }
}

struct ECEFPoint @0xc25bbbd524983447 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

struct ECEFPointDEPRECATED @0xe10e21168db0c7f7 {
  x @0 :Float32;
  y @1 :Float32;
  z @2 :Float32;
}

struct GPSPlannerPoints @0xab54c59699f8f9f3 {
  curPosDEPRECATED @0 :ECEFPointDEPRECATED;
  pointsDEPRECATED @1 :List(ECEFPointDEPRECATED);
  curPos @6 :ECEFPoint;
  points @7 :List(ECEFPoint);
  valid @2 :Bool;
  trackName @3 :Text;
  speedLimit @4 :Float32;
  accelTarget @5 :Float32;
}

struct GPSPlannerPlan @0xf5ad1d90cdc1dd6b {
  valid @0 :Bool;
  poly @1 :List(Float32);
  trackName @2 :Text;
  speed @3 :Float32;
  acceleration @4 :Float32;
  pointsDEPRECATED @5 :List(ECEFPointDEPRECATED);
  points @6 :List(ECEFPoint);
  xLookahead @7 :Float32;
}

struct UiNavigationEvent @0x90c8426c3eaddd3b {
  type @0: Type;
  status @1: Status;
  distanceTo @2: Float32;
  endRoadPointDEPRECATED @3: ECEFPointDEPRECATED;
  endRoadPoint @4: ECEFPoint;

  enum Type @0xe8db07dcf8fcea05 {
    none @0;
    laneChangeLeft @1;
    laneChangeRight @2;
    mergeLeft @3;
    mergeRight @4;
    turnLeft @5;
    turnRight @6;
  }

  enum Status @0xb9aa88c75ef99a1f {
    none @0;
    passive @1;
    approaching @2;
    active @3;
  }
}

struct LiveLocationData @0xb99b2bc7a57e8128 {
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

  struct Accuracy @0x943dc4625473b03f {
    pNEDError @0 :List(Float32);
    vNEDError @1 :List(Float32);
    rollError @2 :Float32;
    pitchError @3 :Float32;
    headingError @4 :Float32;
    ellipsoidSemiMajorError @5 :Float32;
    ellipsoidSemiMinorError @6 :Float32;
    ellipsoidOrientationError @7 :Float32;
  }

  enum SensorSource @0xc871d3cc252af657 {
    applanix @0;
    kalman @1;
    orbslam @2;
    timing @3;
    dummy @4;
  }
}

struct OrbOdometry @0xd7700859ed1f5b76 {
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

struct OrbFeatures @0xcd60164a8a0159ef {
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

struct OrbFeaturesSummary @0xd500d30c5803fa4f {
  timestampEof @0 :UInt64;
  timestampLastEof @1 :UInt64;

  featureCount @2 :UInt16;
  matchCount @3 :UInt16;
  computeNs @4 :UInt64;
}

struct OrbKeyFrame @0xc8233c0345e27e24 {
  # this is a globally unique id for the KeyFrame
  id @0: UInt64;

  # this is the location of the KeyFrame
  pos @1: ECEFPoint;

  # these are the features in the world
  # len(dpos) == len(descriptors) * 32
  dpos @2 :List(ECEFPoint);
  descriptors @3 :Data;
}

struct KalmanOdometry @0x92e21bb7ea38793a {
  trans @0 :List(Float32); # m/s in device frame
  rot @1 :List(Float32); # rad/s in device frame
  transStd @2 :List(Float32); # std m/s in device frame
  rotStd @3 :List(Float32); # std rad/s in device frame
}

struct OrbObservation @0x9b326d4e436afec7 {
  observationMonoTime @0 :UInt64;
  normalizedCoordinates @1 :List(Float32);
  locationECEF @2 :List(Float64);
  matchDistance @3: UInt32;
}

struct CalibrationFeatures @0x8fdfadb254ea867a {
  frameId @0 :UInt32;

  p0 @1 :List(Float32);
  p1 @2 :List(Float32);
  status @3 :List(Int8);
}

struct NavStatus @0xbd8822120928120c {
  isNavigating @0 :Bool;
  currentAddress @1 :Address;

  struct Address @0xce7cd672cacc7814 {
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

struct NavUpdate @0xdb98be6565516acb {
  isNavigating @0 :Bool;
  curSegment @1 :Int32;
  segments @2 :List(Segment);

  struct LatLng @0x9eaef9187cadbb9b {
    lat @0 :Float64;
    lng @1 :Float64;
  }

  struct Segment @0xa5b39b4fc4d7da3f {
    from @0 :LatLng;
    to @1 :LatLng;
    updateTime @2 :Int32;
    distance @3 :Int32;
    crossTime @4 :Int32;
    exitNo @5 :Int32;
    instruction @6 :Instruction;

    parts @7 :List(LatLng);

    enum Instruction @0xc5417a637451246f {
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

struct TrafficEvent @0xacfa74a094e62626 {
  type @0 :Type;
  distance @1 :Float32;
  action @2 :Action;
  resuming @3 :Bool;

  enum Type @0xd85d75253435bf4b {
    stopSign @0;
    lightRed @1;
    lightYellow @2;
    lightGreen @3;
    stopLight @4;
  }

  enum Action @0xa6f6ce72165ccb49 {
    none @0;
    yield @1;
    stop @2;
    resumeReady @3;
  }

}


struct AndroidGnss @0xdfdf30d03fc485bd {
  union {
    measurements @0 :Measurements;
    navigationMessage @1 :NavigationMessage;
  }

  struct Measurements @0xa20710d4f428d6cd {
    clock @0 :Clock;
    measurements @1 :List(Measurement);

    struct Clock @0xa0e27b453a38f450 {
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

    struct Measurement @0xd949bf717d77614d {
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

      enum Constellation @0x9ef1f3ff0deb5ffb {
        unknown @0;
        gps @1;
        sbas @2;
        glonass @3;
        qzss @4;
        beidou @5;
        galileo @6;
      }

      enum State @0xcbb9490adce12d72 {
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

      enum MultipathIndicator @0xc04e7b6231d4caa8 {
        unknown @0;
        detected @1;
        notDetected @2;
      }
    }
  }

  struct NavigationMessage @0xe2517b083095fd4e {
    type @0 :Int32;
    svId @1 :Int32;
    messageId @2 :Int32;
    submessageId @3 :Int32;
    data @4 :Data;
    status @5 :Status;

    enum Status @0xec1ff7996b35366f {
      unknown @0;
      parityPassed @1;
      parityRebuilt @2;
    }
  }
}

struct LidarPts @0xe3d6685d4e9d8f7a {
  r @0 :List(UInt16);        # uint16   m*500.0
  theta @1 :List(UInt16);    # uint16 deg*100.0
  reflect @2 :List(UInt8);   # uint8      0-255

  # For storing out of file.
  idx @3 :UInt64;

  # For storing in file
  pkt @4 :Data;
}


