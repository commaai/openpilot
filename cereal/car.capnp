using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0x8e2af1e708af8b8d;

# ******* events causing controls state machine transition *******

struct CarEvent @0x9b1657f34caf3ad3 {
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

  enum EventName @0xbaa8c5d505f727de {
    canError @0;
    steerUnavailable @1;
    wrongGear @4;
    doorOpen @5;
    seatbeltNotLatched @6;
    espDisabled @7;
    wrongCarMode @8;
    steerTempUnavailable @9;
    reverseGear @10;
    buttonCancel @11;
    buttonEnable @12;
    pedalPressed @13;  # exits active state
    preEnableStandstill @73;  # added during pre-enable state with brake
    gasPressedOverride @108;  # added when user is pressing gas with no disengage on gas
    steerOverride @114;
    cruiseDisabled @14;
    speedTooLow @17;
    outOfSpace @18;
    overheat @19;
    calibrationIncomplete @20;
    calibrationInvalid @21;
    calibrationRecalibrating @117;
    controlsMismatch @22;
    pcmEnable @23;
    pcmDisable @24;
    radarFault @26;
    brakeHold @28;
    parkBrake @29;
    manualRestart @30;
    lowSpeedLockout @31;
    joystickDebug @34;
    steerTempUnavailableSilent @35;
    resumeRequired @36;
    preDriverDistracted @37;
    promptDriverDistracted @38;
    driverDistracted @39;
    preDriverUnresponsive @43;
    promptDriverUnresponsive @44;
    driverUnresponsive @45;
    belowSteerSpeed @46;
    lowBattery @48;
    accFaulted @51;
    sensorDataInvalid @52;
    commIssue @53;
    commIssueAvgFreq @109;
    tooDistracted @54;
    posenetInvalid @55;
    soundsUnavailable @56;
    preLaneChangeLeft @57;
    preLaneChangeRight @58;
    laneChange @59;
    lowMemory @63;
    stockAeb @64;
    ldw @65;
    carUnrecognized @66;
    invalidLkasSetting @69;
    speedTooHigh @70;
    laneChangeBlocked @71;
    relayMalfunction @72;
    stockFcw @74;
    startup @75;
    startupNoCar @76;
    startupNoControl @77;
    startupMaster @78;
    startupNoFw @104;
    fcw @79;
    steerSaturated @80;
    belowEngageSpeed @84;
    noGps @85;
    wrongCruiseMode @87;
    modeldLagging @89;
    deviceFalling @90;
    fanMalfunction @91;
    cameraMalfunction @92;
    cameraFrameRate @110;
    processNotRunning @95;
    dashcamMode @96;
    controlsInitializing @98;
    usbError @99;
    roadCameraError @100;
    driverCameraError @101;
    wideRoadCameraError @102;
    highCpuUsage @105;
    cruiseMismatch @106;
    lkasDisabled @107;
    canBusMissing @111;
    controlsdLagging @112;
    resumeBlocked @113;
    steerTimeLimit @115;
    vehicleSensorsInvalid @116;
    locationdTemporaryError @103;
    locationdPermanentError @118;
    paramsdTemporaryError @50;
    paramsdPermanentError @119;
    actuatorsApiUnavailable @120;
    espActive @121;

    radarCanErrorDEPRECATED @15;
    communityFeatureDisallowedDEPRECATED @62;
    radarCommIssueDEPRECATED @67;
    driverMonitorLowAccDEPRECATED @68;
    gasUnavailableDEPRECATED @3;
    dataNeededDEPRECATED @16;
    modelCommIssueDEPRECATED @27;
    ipasOverrideDEPRECATED @33;
    geofenceDEPRECATED @40;
    driverMonitorOnDEPRECATED @41;
    driverMonitorOffDEPRECATED @42;
    calibrationProgressDEPRECATED @47;
    invalidGiraffeHondaDEPRECATED @49;
    invalidGiraffeToyotaDEPRECATED @60;
    internetConnectivityNeededDEPRECATED @61;
    whitePandaUnsupportedDEPRECATED @81;
    commIssueWarningDEPRECATED @83;
    focusRecoverActiveDEPRECATED @86;
    neosUpdateRequiredDEPRECATED @88;
    modelLagWarningDEPRECATED @93;
    startupOneplusDEPRECATED @82;
    startupFuzzyFingerprintDEPRECATED @97;
    noTargetDEPRECATED @25;
    brakeUnavailableDEPRECATED @2;
    plannerErrorDEPRECATED @32;
    gpsMalfunctionDEPRECATED @94;
  }
}

# ******* main car state @ 100hz *******
# all speeds in m/s

struct CarState {
  events @13 :List(CarEvent);

  # CAN health
  canValid @26 :Bool;       # invalid counter/checksums
  canTimeout @40 :Bool;     # CAN bus dropped out
  canErrorCounter @48 :UInt32;

  # car speed
  vEgo @1 :Float32;          # best estimate of speed
  aEgo @16 :Float32;         # best estimate of acceleration
  vEgoRaw @17 :Float32;      # unfiltered speed from CAN sensors
  vEgoCluster @44 :Float32;  # best estimate of speed shown on car's instrument cluster, used for UI

  yawRate @22 :Float32;     # best estimate of yaw rate
  standstill @18 :Bool;
  wheelSpeeds @2 :WheelSpeeds;

  # gas pedal, 0.0-1.0
  gas @3 :Float32;        # this is user pedal only
  gasPressed @4 :Bool;    # this is user pedal only

  engineRpm @46 :Float32;

  # brake pedal, 0.0-1.0
  brake @5 :Float32;      # this is user pedal only
  brakePressed @6 :Bool;  # this is user pedal only
  regenBraking @45 :Bool; # this is user pedal only
  parkingBrake @39 :Bool;
  brakeHoldActive @38 :Bool;

  # steering wheel
  steeringAngleDeg @7 :Float32;
  steeringAngleOffsetDeg @37 :Float32; # Offset betweens sensors in case there multiple
  steeringRateDeg @15 :Float32;
  steeringTorque @8 :Float32;      # TODO: standardize units
  steeringTorqueEps @27 :Float32;  # TODO: standardize units
  steeringPressed @9 :Bool;        # if the user is using the steering wheel
  steerFaultTemporary @35 :Bool;   # temporary EPS fault
  steerFaultPermanent @36 :Bool;   # permanent EPS fault
  stockAeb @30 :Bool;
  stockFcw @31 :Bool;
  espDisabled @32 :Bool;
  accFaulted @42 :Bool;
  carFaultedNonCritical @47 :Bool;  # some ECU is faulted, but car remains controllable
  espActive @51 :Bool;

  # cruise state
  cruiseState @10 :CruiseState;

  # gear
  gearShifter @14 :GearShifter;

  # button presses
  buttonEvents @11 :List(ButtonEvent);
  leftBlinker @20 :Bool;
  rightBlinker @21 :Bool;
  genericToggle @23 :Bool;

  # lock info
  doorOpen @24 :Bool;
  seatbeltUnlatched @25 :Bool;

  # clutch (manual transmission only)
  clutchPressed @28 :Bool;

  # blindspot sensors
  leftBlindspot @33 :Bool; # Is there something blocking the left lane change
  rightBlindspot @34 :Bool; # Is there something blocking the right lane change

  fuelGauge @41 :Float32; # battery or fuel tank level from 0.0 to 1.0
  charging @43 :Bool;

  # process meta
  cumLagMs @50 :Float32;

  struct WheelSpeeds {
    # optional wheel speeds
    fl @0 :Float32;
    fr @1 :Float32;
    rl @2 :Float32;
    rr @3 :Float32;
  }

  struct CruiseState {
    enabled @0 :Bool;
    speed @1 :Float32;
    speedCluster @6 :Float32;  # Set speed as shown on instrument cluster
    available @2 :Bool;
    speedOffset @3 :Float32;
    standstill @4 :Bool;
    nonAdaptive @5 :Bool;
  }

  enum GearShifter {
    unknown @0;
    park @1;
    drive @2;
    neutral @3;
    reverse @4;
    sport @5;
    low @6;
    brake @7;
    eco @8;
    manumatic @9;
  }

  # send on change
  struct ButtonEvent {
    pressed @0 :Bool;
    type @1 :Type;

    enum Type {
      unknown @0;
      leftBlinker @1;
      rightBlinker @2;
      accelCruise @3;
      decelCruise @4;
      cancel @5;
      altButton1 @6;
      altButton2 @7;
      altButton3 @8;
      setCruise @9;
      resumeCruise @10;
      gapAdjustCruise @11;
    }
  }

  # deprecated
  errorsDEPRECATED @0 :List(CarEvent.EventName);
  brakeLightsDEPRECATED @19 :Bool;
  steeringRateLimitedDEPRECATED @29 :Bool;
  canMonoTimesDEPRECATED @12: List(UInt64);
  canRcvTimeoutDEPRECATED @49 :Bool;
}

# ******* radar state @ 20hz *******

struct RadarData @0x888ad6581cf0aacb {
  errors @0 :List(Error);
  points @1 :List(RadarPoint);

  enum Error {
    canError @0;
    fault @1;
    wrongConfig @2;
  }

  # similar to LiveTracks
  # is one timestamp valid for all? I think so
  struct RadarPoint {
    trackId @0 :UInt64;  # no trackId reuse

    # these 3 are the minimum required
    dRel @1 :Float32; # m from the front bumper of the car
    yRel @2 :Float32; # m
    vRel @3 :Float32; # m/s

    # these are optional and valid if they are not NaN
    aRel @4 :Float32; # m/s^2
    yvRel @5 :Float32; # m/s

    # some radars flag measurements VS estimates
    measured @6 :Bool;
  }

  # deprecated
  canMonoTimesDEPRECATED @2 :List(UInt64);
}

# ******* car controls @ 100hz *******

struct CarControl {
  # must be true for any actuator commands to work
  enabled @0 :Bool;
  latActive @11: Bool;
  longActive @12: Bool;

  # Actuator commands as computed by controlsd
  actuators @6 :Actuators;

  # moved to CarOutput
  actuatorsOutputDEPRECATED @10 :Actuators;

  leftBlinker @15: Bool;
  rightBlinker @16: Bool;

  orientationNED @13 :List(Float32);
  angularVelocity @14 :List(Float32);

  cruiseControl @4 :CruiseControl;
  hudControl @5 :HUDControl;

  struct Actuators {
    # range from 0.0 - 1.0
    gas @0: Float32;
    brake @1: Float32;
    # range from -1.0 - 1.0
    steer @2: Float32;
    # value sent over can to the car
    steerOutputCan @8: Float32;
    steeringAngleDeg @3: Float32;

    curvature @7: Float32;

    speed @6: Float32; # m/s
    accel @4: Float32; # m/s^2
    longControlState @5: LongControlState;

    enum LongControlState @0xe40f3a917d908282{
      off @0;
      pid @1;
      stopping @2;
      starting @3;
    }
  }

  struct CruiseControl {
    cancel @0: Bool;
    resume @1: Bool;
    override @4: Bool;
    speedOverrideDEPRECATED @2: Float32;
    accelOverrideDEPRECATED @3: Float32;
  }

  struct HUDControl {
    speedVisible @0: Bool;
    setSpeed @1: Float32;
    lanesVisible @2: Bool;
    leadVisible @3: Bool;
    visualAlert @4: VisualAlert;
    audibleAlert @5: AudibleAlert;
    rightLaneVisible @6: Bool;
    leftLaneVisible @7: Bool;
    rightLaneDepart @8: Bool;
    leftLaneDepart @9: Bool;
    leadDistanceBars @10: Int8;  # 1-3: 1 is closest, 3 is farthest. some ports may utilize 2-4 bars instead

    enum VisualAlert {
      # these are the choices from the Honda
      # map as good as you can for your car
      none @0;
      fcw @1;
      steerRequired @2;
      brakePressed @3;
      wrongGear @4;
      seatbeltUnbuckled @5;
      speedTooHigh @6;
      ldw @7;
    }

    enum AudibleAlert {
      none @0;

      engage @1;
      disengage @2;
      refuse @3;

      warningSoft @4;
      warningImmediate @5;

      prompt @6;
      promptRepeat @7;
      promptDistracted @8;
    }
  }

  gasDEPRECATED @1 :Float32;
  brakeDEPRECATED @2 :Float32;
  steeringTorqueDEPRECATED @3 :Float32;
  activeDEPRECATED @7 :Bool;
  rollDEPRECATED @8 :Float32;
  pitchDEPRECATED @9 :Float32;
}

struct CarOutput {
  # Any car specific rate limits or quirks applied by
  # the CarController are reflected in actuatorsOutput
  # and matches what is sent to the car
  actuatorsOutput @0 :CarControl.Actuators;
}

# ****** car param ******

struct CarParams {
  carName @0 :Text;
  carFingerprint @1 :Text;
  fuzzyFingerprint @55 :Bool;

  notCar @66 :Bool;  # flag for non-car robotics platforms

  pcmCruise @3 :Bool;        # is openpilot's state tied to the PCM's cruise state?
  enableDsu @5 :Bool;        # driving support unit
  enableBsm @56 :Bool;       # blind spot monitoring
  flags @64 :UInt32;         # flags for car specific quirks
  experimentalLongitudinalAvailable @71 :Bool;

  minEnableSpeed @7 :Float32;
  minSteerSpeed @8 :Float32;
  safetyConfigs @62 :List(SafetyConfig);
  alternativeExperience @65 :Int16;      # panda flag for features like no disengage on gas

  # Car docs fields
  maxLateralAccel @68 :Float32;
  autoResumeSng @69 :Bool;               # describes whether car can resume from a stop automatically

  # things about the car in the manual
  mass @17 :Float32;            # [kg] curb weight: all fluids no cargo
  wheelbase @18 :Float32;       # [m] distance from rear axle to front axle
  centerToFront @19 :Float32;   # [m] distance from center of mass to front axle
  steerRatio @20 :Float32;      # [] ratio of steering wheel angle to front wheel angle
  steerRatioRear @21 :Float32;  # [] ratio of steering wheel angle to rear wheel angle (usually 0)

  # things we can derive
  rotationalInertia @22 :Float32;    # [kg*m2] body rotational inertia
  tireStiffnessFactor @72 :Float32;  # scaling factor used in calculating tireStiffness[Front,Rear]
  tireStiffnessFront @23 :Float32;   # [N/rad] front tire coeff of stiff
  tireStiffnessRear @24 :Float32;    # [N/rad] rear tire coeff of stiff

  longitudinalTuning @25 :LongitudinalPIDTuning;
  lateralParams @48 :LateralParams;
  lateralTuning :union {
    pid @26 :LateralPIDTuning;
    indiDEPRECATED @27 :LateralINDITuning;
    lqrDEPRECATED @40 :LateralLQRTuning;
    torque @67 :LateralTorqueTuning;
  }

  steerLimitAlert @28 :Bool;
  steerLimitTimer @47 :Float32;  # time before steerLimitAlert is issued

  vEgoStopping @29 :Float32; # Speed at which the car goes into stopping state
  vEgoStarting @59 :Float32; # Speed at which the car goes into starting state
  stoppingControl @31 :Bool; # Does the car allow full control even at lows speeds when stopping
  steerControlType @34 :SteerControlType;
  radarUnavailable @35 :Bool; # True when radar objects aren't visible on CAN or aren't parsed out
  stopAccel @60 :Float32; # Required acceleration to keep vehicle stationary
  stoppingDecelRate @52 :Float32; # m/s^2/s while trying to stop
  startAccel @32 :Float32; # Required acceleration to get car moving
  startingState @70 :Bool; # Does this car make use of special starting state

  steerActuatorDelay @36 :Float32; # Steering wheel actuator delay in seconds
  longitudinalActuatorDelay @58 :Float32; # Gas/Brake actuator delay in seconds
  openpilotLongitudinalControl @37 :Bool; # is openpilot doing the longitudinal control?
  carVin @38 :Text; # VIN number queried during fingerprinting
  dashcamOnly @41: Bool;
  passive @73: Bool;   # is openpilot in control?
  transmissionType @43 :TransmissionType;
  carFw @44 :List(CarFw);

  radarTimeStep @45: Float32 = 0.05;  # time delta between radar updates, 20Hz is very standard
  fingerprintSource @49: FingerprintSource;
  networkLocation @50 :NetworkLocation;  # Where Panda/C2 is integrated into the car's CAN network

  wheelSpeedFactor @63 :Float32; # Multiplier on wheels speeds to computer actual speeds

  struct SafetyConfig {
    safetyModel @0 :SafetyModel;
    safetyParam @3 :UInt16;
    safetyParamDEPRECATED @1 :Int16;
    safetyParam2DEPRECATED @2 :UInt32;
  }

  struct LateralParams {
    torqueBP @0 :List(Int32);
    torqueV @1 :List(Int32);
  }

  struct LateralPIDTuning {
    kpBP @0 :List(Float32);
    kpV @1 :List(Float32);
    kiBP @2 :List(Float32);
    kiV @3 :List(Float32);
    kf @4 :Float32;
  }

  struct LateralTorqueTuning {
    useSteeringAngle @0 :Bool;
    kp @1 :Float32;
    ki @2 :Float32;
    friction @3 :Float32;
    kf @4 :Float32;
    steeringAngleDeadzoneDeg @5 :Float32;
    latAccelFactor @6 :Float32;
    latAccelOffset @7 :Float32;
  }

  struct LongitudinalPIDTuning {
    kpBP @0 :List(Float32);
    kpV @1 :List(Float32);
    kiBP @2 :List(Float32);
    kiV @3 :List(Float32);
    kf @6 :Float32;
    deadzoneBPDEPRECATED @4 :List(Float32);
    deadzoneVDEPRECATED @5 :List(Float32);
  }

  struct LateralINDITuning {
    outerLoopGainBP @4 :List(Float32);
    outerLoopGainV @5 :List(Float32);
    innerLoopGainBP @6 :List(Float32);
    innerLoopGainV @7 :List(Float32);
    timeConstantBP @8 :List(Float32);
    timeConstantV @9 :List(Float32);
    actuatorEffectivenessBP @10 :List(Float32);
    actuatorEffectivenessV @11 :List(Float32);

    outerLoopGainDEPRECATED @0 :Float32;
    innerLoopGainDEPRECATED @1 :Float32;
    timeConstantDEPRECATED @2 :Float32;
    actuatorEffectivenessDEPRECATED @3 :Float32;
  }

  struct LateralLQRTuning {
    scale @0 :Float32;
    ki @1 :Float32;
    dcGain @2 :Float32;

    # State space system
    a @3 :List(Float32);
    b @4 :List(Float32);
    c @5 :List(Float32);

    k @6 :List(Float32);  # LQR gain
    l @7 :List(Float32);  # Kalman gain
  }

  enum SafetyModel {
    silent @0;
    hondaNidec @1;
    toyota @2;
    elm327 @3;
    gm @4;
    hondaBoschGiraffe @5;
    ford @6;
    cadillac @7;
    hyundai @8;
    chrysler @9;
    tesla @10;
    subaru @11;
    gmPassive @12;
    mazda @13;
    nissan @14;
    volkswagen @15;
    toyotaIpas @16;
    allOutput @17;
    gmAscm @18;
    noOutput @19;  # like silent but without silent CAN TXs
    hondaBosch @20;
    volkswagenPq @21;
    subaruPreglobal @22;  # pre-Global platform
    hyundaiLegacy @23;
    hyundaiCommunity @24;
    volkswagenMlb @25;
    hongqi @26;
    body @27;
    hyundaiCanfd @28;
    volkswagenMqbEvo @29;
    chryslerCusw @30;
    psa @31;
  }

  enum SteerControlType {
    torque @0;
    angle @1;

    curvatureDEPRECATED @2;
  }

  enum TransmissionType {
    unknown @0;
    automatic @1;  # Traditional auto, including DSG
    manual @2;  # True "stick shift" only
    direct @3;  # Electric vehicle or other direct drive
    cvt @4;
  }

  struct CarFw {
    ecu @0 :Ecu;
    fwVersion @1 :Data;
    address @2 :UInt32;
    subAddress @3 :UInt8;
    responseAddress @4 :UInt32;
    request @5 :List(Data);
    brand @6 :Text;
    bus @7 :UInt8;
    logging @8 :Bool;
    obdMultiplexing @9 :Bool;
  }

  enum Ecu {
    eps @0;
    abs @1;
    fwdRadar @2;
    fwdCamera @3;
    engine @4;
    unknown @5;
    transmission @8; # Transmission Control Module
    hybrid @18; # hybrid control unit, e.g. Chrysler's HCP, Honda's IMA Control Unit, Toyota's hybrid control computer
    srs @9; # airbag
    gateway @10; # can gateway
    hud @11; # heads up display
    combinationMeter @12; # instrument cluster
    electricBrakeBooster @15;
    shiftByWire @16;
    adas @19;
    cornerRadar @21;
    hvac @20;
    parkingAdas @7;  # parking assist system ECU, e.g. Toyota's IPAS, Hyundai's RSPA, etc.
    epb @22;  # electronic parking brake
    telematics @23;
    body @24;  # body control module

    # Toyota only
    dsu @6;

    # Honda only
    vsa @13; # Vehicle Stability Assist
    programmedFuelInjection @14;

    debug @17;
  }

  enum FingerprintSource {
    can @0;
    fw @1;
    fixed @2;
  }

  enum NetworkLocation {
    fwdCamera @0;  # Standard/default integration at LKAS camera
    gateway @1;    # Integration at vehicle's CAN gateway
  }

  enableGasInterceptorDEPRECATED @2 :Bool;
  enableCameraDEPRECATED @4 :Bool;
  enableApgsDEPRECATED @6 :Bool;
  steerRateCostDEPRECATED @33 :Float32;
  isPandaBlackDEPRECATED @39 :Bool;
  hasStockCameraDEPRECATED @57 :Bool;
  safetyParamDEPRECATED @10 :Int16;
  safetyModelDEPRECATED @9 :SafetyModel;
  safetyModelPassiveDEPRECATED @42 :SafetyModel = silent;
  minSpeedCanDEPRECATED @51 :Float32;
  communityFeatureDEPRECATED @46: Bool;
  startingAccelRateDEPRECATED @53 :Float32;
  steerMaxBPDEPRECATED @11 :List(Float32);
  steerMaxVDEPRECATED @12 :List(Float32);
  gasMaxBPDEPRECATED @13 :List(Float32);
  gasMaxVDEPRECATED @14 :List(Float32);
  brakeMaxBPDEPRECATED @15 :List(Float32);
  brakeMaxVDEPRECATED @16 :List(Float32);
  directAccelControlDEPRECATED @30 :Bool;
  maxSteeringAngleDegDEPRECATED @54 :Float32;
  longitudinalActuatorDelayLowerBoundDEPRECATEDDEPRECATED @61 :Float32;
}
