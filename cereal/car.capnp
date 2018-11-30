using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Java = import "./include/java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("Car");

@0x8e2af1e708af8b8d;

# ******* events causing controls state machine transition *******

struct CarEvent @0x9b1657f34caf3ad3 {
  name @0 :EventName;
  enable @1 :Bool;
  noEntry @2 :Bool;
  warning @3 :Bool;
  userDisable @4 :Bool;
  softDisable @5 :Bool;
  immediateDisable @6 :Bool;
  preEnable @7 :Bool;
  permanent @8 :Bool;

  enum EventName @0xbaa8c5d505f727de {
    # TODO: copy from error list
    commIssue @0;
    steerUnavailable @1;
    brakeUnavailable @2;
    gasUnavailable @3;
    wrongGear @4;
    doorOpen @5;
    seatbeltNotLatched @6;
    espDisabled @7;
    wrongCarMode @8;
    steerTempUnavailable @9;
    reverseGear @10;
    buttonCancel @11;
    buttonEnable @12;
    pedalPressed @13;
    cruiseDisabled @14;
    radarCommIssue @15;
    dataNeeded @16;
    speedTooLow @17;
    outOfSpace @18;
    overheat @19;
    calibrationIncomplete @20;
    calibrationInvalid @21;
    controlsMismatch @22;
    pcmEnable @23;
    pcmDisable @24;
    noTarget @25;
    radarFault @26;
    modelCommIssue @27;
    brakeHold @28;
    parkBrake @29;
    manualRestart @30;
    lowSpeedLockout @31;
    plannerError @32;
    ipasOverride @33;
    debugAlert @34;
    steerTempUnavailableMute @35;
    resumeRequired @36;
    preDriverDistracted @37;
    promptDriverDistracted @38;
    driverDistracted @39;
    geofence @40;
    driverMonitorOn @41;
    driverMonitorOff @42;
    preDriverUnresponsive @43;
    promptDriverUnresponsive @44;
    driverUnresponsive @45;
    belowSteerSpeed @46;
    calibrationProgress @47;
    lowBattery @48;
    invalidGiraffeHonda @49;
  }
}

# ******* main car state @ 100hz *******
# all speeds in m/s

struct CarState {
  errorsDEPRECATED @0 :List(CarEvent.EventName);
  events @13 :List(CarEvent);

  # car speed
  vEgo @1 :Float32;         # best estimate of speed
  aEgo @16 :Float32;        # best estimate of acceleration
  vEgoRaw @17 :Float32;     # unfiltered speed from CAN sensors
  yawRate @22 :Float32;     # best estimate of yaw rate
  standstill @18 :Bool;
  wheelSpeeds @2 :WheelSpeeds;

  # gas pedal, 0.0-1.0
  gas @3 :Float32;        # this is user + computer
  gasPressed @4 :Bool;    # this is user pedal only

  # brake pedal, 0.0-1.0
  brake @5 :Float32;      # this is user pedal only
  brakePressed @6 :Bool;  # this is user pedal only
  brakeLights @19 :Bool;

  # steering wheel
  steeringAngle @7 :Float32;   # deg
  steeringRate @15 :Float32;   # deg/s
  steeringTorque @8 :Float32;  # TODO: standardize units
  steeringPressed @9 :Bool;    # if the user is using the steering wheel

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

  # which packets this state came from
  canMonoTimes @12: List(UInt64);

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
    available @2 :Bool;
    speedOffset @3 :Float32;
    standstill @4 :Bool;
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
    }
  }
}

# ******* radar state @ 20hz *******

struct RadarState {
  errors @0 :List(Error);
  points @1 :List(RadarPoint);

  # which packets this state came from
  canMonoTimes @2 :List(UInt64);

  enum Error {
    commIssue @0;
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
}

# ******* car controls @ 100hz *******

struct CarControl {
  # must be true for any actuator commands to work
  enabled @0 :Bool;
  active @7 :Bool;

  gasDEPRECATED @1 :Float32;
  brakeDEPRECATED @2 :Float32;
  steeringTorqueDEPRECATED @3 :Float32;

  actuators @6 :Actuators;

  cruiseControl @4 :CruiseControl;
  hudControl @5 :HUDControl;

  struct Actuators {
    # range from 0.0 - 1.0
    gas @0: Float32;
    brake @1: Float32;
    # range from -1.0 - 1.0
    steer @2: Float32;
    steerAngle @3: Float32;
  }

  struct CruiseControl {
    cancel @0: Bool;
    override @1: Bool;
    speedOverride @2: Float32;
    accelOverride @3: Float32;
  }

  struct HUDControl {
    speedVisible @0: Bool;
    setSpeed @1: Float32;
    lanesVisible @2: Bool;
    leadVisible @3: Bool;
    visualAlert @4: VisualAlert;
    audibleAlert @5: AudibleAlert;

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
    }

    enum AudibleAlert {
      # these are the choices from the Honda
      # map as good as you can for your car
      none @0;
      chimeEngage @1;
      chimeDisengage @2;
      chimeError @3;
      chimeWarning1 @4;
      chimeWarning2 @5;
      chimeWarningRepeat @6;
      chimePrompt @7;
    }
  }
}

# ****** car param ******

struct CarParams {
  carName @0 :Text;
  radarNameDEPRECATED @1 :Text;
  carFingerprint @2 :Text;

  enableSteerDEPRECATED @3 :Bool;
  enableGasInterceptor @4 :Bool;
  enableBrakeDEPRECATED @5 :Bool;
  enableCruise @6 :Bool;
  enableCamera @26 :Bool;
  enableDsu @27 :Bool; # driving support unit
  enableApgs @28 :Bool; # advanced parking guidance system

  minEnableSpeed @17 :Float32;
  minSteerSpeed @49 :Float32;
  safetyModel @18 :Int16;
  safetyParam @41 :Int16;

  steerMaxBP @19 :List(Float32);
  steerMaxV @20 :List(Float32);
  gasMaxBP @21 :List(Float32);
  gasMaxV @22 :List(Float32);
  brakeMaxBP @23 :List(Float32);
  brakeMaxV @24 :List(Float32);

  longPidDeadzoneBP @32 :List(Float32);
  longPidDeadzoneV @33 :List(Float32);

  enum SafetyModels {
    # does NOT match board setting
    noOutput @0;
    honda @1;
    toyota @2;
    elm327 @3;
    gm @4;
    hondaBosch @5;
    ford @6;
    cadillac @7;
    hyundai @8;
    chrysler @9;
    tesla @10;
  }

  # things about the car in the manual
  mass @7 :Float32;             # [kg] running weight
  wheelbase @8 :Float32;        # [m] distance from rear to front axle
  centerToFront @9 :Float32;   # [m] GC distance to front axle
  steerRatio @10 :Float32;       # [] ratio between front wheels and steering wheel angles
  steerRatioRear @11 :Float32;  # [] rear steering ratio wrt front steering (usually 0)

  # things we can derive
  rotationalInertia @12 :Float32;    # [kg*m2] body rotational inertia
  tireStiffnessFront @13 :Float32;   # [N/rad] front tire coeff of stiff
  tireStiffnessRear @14 :Float32;    # [N/rad] rear tire coeff of stiff

  # Kp and Ki for the lateral control
  steerKpBP @42 :List(Float32);
  steerKpV @43 :List(Float32);
  steerKiBP @44 :List(Float32);
  steerKiV @45 :List(Float32);
  steerKpDEPRECATED @15 :Float32;
  steerKiDEPRECATED @16 :Float32;
  steerKf @25 :Float32;

  # Kp and Ki for the longitudinal control
  longitudinalKpBP @36 :List(Float32);
  longitudinalKpV @37 :List(Float32);
  longitudinalKiBP @38 :List(Float32);
  longitudinalKiV @39 :List(Float32);

  steerLimitAlert @29 :Bool;

  vEgoStopping @30 :Float32; # Speed at which the car goes into stopping state
  directAccelControl @31 :Bool; # Does the car have direct accel control or just gas/brake
  stoppingControl @34 :Bool; # Does the car allows full control even at lows speeds when stopping
  startAccel @35 :Float32; # Required acceleraton to overcome creep braking
  steerRateCost @40 :Float32; # Lateral MPC cost on steering rate
  steerControlType @46 :SteerControlType;
  radarOffCan @47 :Bool; # True when radar objects aren't visible on CAN

  steerActuatorDelay @48 :Float32; # Steering wheel actuator delay in seconds

  enum SteerControlType {
    torque @0;
    angle @1;
  }
}
