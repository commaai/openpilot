using Cxx = import "c++.capnp";
$Cxx.namespace("cereal");

using Java = import "java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("Car");

@0x8e2af1e708af8b8d;

# ******* events causing controls state machine transition *******

struct CarEvent @0x9b1657f34caf3ad3 {
  name @0 :EventName;
  enable @1 :Bool;
  preEnable @7 :Bool;
  noEntry @2 :Bool;
  warning @3 :Bool;
  userDisable @4 :Bool;
  softDisable @5 :Bool;
  immediateDisable @6 :Bool;

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
    calibrationInProgress @20;
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
  }
}

# ******* main car state @ 100hz *******
# all speeds in m/s

struct CarState {
  errorsDEPRECATED @0 :List(CarEvent.EventName);
  events @13 :List(CarEvent);

  # car speed
  vEgo @1 :Float32;       # best estimate of speed
  aEgo @16 :Float32;       # best estimate of acceleration
  vEgoRaw @17 :Float32;       # unfiltered speed
  standstill @18 :Bool;
  wheelSpeeds @2 :WheelSpeeds;

  # gas pedal, 0.0-1.0
  gas @3 :Float32;        # this is user + computer
  gasPressed @4 :Bool;    # this is user pedal only

  # brake pedal, 0.0-1.0
  brake @5 :Float32;      # this is user pedal only
  brakePressed @6 :Bool;  # this is user pedal only

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
  }
}

# ******* car controls @ 100hz *******

struct CarControl {
  # must be true for any actuator commands to work
  enabled @0 :Bool;

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
      beepSingle @1;
      beepTriple @2;
      beepRepeated @3;
      chimeSingle @4;
      chimeDouble @5;
      chimeRepeated @6;
      chimeContinuous @7;
    }
  }
}

# ****** car param ******

struct CarParams {
  carName @0 :Text;
  radarName @1 :Text;
  carFingerprint @2 :Text;

  enableSteer @3 :Bool;
  enableGas @4 :Bool;
  enableBrake @5 :Bool;
  enableCruise @6 :Bool;
  enableCamera @27 :Bool;
  enableDsu @28 :Bool; # driving support unit
  enableApgs @29 :Bool; # advanced parking guidance system

  minEnableSpeed @18 :Float32;
  safetyModel @19 :Int16;

  steerMaxBP @20 :List(Float32);
  steerMaxV @21 :List(Float32);
  gasMaxBP @22 :List(Float32);
  gasMaxV @23 :List(Float32);
  brakeMaxBP @24 :List(Float32);
  brakeMaxV @25 :List(Float32);

  enum SafetyModels {
    # does NOT match board setting
    noOutput @0;
    honda @1;
    toyota @2;
    elm327 @3;
  }

  # things about the car in the manual
  m @7 :Float32;     # [kg] running weight
  l @8 :Float32;     # [m] wheelbase
  sR @9 :Float32;    # [] steering ratio
  aF @10 :Float32;   # [m] GC distance to front axle
  aR @11 :Float32;   # [m] GC distance to rear axle
  chi @12 :Float32;  # [] rear steering ratio wrt front steering (usually 0)

  # things we can derive
  j @13 :Float32;    # [kg*m2] body rotational inertia
  cF @14 :Float32;   # [N/rad] front tire coeff of stiff
  cR @15 :Float32;   # [N/rad] rear tire coeff of stiff

  # Kp and Ki for the lateral control
  steerKp @16 :Float32;
  steerKi @17 :Float32;
  steerKf @26 :Float32;

  steerLimitAlert @30 :Bool;

  # TODO: Kp and Ki for long control, perhaps not needed?
}
