using Cxx = import "c++.capnp";
$Cxx.namespace("cereal");

@0x8e2af1e708af8b8d;

# ******* main car state @ 100hz *******
# all speeds in m/s

struct CarState {
  errors @0: List(Error);

  # car speed
  vEgo @1 :Float32;       # best estimate of speed
  wheelSpeeds @2 :WheelSpeeds;

  # gas pedal, 0.0-1.0
  gas @3 :Float32;        # this is user + computer
  gasPressed @4 :Bool;    # this is user pedal only

  # brake pedal, 0.0-1.0
  brake @5 :Float32;      # this is user pedal only
  brakePressed @6 :Bool;  # this is user pedal only

  # steering wheel
  steeringAngle @7 :Float32;   # deg
  steeringTorque @8 :Float32;  # TODO: standardize units
  steeringPressed @9 :Bool;    # if the user is using the steering wheel

  # cruise state
  cruiseState @10 :CruiseState;

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
    enabled @0: Bool;
    speed @1: Float32;
  }

  enum Error {
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
    steerTemporarilyUnavailable @9;
    reverseGear @10;
    # ...
  }

  # send on change
  struct ButtonEvent {
    pressed @0: Bool;
    type @1: Type;

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
  errors @0: List(Error);
  points @1: List(RadarPoint);

  # which packets this state came from
  canMonoTimes @2: List(UInt64);

  enum Error {
    notValid @0;
  }

  # similar to LiveTracks
  # is one timestamp valid for all? I think so
  struct RadarPoint {
    trackId @0: UInt64;  # no trackId reuse

    # these 3 are the minimum required
    dRel @1: Float32; # m from the front bumper of the car
    yRel @2: Float32; # m
    vRel @3: Float32; # m/s

    # these are optional and valid if they are not NaN
    aRel @4: Float32; # m/s^2
    yvRel @5: Float32; # m/s
  }
}

# ******* car controls @ 100hz *******

struct CarControl {
  # must be true for any actuator commands to work
  enabled @0: Bool;

  # range from 0.0 - 1.0
  gas @1: Float32;
  brake @2: Float32;

  # range from -1.0 - 1.0
  steeringTorque @3 :Float32;

  cruiseControl @4: CruiseControl;
  hudControl @5: HUDControl;

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

