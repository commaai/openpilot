import re

TYPE_MAP = {
  'Text': 'str',
  'Bool': 'bool',
  'Int16': 'int',
  'UInt32': 'int',
  'UInt64': 'int',
  'Float32': 'float',
  'Data': 'bytes',
  'List': 'list',
}

TXT = """
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
"""

if __name__ == '__main__':

  if not TXT.strip():
    TXT = input('Paste one struct only')

  in_struct = False

  builder = []

  for line in TXT.splitlines():
    line = line.strip()
    if re.search(':.*;', line):
      if not in_struct:
        # print(line)
        name, typ, cmt = re.search('([a-zA-Z]+)\s*@\s*\d+\s*:\s*([a-zA-Z0-9\(\)]+)(?:.*#(.*))?', line.strip()).groups()  # type: ignore  # noqa
        # print((name, typ, cmt))
        if name.endswith('DEPRECATED'):
          continue

        if 'List' in typ:
          second_typ = typ.split("(")[1][:-1]
          typ = f'list[{TYPE_MAP.get(second_typ, second_typ)}]'

        new_typ = TYPE_MAP.get(typ, typ)
        # print(f'  {name}: {new_typ} = auto_field()')
        # print()
        new_cmt = f'  # {cmt.strip()}' if cmt else ''
        builder.append(f'  {name}: {new_typ} = auto_field(){new_cmt}')
    elif re.search('{', line):
      in_struct = True
    elif re.search('}', line):
      in_struct = False
    elif line == '':
      builder.append('')

  print('\n'.join(builder))
