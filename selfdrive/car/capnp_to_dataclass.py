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
"""

if __name__ == '__main__':

  TXT = input('Paste one struct only')

  in_struct = False

  builder = []

  for line in TXT.splitlines():
    line = line.strip()
    if re.search(':.*;', line):
      if not in_struct:
        # print(line)
        name, typ, cmt = re.search('([a-zA-Z]+)\s*@\s*\d+\s*:\s*([a-zA-Z0-9\(\)]+)(?:.*#(.*))?', line.strip()).groups()  # noqa # type: ignore
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
