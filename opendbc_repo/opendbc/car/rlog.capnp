@0xce500edaaae36b0e;

# Minimal schema for parsing rlog CAN messages
# Subset of cereal/log.capnp

using Car = import "car.capnp";

struct CanData {
  address @0 :UInt32;
  busTimeDEPRECATED @1 :UInt16;
  dat @2 :Data;
  src @3 :UInt8;
}

# Field types through safetyModel preserve the historical data layout.
struct PandaState {
  voltage @0 :UInt32;
  current @1 :UInt32;
  ignitionLine @2 :Bool;
  controlsAllowed @3 :Bool;
  gasInterceptorDetectedDEPRECATED @4 :Bool;
  startedSignalDetectedDEPRECATED @5 :Bool;
  hasGpsDEPRECATED @6 :Bool;
  rxBufferOverflow @7 :UInt32;
  txBufferOverflow @8 :UInt32;
  gmlanSendErrsDEPRECATED @9 :UInt32;
  pandaType @10 :UInt16;
  fanSpeedRpmDEPRECATED @11 :UInt16;
  usbPowerModeDEPRECATED @12 :UInt16;
  ignitionCan @13 :Bool;
  safetyModel @14 :Car.CarParams.SafetyModel;
}

struct Event {
  logMonoTime @0 :UInt64;
  # This field is outside the union in cereal; placing it inside shifts later discriminants.
  valid @67 :Bool = true;

  union {
    initData @1 :Void;
    frame @2 :Void;
    gpsNMEA @3 :Void;
    sensorEventDEPRECATED @4 :Void;
    can @5 :List(CanData);
    reserved6 @6 :Void;
    reserved7 @7 :Void;
    reserved8 @8 :Void;
    reserved9 @9 :Void;
    reserved10 @10 :Void;
    reserved11 @11 :Void;
    pandaStateDEPRECATED @12 :PandaState;
    reserved13 @13 :Void;
    reserved14 @14 :Void;
    reserved15 @15 :Void;
    reserved16 @16 :Void;
    reserved17 @17 :Void;
    reserved18 @18 :Void;
    reserved19 @19 :Void;
    reserved20 @20 :Void;
    reserved21 @21 :Void;
    reserved22 @22 :Void;
    reserved23 @23 :Void;
    reserved24 @24 :Void;
    reserved25 @25 :Void;
    reserved26 @26 :Void;
    reserved27 @27 :Void;
    reserved28 @28 :Void;
    reserved29 @29 :Void;
    reserved30 @30 :Void;
    reserved31 @31 :Void;
    reserved32 @32 :Void;
    reserved33 @33 :Void;
    reserved34 @34 :Void;
    reserved35 @35 :Void;
    reserved36 @36 :Void;
    reserved37 @37 :Void;
    reserved38 @38 :Void;
    reserved39 @39 :Void;
    reserved40 @40 :Void;
    reserved41 @41 :Void;
    reserved42 @42 :Void;
    reserved43 @43 :Void;
    reserved44 @44 :Void;
    reserved45 @45 :Void;
    reserved46 @46 :Void;
    reserved47 @47 :Void;
    reserved48 @48 :Void;
    reserved49 @49 :Void;
    reserved50 @50 :Void;
    reserved51 @51 :Void;
    reserved52 @52 :Void;
    reserved53 @53 :Void;
    reserved54 @54 :Void;
    reserved55 @55 :Void;
    reserved56 @56 :Void;
    reserved57 @57 :Void;
    reserved58 @58 :Void;
    reserved59 @59 :Void;
    reserved60 @60 :Void;
    reserved61 @61 :Void;
    reserved62 @62 :Void;
    reserved63 @63 :Void;
    reserved64 @64 :Void;
    reserved65 @65 :Void;
    reserved66 @66 :Void;
    reserved68 @68 :Void;
    carParams @69 :Car.CarParams;
    reserved70 @70 :Void;
    reserved71 @71 :Void;
    reserved72 @72 :Void;
    reserved73 @73 :Void;
    reserved74 @74 :Void;
    reserved75 @75 :Void;
    reserved76 @76 :Void;
    reserved77 @77 :Void;
    reserved78 @78 :Void;
    reserved79 @79 :Void;
    reserved80 @80 :Void;
    pandaStates @81 :List(PandaState);
  }
}
