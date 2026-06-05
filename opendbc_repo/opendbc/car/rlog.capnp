@0xce500edaaae36b0e;

# Minimal schema for parsing rlog CAN messages
# Subset of cereal/log.capnp

struct CanData {
  address @0 :UInt32;
  busTimeDEPRECATED @1 :UInt16;
  dat @2 :Data;
  src @3 :UInt8;
}

struct Event {
  logMonoTime @0 :UInt64;

  union {
    initData @1 :Void;
    frame @2 :Void;
    gpsNMEA @3 :Void;
    sensorEventDEPRECATED @4 :Void;
    can @5 :List(CanData);
  }
}
