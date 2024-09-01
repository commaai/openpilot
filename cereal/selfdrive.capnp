using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Car = import "car.capnp";

@0x87b6c28418fd19cd;

enum LongitudinalPersonality @0xd692e23d1a247d99 {
  aggressive @0;
  standard @1;
  relaxed @2;
}

struct SelfdriveState @0xb52430dc48f4a83b {
  # high level system state
  state @0 :OpenpilotState;
  enabled @1 :Bool;
  active @2 :Bool;
  engageable @9 :Bool;  # can OP be engaged?

  # UI alerts
  alertText1 @3 :Text;
  alertText2 @4 :Text;
  alertStatus @5 :AlertStatus;
  alertSize @6 :AlertSize;
  alertType @7 :Text;
  alertSound @8 :Car.CarControl.HUDControl.AudibleAlert;

  # configurable driving settings
  experimentalMode @10 :Bool;
  personality @11 :LongitudinalPersonality;

  enum OpenpilotState @0xdbe58b96d2d1ac61 {
    disabled @0;
    preEnabled @1;
    enabled @2;
    softDisabling @3;
    overriding @4;  # superset of overriding with steering or accelerator
  }

  enum AlertStatus @0xa0d0dcd113193c62 {
    normal @0;
    userPrompt @1;
    critical @2;
  }

  enum AlertSize @0xe98bb99d6e985f64 {
    none @0;
    small @1;
    mid @2;
    full @3;
  }
}

