using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Java = import "./include/java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("Ui");

using Car = import "car.capnp";

@0xce6ca45dddcd5317;

struct UIButtonInfo {
    # button ID 0..5
    btn_id @0 :Int8;
    # internal button name
    btn_name @0 :Text;
    # display label for button (3 chars)
    btn_label @1 :Text;
    # buttons status: 0 = DISABLED, 1 = AVAILABLE, 2 = ENABLED, 3 = WARNING, 9 = NOT AVAILABLE
    btn_status @2 :Int16;
    # small font label shows below the main label, max 7 chars
    btn_label2 @3 :Text;
}

struct UIButtonStatus {
    # button ID 0..5
    btn_id @0 :Int8;
    # buttons status: 0 = DISABLED, 1 = AVAILABLE, 2 = ENABLED, 3 = WARNING, 9 = NOT AVAILABLE
    btn_status @2 :Int16;
}

struct UICustomAlert {
    ca_status @0 :Int8;
    ca_text @1 :Text;
}

struct UISetCar {
    ic_carFolder@0 :Text;
    ic_carName @1 :Text;
}

struct UIEvent {
    # in nanoseconds?
    logMonoTime @0 :UInt64;

    union {
        uiButtonInfo @1 :UIButtonInfo;
        uiCustomAlert @2 :UICustomAlert;
        uiSetCar @3 :UISetCar;
        uiButtonStatus @4 :UIButtonStatus;
    }
}