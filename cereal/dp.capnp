using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

using Java = import "./include/java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("dp");

@0xbfa7e645486440c7;

# dp.capnp: a home for deprecated structs

# dp
struct DragonConf {
  dpThermalStarted @0 :Bool;
  dpThermalOverheat @1 :Bool;
  dpAtl @2 :Bool;
  dpDashcamd @3 :Bool;
  dpAutoShutdown @4 :Bool;
  dpAthenad @5 :Bool;
  dpUploader @6 :Bool;
  dpSteeringOnSignal @7 :Bool;
  dpSignalOffDelay @8 :UInt8;
  dpLateralMode @9 :UInt8;
  dpLcMinMph @10 :Float32;
  dpLcAutoCont @11 :Bool;
  dpLcAutoMinMph @12 :Float32;
  dpLcAutoDelay @13 :Float32;
  dpAllowGas @14 :Bool;
  dpFollowingProfileCtrl @15 :Bool;
  dpFollowingProfile @16 :UInt8;
  dpAccelProfileCtrl @17 :Bool;
  dpAccelProfile @18 :UInt8;
  dpGearCheck @19 :Bool;
  dpSpeedCheck @20 :Bool;
  dpUiDisplayMode @21 :UInt8;
  dpUiSpeed @22 :Bool;
  dpUiEvent @23 :Bool;
  dpUiMaxSpeed @24 :Bool;
  dpUiFace @25 :Bool;
  dpUiLane @26 :Bool;
  dpUiLead @27 :Bool;
  dpUiDev @28 :Bool;
  dpUiDevMini @29 :Bool;
  dpUiBlinker @30 :Bool;
  dpUiBrightness @31 :UInt8;
  dpUiVolume @32 :Int8;
  dpAppExtGps @33 :Bool;
  dpAppTomtom @34 :Bool;
  dpAppTomtomAuto @35 :Bool;
  dpAppTomtomManual @36 :Int8;
  dpAppMixplorer @37 :Bool;
  dpAppMixplorerManual @38 :Int8;
  dpCarDetected @39 :Text;
  dpToyotaLdw @40 :Bool;
  dpToyotaSng @41 :Bool;
  dpVwPanda @42 :Bool;
  dpVwTimebombAssist @43 :Bool;
  dpIpAddr @44 :Text;
  dpCameraOffset @45 :Int8;
  dpPathOffset @46 :Int8;
  dpLocale @47 :Text;
  dpSrLearner @48 :Bool;
  dpSrCustom @49 :Float32;
  dpSrStock @50 :Float32;
  dpDebug @51 :Bool;
}