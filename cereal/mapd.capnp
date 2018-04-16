using Cxx = import "c++.capnp";
$Cxx.namespace("cereal");

using Java = import "java.capnp";
$Java.package("ai.comma.openpilot.cereal");
$Java.outerClassname("Map");

using Log = import "log.capnp";

@0xe1a6ab330ea7205f;

struct MapdRequest {
  pos @0 :Log.ECEFPoint;
}

struct MapdResponse {
  kfs @0 :List(Log.OrbKeyFrame);
}
