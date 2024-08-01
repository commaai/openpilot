using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");

@0xa086df597ef5d7a0;

# Geometry
struct Point {
  x @0: Float64;
  y @1: Float64;
  z @2: Float64;
}

struct PolyLine {
  points @0: List(Point);
}

# Map features
struct Lane {
  id @0 :Text;

  leftBoundary @1 :LaneBoundary;
  rightBoundary @2 :LaneBoundary;

  leftAdjacentId @3 :Text;
  rightAdjacentId @4 :Text;

  inboundIds @5 :List(Text);
  outboundIds @6 :List(Text);

  struct LaneBoundary {
    polyLine @0 :PolyLine;
    startHeading @1 :Float32; # WRT north
  }
}

# Map tiles
struct TileSummary {
  version @0 :Text;
  updatedAt @1 :UInt64; # Millis since epoch

  level @2 :UInt8;
  x @3 :UInt16;
  y @4 :UInt16;
}

struct MapTile {
  summary @0 :TileSummary;
  lanes @1 :List(Lane);
}
