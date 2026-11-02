# Live

Plot, CAN, logs, inspector and video follow a live edge. A route opened live is a store whose last slot grows.

| URL | Source | Signaling | Data |
|---|---|---|---|
| `/<dongle>/live` | the device's webrtcd | athena startStream | camera tracks + cereal services |
| `/live/local` | a webrtcd on this PC (openpilot, sim, tools/replay) | the daemon forwards the offer to 127.0.0.1:5001 | same |
| `/live/panda` | WebUSB to a panda, receive only: never sends CAN or changes the safety mode | none | CAN, panda health |
| `/live/replay` | a route's rlog played back as if live | none | cereal services |

## Device protocol

The offer's request gains optional fields; a device without them answers as stock webrtcd does (JSON carState and
deviceState on "data"):

```
startStream(sdp, enabled, services?: [name], cameras?: [road|wideRoad|driver])
data channel "data"    unchanged: JSON {type, logMonoTime, valid, data}, and the existing ping/pong/disconnect
data channel "cereal"  binary, device → browser: one capnp Event per message, exactly as logged, for `services`
```
