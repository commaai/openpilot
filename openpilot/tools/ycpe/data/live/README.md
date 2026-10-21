# Live

Plot, CAN, logs, inspector and video follow a live edge. A route opened live is a store whose last slot grows.

| URL | Source | Signaling | Data |
|---|---|---|---|
| `/<dongle>/live` | the device's webrtcd | athena startStream | camera tracks + cereal services |
| `/live/local` | a webrtcd on this PC (openpilot, sim, tools/replay) | the daemon forwards the offer to 127.0.0.1:5001 | same |
| `/live/panda` | WebUSB to a panda, receive only: never sends CAN or changes the safety mode | none | CAN, panda health |
| `/live/replay` | a route's rlog played back as if live | none | cereal services |
| any route | when the device is online, its segments' files over WebRTC (the "p2p" source) | athena startStream | rlog, qlog, cameras |

## Device protocol

The offer's request gains optional fields; a device without them answers as stock webrtcd does (JSON carState and
deviceState on "data"):

```
startStream(sdp, enabled, services?: [name], cameras?: [road|wideRoad|driver], files?: bool)
data channel "data"    unchanged: JSON {type, logMonoTime, valid, data}, and the existing ping/pong/disconnect
data channel "cereal"  binary, device → browser: one capnp Event per message, exactly as logged, for `services`
data channel "files"   request/response, read-only, confined to the log root:
  → {id, op: "list"}                              ← {id, routes: [{name, segments: [{n, files: {name: size}}]}]}
  → {id, op: "read", path: "<route>--<n>/<file>", offset?, length?}
  ← {id, size} then binary chunks (8-byte id + offset header, ≤64 KB), ← {id, done} | {id, error}
  flow control on bufferedAmount; one read streams at a time per channel; cancel: → {id, op: "cancel"}
```
