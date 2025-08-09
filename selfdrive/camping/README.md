Camping mode receiver

This adds an offroad casting receiver launcher (`campingd`).

Supported receivers (best-effort)
- Preferred: `openscreen-cast-receiver` (Open Screen cast mirroring)
- DLNA audio: `gmediarender`
- Optional: `miracle-wifid` (MiracleCast daemon; requires Wi‑Fi P2P support)

Install options
1) Ship/commit the receiver binary into this repo at `selfdrive/camping/bin/`:
   - `selfdrive/camping/bin/openscreen-cast-receiver` (or `gmediarender`)
   - Then run the installer to deploy to the device:
     - `./scripts/install_camping_receiver.sh`
2) Manually place binaries on-device at `/data/camping/bin/` and mark executable

Runtime
- Manager starts `campingd` offroad when Param `CampingMode` is true (toggle in Settings → Developer)
- `campingd` checks in order:
  1) `/data/camping/bin/openscreen-cast-receiver`
  2) `/usr/bin/openscreen-cast-receiver`
  3) `/data/camping/bin/gmediarender`
  4) `/usr/bin/gmediarender`
  5) `/usr/bin/mkchromecast`
  6) `/data/camping/bin/miracle-wifid`
- If none found, it stays idle until `CampingMode` is turned off

Notes
- True Google Cast receivers require Google Cast SDK/certification; Open Screen is a best-effort alternative.
- DLNA audio via gmediarender can provide basic casting for audio apps.
- MiracleCast typically needs root/CAP_NET_ADMIN and may disrupt Wi‑Fi while active.
