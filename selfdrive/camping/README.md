Camping mode receiver

This adds an offroad casting receiver launcher (`campingd`).

Miracast-only receiver
- Uses `miracle-wifid` (MiracleCast Wi‑Fi Direct daemon)
- Optional helper: `miracle-sinkctl -a` to auto-accept connections

Install/build
1) Submodule: `selfdrive/camping/miraclecast` is a git submodule (cloned on device during build)
2) Build happens automatically at the end of the normal scons build on device
   - Script: `selfdrive/camping/build_miracast.sh`
   - Outputs to `selfdrive/camping/bin/` and auto-copies to `/data/camping/bin/`
3) Alternatively, run the build script manually and/or `./scripts/install_camping_receiver.sh`

Runtime
- Manager starts `campingd` offroad when Param `CampingMode` is true (toggle in Settings → Developer)
- `campingd` starts `/data/camping/bin/miracle-wifid` and, if present, `/data/camping/bin/miracle-sinkctl -a`
- If not found, it stays idle until `CampingMode` is turned off

Notes
- Requires Wi‑Fi Direct (P2P) support on the device; Comma 3X has Wi‑Fi hardware but success depends on driver capabilities in agnos.
- MiracleCast typically needs root/CAP_NET_ADMIN and may disrupt or temporarily commandeer Wi‑Fi while active.
