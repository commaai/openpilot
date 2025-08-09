Camping mode receiver

This adds an offroad casting receiver launcher (`campingd`).

Miracast-only receiver
- Uses `miracle-wifid` (MiracleCast Wi‑Fi Direct daemon)
- Optional helper: `miracle-sinkctl -a` to auto-accept connections

Install options
1) Build MiracleCast via submodule helper:
   - `selfdrive/camping/build_miracast.sh`
   - Binaries will be in `selfdrive/camping/bin/`
   - Then install to the device: `./scripts/install_camping_receiver.sh`
2) Or manually place `miracle-wifid` and `miracle-sinkctl` into `/data/camping/bin/` and mark executable

Runtime
- Manager starts `campingd` offroad when Param `CampingMode` is true (toggle in Settings → Developer)
- `campingd` starts `/data/camping/bin/miracle-wifid` and, if present, `/data/camping/bin/miracle-sinkctl -a`
- If not found, it stays idle until `CampingMode` is turned off

Notes
- Requires Wi‑Fi Direct (P2P) support on the device; Comma 3X has Wi‑Fi hardware but success depends on driver capabilities in agnos.
- MiracleCast typically needs root/CAP_NET_ADMIN and may disrupt or temporarily commandeer Wi‑Fi while active.
