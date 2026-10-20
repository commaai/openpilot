#!/usr/bin/env python3
"""writes panda.bin.gz: two seconds of the demo route's CAN as a panda sends it over USB, packed by panda's own library,
and its health and CAN health packets (openpilot venv, PYTHONPATH=<checkout root>)

layout: u32 header length, header JSON (control responses and what the frames should decode to), then one
[u32 microseconds, u32 length, bytes] per rlog can event"""
import base64
import gzip
import json
import struct
from pathlib import Path

from panda import Panda, pack_can_buffer, calculate_checksum
from openpilot.tools.lib.logreader import LogReader

ROUTE = "5beb9b58bd12b691/0000010a--a51155e496/2/r"
SECONDS = 2


def pack(frames):
  out = bytearray()
  for address, dat, src in frames:
    packet = pack_can_buffer([(address, dat, src & 7)])[0]
    if src >= 128:  # returned (128 + bus) and rejected (192 + bus) frames, which pack_can_buffer can't make
      packet[1] |= 1 if src >= 192 else 2
      packet[5] = 0
      packet[5] = calculate_checksum(packet)
    out += packet
  return out


def health(ps):
  temperature = 40 + 35  # pandaStates has none: 35 C
  flags = sum(bit << i for i, bit in enumerate([ps.ignitionLine, ps.ignitionCan, ps.controlsAllowed, ps.powerSaveEnabled, ps.heartbeatLost,
                                                ps.safetyRxChecksInvalid, False, ps.nmiReset, ps.hardfaultReset]))
  faults = sum(1 << f.raw for f in ps.faults)
  return Panda.HEALTH_STRUCT.pack(ps.uptime, ps.voltage, ps.current, ps.safetyTxBlocked, ps.safetyRxInvalid, ps.txBufferOverflow,
                                  ps.rxBufferOverflow, faults, flags, ps.harnessStatus.raw, ps.safetyModel.raw, ps.safetyParam,
                                  ps.faultStatus.raw, ps.alternativeExperience, round(ps.interruptLoad * 255), ps.fanPower, ps.spiErrorCount,
                                  round(ps.sbu1Voltage * 1000), round(ps.sbu2Voltage * 1000), ps.soundOutputLevel, temperature)


def b64(b):
  return base64.b64encode(b).decode()


def can_health(cs):
  return Panda.CAN_HEALTH_STRUCT.pack(cs.busOff, cs.busOffCnt, cs.errorWarning, cs.errorPassive, cs.lastError.raw, cs.lastStoredError.raw,
                                      cs.lastDataError.raw, cs.lastDataStoredError.raw, cs.receiveErrorCnt, cs.transmitErrorCnt,
                                      cs.totalErrorCnt, cs.totalTxLostCnt, cs.totalRxLostCnt, cs.totalTxCnt, cs.totalRxCnt, cs.totalFwdCnt, 0,
                                      cs.canSpeed, cs.canDataSpeed, cs.canfdEnabled, cs.brsEnabled, cs.canfdNonIso, cs.irq0CallRate,
                                      cs.irq1CallRate, cs.irq2CallRate, cs.canCoreResetCnt)


transfers, frames, state, t0 = bytearray(), [], None, None
for m in LogReader(ROUTE):
  if m.which() == "can":
    t0 = t0 or m.logMonoTime
    if m.logMonoTime - t0 >= SECONDS * 1e9:
      break
    batch = [(c.address, bytes(c.dat), c.src) for c in m.can]
    data = pack(batch)
    transfers += struct.pack("<II", (m.logMonoTime - t0) // 1000, len(data)) + data
    frames += batch
  elif m.which() == "pandaStates" and state is None:
    state = m.pandaStates[0]


header = json.dumps({
  "route": ROUTE,
  "serial": "fake0123456789abcdef",
  "hwType": state.pandaType.raw,
  "versions": [Panda.HEALTH_PACKET_VERSION, Panda.CAN_PACKET_VERSION],
  "health": b64(health(state)),
  "canHealth": [b64(can_health(cs)) for cs in (state.canState0, state.canState1, state.canState2)],
  "state": {k: v for k, v in state.to_dict().items() if k != "deprecated" and "RESERVED" not in k},
  "frames": len(frames),
  "bySrc": {src: sum(f[2] == src for f in frames) for src in sorted({f[2] for f in frames})},
  "bytes": sum(len(f[1]) for f in frames),
  "first": [[address, src, dat.hex()] for address, dat, src in frames[:20]],
  "last": [[address, src, dat.hex()] for address, dat, src in frames[-20:]],
}).encode()
out = Path(__file__).with_name("panda.bin.gz")
out.write_bytes(gzip.compress(struct.pack("<I", len(header)) + header + transfers, mtime=0))
print(f"{out}: {len(frames)} frames, {out.stat().st_size} bytes")
