# the car for test_faults.py, a Toyota route's CAN for the UP's can_replay(), its controls and faults as overrides
from opendbc.can.packer import CANPacker
from opendbc.car.can_definitions import CanData
from opendbc.car.toyota.interface import CarInterface
from opendbc.car.toyota.radar_interface import RadarInterface
from opendbc.car.toyota.toyotacan import toyota_checksum
from opendbc.car.toyota.values import CAR
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.tools.lib.logreader import LogReader

ROUTE = "5f5afb36036506e4/2019-05-14--02-09-54/0:2"
CAR_FINGERPRINT = CAR.TOYOTA_COROLLA_TSS2
# the route is parked for its first 34 s and stops twice, loop where it keeps moving (5-11 m/s)
WINDOW = (60., 102.)
DT = 0.01  # can_replay()'s frame period
# master's toyota parser wants these on the cam bus, the 2019 route only has them as tx echoes
CAM_ECHOES = {0x343, 0x412}

PCM_CRUISE, PCM_CRUISE_2, BRAKE_MODULE, STEER_TORQUE_SENSOR, EPS_STATUS = 0x1D2, 0x1D3, 0x226, 0x260, 0x262
# faults that silence whole buses, by the buses
OUTAGES = {'can_all_silent': (0, 1, 2), 'can_bus0_silent': (0,)}
FAULTS = ('pcm_cancel', 'acc_fault', 'eps_temp_fault', 'eps_perm_fault', 'can_bad_checksum_eps', 'brake_press', *OUTAGES)
# a short wheel touch every few seconds keeps driver monitoring happy, the route's own torque would be steerOverride
TOUCH_TORQUE, TOUCH_PERIOD, TOUCH_TIME = 150, 5.0, 0.3
# the UP's CAN lanes share an SPI link that can't keep up with 100 Hz, the parsers learn the slower rate
MAX_RATE_FRAMES = 2


def needed_messages(can_msgs: list) -> set[tuple[int, int]]:
  # the parsers only add the messages they read once they run, everything else is dropped
  CI = CarInterface(CarInterface.get_non_essential_params(CAR_FINGERPRINT))
  for i, msgs in enumerate(can_msgs[:600]):
    CI.update([(i * 10_000_000, [CanData(a, d, b) for a, d, b in msgs])])
  return {(p.bus, addr) for p in CI.can_parsers.values() if p is not None for addr in p.message_states}


def synth_radar(n: int) -> list[list]:
  # an empty TSS2 radar, the route's trigger is ~9 Hz and radarTracks is checked at 20 Hz. packing counts the COUNTERs
  packer = CANPacker('toyota_tss2_adas')
  ri = RadarInterface(CarInterface.get_non_essential_params(CAR_FINGERPRINT))
  tracks = [a for a in ri.RADAR_A_MSGS + ri.RADAR_B_MSGS if a != ri.trigger_msg]
  frames = []
  for i in range(n):
    msgs = [packer.make_can_msg(a, 1, {}) for k, a in enumerate(tracks) if i % 30 == k % 30]
    if i % 5 == 0:
      msgs.append(packer.make_can_msg(ri.trigger_msg, 1, {}))
    if i % 10 == 0:
      msgs.append(packer.make_can_msg(0x123, 1, {'RADAR_STATUS': 1}))
    frames.append(msgs)
  return frames


def load_frames() -> list[list]:
  lr = [m for m in LogReader(ROUTE) if m.which() == 'can']
  can_msgs = [m[1] for m in can_capnp_to_list([m.as_builder().to_bytes() for m in lr])]
  needed = needed_messages(can_msgs)

  frames: list[list] = [[] for _ in range(round((WINDOW[1] - WINDOW[0]) / DT))]
  for m, msgs in zip(lr, can_msgs, strict=True):
    i = int(((m.logMonoTime - lr[0].logMonoTime) / 1e9 - WINDOW[0]) // DT)
    if 0 <= i < len(frames):
      frames[i] += [(a, d, b) for a, d, b in msgs if (b, a) in needed] + [(a, d, 2) for a, d, b in msgs if b == 128 and a in CAM_ECHOES]
  frames = [f + radar for f, radar in zip(frames, synth_radar(len(frames)), strict=True)]
  return decimate(frames)


def decimate(frames: list[list]) -> list[list]:
  # each message at most every MAX_RATE_FRAMES frames
  last: dict[tuple[int, int], int] = {}
  out = []
  for i, msgs in enumerate(frames):
    kept = []
    for addr, dat, bus in msgs:
      if i - last.get((addr, bus), -MAX_RATE_FRAMES) >= MAX_RATE_FRAMES:
        kept.append((addr, dat, bus))
        last[(addr, bus)] = i
    out.append(kept)
  return out


def torque(dat: bytes, driver: int, bad_checksum: bool = False) -> bytes:
  d = bytearray(dat)
  d[1:3] = driver.to_bytes(2, 'big', signed=True)
  d[-1] = toyota_checksum(STEER_TORQUE_SENSOR, None, d) ^ (0xFF if bad_checksum else 0)
  return bytes(d)


def brake(dat: bytes, pressed: bool) -> bytes:
  d = bytearray(dat)
  d[4] = d[4] | 0x20 if pressed else d[4] & ~0x20
  return bytes(d)


class Car:
  def __init__(self, up):
    self.up = up
    self.state = {"cruise": False, "touch": True, "fault": ""}
    packer = CANPacker('toyota_nodsu_pt_generated')
    def pack(name, values):
      return packer.make_can_msg(name, 0, values)[1]
    self.pcm = {on: pack('PCM_CRUISE', {'CRUISE_ACTIVE': on, 'GAS_RELEASED': 1, 'CRUISE_STATE': 8 if on else 0}) for on in (False, True)}
    self.pcm_2 = {faulted: pack('PCM_CRUISE_2', {'MAIN_ON': 1, 'SET_SPEED': 60, 'ACC_FAULTED': faulted}) for faulted in (False, True)}
    # see opendbc toyota carstate TEMP/PERM_STEER_FAULTS
    self.eps_faults = {
      'eps_temp_fault': pack('EPS_STATUS', {'LKA_STATE': 9}),
      'eps_perm_fault': pack('EPS_STATUS', {'LKA_STATE': 17}),
    }

    frames = load_frames()
    self.messages = {(a, b): d for msgs in frames for a, d, b in msgs}
    # cruise off, the touch pulses, and none of the route's brake taps, which would disengage
    for i, msgs in enumerate(frames):
      touch = TOUCH_TORQUE if i * DT % TOUCH_PERIOD < TOUCH_TIME else 0
      for j, (addr, dat, bus) in enumerate(msgs):
        if bus != 0:
          continue
        if addr == PCM_CRUISE:
          dat = self.pcm[False]
        elif addr == PCM_CRUISE_2:
          dat = self.pcm_2[False]
        elif addr == STEER_TORQUE_SENSOR:
          dat = torque(dat, touch)
        elif addr == BRAKE_MODULE:
          dat = brake(dat, False)
        msgs[j] = (addr, dat, bus)
    # the harness in, ignition off
    up.reset()
    up.can_replay(frames)

  def set(self, **state) -> None:
    self.state |= state
    fault, o = self.state["fault"], {}
    if self.state["cruise"] and fault != 'pcm_cancel':
      o[(PCM_CRUISE, 0)] = self.pcm[True]
    if fault == 'acc_fault':
      o[(PCM_CRUISE_2, 0)] = self.pcm_2[True]
    if fault in self.eps_faults:
      o[(EPS_STATUS, 0)] = self.eps_faults[fault]
    if not self.state["touch"] or fault == 'can_bad_checksum_eps':
      o[(STEER_TORQUE_SENSOR, 0)] = torque(self.messages[(STEER_TORQUE_SENSOR, 0)], 0, fault == 'can_bad_checksum_eps')
    if fault == 'brake_press':
      o[(BRAKE_MODULE, 0)] = brake(self.messages[(BRAKE_MODULE, 0)], True)
    silent = OUTAGES.get(fault, ())
    o |= {(a, b): None for a, b in self.messages if b in silent}
    self.up.can_replay_set(o)
