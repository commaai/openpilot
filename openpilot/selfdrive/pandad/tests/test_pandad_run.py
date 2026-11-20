import os
import time
from unittest import mock
from unittest.mock import call

import openpilot.cereal.messaging as messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.pandad import pandad
from openpilot.selfdrive.pandad.pandad import CanSender, Pandad
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp
from openpilot.selfdrive.pandad.tests.helpers import FakePubMaster, SafetyModel, make_panda


def sendcan(frames, log_mono_time: int) -> bytes:
  msg = messaging.log_from_bytes(can_list_to_can_capnp(frames, 'sendcan')).as_builder()
  msg.logMonoTime = log_mono_time
  return msg.to_bytes()


class TestCanSender(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.panda = make_panda()
    # publisher first, or the subscriber skips what's sent before its first read
    self.pub = messaging.pub_sock('sendcan')
    self.sock = messaging.sub_sock('sendcan')
    self.log = mock.patch.object(pandad, 'cloudlog').start()
    self.addCleanup(mock.patch.stopall)

  def sent(self):
    return [c.args[0] for c in self.panda.can_send_nonblocking.call_args_list]

  @parameterized.expand([(False,), (True,)])
  def test_send(self, fake_send):
    sender, now = CanSender(self.panda, self.sock, fake_send), time.monotonic_ns()
    for msg in (sendcan([(1, b'1', 0), (5, b'5', 4)], now),  # bus 4 is another panda's
                sendcan([(2, b'2', 0)], now - 1_100_000_000),  # stale
                sendcan([(3, b'3', 0)], now + 30_000_000_000),  # from the future
                sendcan([(4, b'123456789', 0)], now),  # no DLC for 9 bytes: dropped alone
                b'garbage',
                sendcan([(6, b'6', 1), (7, b'123456789', 4)], now - 500_000_000)):  # another panda's frames aren't checked
      self.pub.send(msg)
    sender.wait(time.monotonic() - 1)
    # what's queued goes out together, even when the loop lags past its tick. FAKESEND sends nothing
    assert self.sent() == ([] if fake_send else [[(1, b'1', 0), (6, b'6', 1)]])
    assert self.log.exception.call_count == 2
    assert self.log.error.call_args_list == ([] if fake_send else [call("sendcan: dropping messages, too old to send")])

  def test_nack(self):
    # what the panda didn't take goes first on the next pass
    sender, now = CanSender(self.panda, self.sock, False), time.monotonic_ns()
    frames = [(i, b'1', 0) for i in range(300)]
    for i in range(0, 300, 100):
      self.pub.send(sendcan(frames[i:i + 100], now))
    self.panda.can_send_nonblocking.side_effect = [170, 0, 100, 2]
    for _ in range(3):
      sender.send()
    assert self.sent() == [frames, frames[170:], frames[170:]]
    # leftovers are dropped once they're 1 s old, and the next message goes out whole.
    # logged once until everything has gone out
    with mock.patch.object(pandad.time, 'monotonic_ns', return_value=now + 1_000_000_000):
      sender.send()
    assert len(self.sent()) == 3 and sender.pending == []
    for t in (now - 2_000_000_000, time.monotonic_ns(), now - 2_000_000_000):
      self.pub.send(sendcan(frames[:2], t))
      sender.send()
    assert self.sent()[-1] == frames[:2]
    assert self.log.error.call_args_list == [call("sendcan: dropping messages, too old to send")] * 2


class TestPandad(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    mock.patch.object(pandad, 'do_exit', False).start()
    self.peripherals = mock.patch.object(pandad.PeripheralControl, 'update').start()
    self.addCleanup(mock.patch.stopall)
    self.pm = messaging.PubMaster(['deviceState', 'selfdriveState'])
    self.sm = messaging.SubMaster(['selfdriveState', 'deviceState', 'cabinCameraState'])

  def publish(self, started: bool, selfdrive: tuple[bool, bool] | None):
    msg = messaging.new_message('deviceState')
    msg.deviceState.started = started
    self.pm.send('deviceState', msg)
    if selfdrive is not None:
      msg = messaging.new_message('selfdriveState', valid=selfdrive[0])
      msg.selfdriveState.enabled = selfdrive[1]
      self.pm.send('selfdriveState', msg)

  @parameterized.expand([
    ("offroad", False, None, False, {}),
    ("onroad_no_selfdrive", True, None, False, {}),
    ("onroad_invalid", True, (False, True), False, {}),
    ("onroad_disabled", True, (True, False), False, {}),
    ("onroad_engaged", True, (True, True), True, {}),
    ("spoofing_started", True, (True, False), False, {"STARTED": "1"}),
  ])
  def test_loop(self, name, started, selfdrive, engaged, env):
    panda, pm = make_panda(ignition_line=not env, power_save_enabled=bool(env)), FakePubMaster()
    with mock.patch.dict(os.environ, env):
      pd = Pandad(panda, pm, self.sm, messaging.sub_sock('sendcan'))
    # 20 ticks of the loop, then a stop signal
    def can_recv():
      self.publish(started, selfdrive)
      pandad.do_exit = panda.can_recv.call_count == 20
      return []
    panda.can_recv.side_effect = can_recv
    t = time.monotonic()
    pd.run()

    # 100 Hz can and logs, 20 Hz fan and IR, 10 Hz state + heartbeat, 2 Hz peripheralState
    assert 0.15 < time.monotonic() - t < 0.4
    assert self.peripherals.call_args_list == [call(self.sm, started, False)] * 4
    assert [s for s, _ in pm.sent].count('pandaStates') == panda.health.call_count == 2
    assert panda.send_heartbeat.call_args_list == [call(engaged)] * 2
    assert panda.debug_read.call_count == 20 and panda.get_fan_rpm.call_count == 1
    # STARTED: the ignition line is on, so power save goes off
    assert pm.last('pandaStates').pandaStates[0].ignitionLine
    ps = pm.last('peripheralState').peripheralState
    assert (ps.pandaType, ps.voltage, ps.current, ps.fanSpeedRpm) == ('tres', 12000, 500, 4321)
    assert (call.set_power_save(False) in panda.method_calls) == bool(env)

    # ELM327 to fingerprint onroad. the relay closes on exit when onroad and not engaged
    assert [c for c in panda.method_calls if c[0] == 'set_safety_mode'] == \
      [call.set_safety_mode(SafetyModel.elm327, 1)] * started + [call.set_safety_mode(SafetyModel.noOutput)] * (started and not engaged)

  def test_selfdrive_stops(self):
    # a stale engaged selfdriveState disengages the heartbeat, so the panda's mismatch check can trip
    panda = make_panda()
    pd = Pandad(panda, FakePubMaster(), self.sm, messaging.sub_sock('sendcan'))
    self.publish(True, (True, True))
    pd.step(0)
    with mock.patch.object(messaging.time, 'monotonic', return_value=time.monotonic() + 0.2):
      pd.step(10)
    assert panda.send_heartbeat.call_args_list == [call(True), call(False)]
