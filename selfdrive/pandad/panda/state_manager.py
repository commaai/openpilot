import os
from cereal import car, log
import cereal.messaging as messaging
from openpilot.common.params import Params

SPOOFING_STARTED = os.getenv("STARTED") == "1"

LEC_ERROR_CODE = {
  "No error": 0, "Stuff error": 1, "Form error": 2, "AckError": 3,
  "Bit1Error": 4, "Bit0Error": 5, "CRCError": 6, "NoChange": 7,
}
class PandaStateManager:
  def __init__(self, pandas, lock):
    self.pandas = pandas
    self.params = Params()
    self.lock = lock

    with self.lock:
      self.hw_types = [int.from_bytes(p.get_type(), 'big') for p in pandas]

  def _fill_state(self, ps, hw_type, health):
    ps.voltage = health.get('voltage', 0)
    ps.current = health.get('current', 0)
    ps.uptime = health.get('uptime', 0)
    ps.safetyTxBlocked = health.get('safety_tx_blocked', 0)
    ps.safetyRxInvalid = health.get('safety_rx_invalid', 0)
    ps.ignitionLine = bool(health.get('ignition_line', 0))
    ps.ignitionCan = bool(health.get('ignition_can', 0))
    ps.controlsAllowed = bool(health.get('controls_allowed', 0))
    ps.txBufferOverflow = health.get('tx_buffer_overflow', 0)
    ps.rxBufferOverflow = health.get('rx_buffer_overflow', 0)
    ps.pandaType = hw_type
    ps.safetyModel = health.get('safety_mode', car.CarParams.SafetyModel.noOutput)
    ps.safetyParam = health.get('safety_param', 0)
    ps.faultStatus = health.get('fault_status', 0)
    ps.powerSaveEnabled = bool(health.get('power_save_enabled', 0))
    ps.heartbeatLost = bool(health.get('heartbeat_lost', 0))
    ps.alternativeExperience = health.get('alternative_experience', 0)
    ps.harnessStatus = health.get('car_harness_status', 0)
    ps.interruptLoad = health.get('interrupt_load', 0)
    ps.fanPower = health.get('fan_power', 0)
    ps.fanStallCount = health.get('fan_stall_count', 0)
    ps.safetyRxChecksInvalid = bool(health.get('safety_rx_checks_invalid', 0))
    ps.spiChecksumErrorCount = health.get('spi_checksum_error_count', 0)
    ps.sbu1Voltage = health.get('sbu1_voltage_mV', 0) / 1000.0
    ps.sbu2Voltage = health.get('sbu2_voltage_mV', 0) / 1000.0

  def _fill_can_state(self, cs, can_health):
    cs.busOff = bool(can_health.get('bus_off', 0))
    cs.busOffCnt = can_health.get('bus_off_cnt', 0)
    cs.errorWarning = bool(can_health.get('error_warning', 0))
    cs.errorPassive = bool(can_health.get('error_passive', 0))
    cs.lastError = LEC_ERROR_CODE.get(can_health.get('last_error', "No error"), 0)
    cs.lastStoredError = LEC_ERROR_CODE.get(can_health.get('last_stored_error', "No error"), 0)
    cs.lastDataError = LEC_ERROR_CODE.get(can_health.get('last_data_error', "No error"), 0)
    cs.lastDataStoredError = LEC_ERROR_CODE.get(can_health.get('last_data_stored_error', "No error"), 0)
    cs.receiveErrorCnt = can_health.get('receive_error_cnt', 0)
    cs.transmitErrorCnt = can_health.get('transmit_error_cnt', 0)
    cs.totalErrorCnt = can_health.get('total_error_cnt', 0)
    cs.totalTxLostCnt = can_health.get('total_tx_lost_cnt', 0)
    cs.totalRxLostCnt = can_health.get('total_rx_lost_cnt', 0)
    cs.totalTxCnt = can_health.get('total_tx_cnt', 0)
    cs.totalRxCnt = can_health.get('total_rx_cnt', 0)
    cs.totalFwdCnt = can_health.get('total_fwd_cnt', 0)
    cs.canSpeed = can_health.get('can_speed', 0)
    cs.canDataSpeed = can_health.get('can_data_speed', 0)
    cs.canfdEnabled = bool(can_health.get('canfd_enabled', 0))
    cs.brsEnabled = bool(can_health.get('brs_enabled', 0))
    cs.canfdNonIso = bool(can_health.get('canfd_non_iso', 0))
    cs.irq0CallRate = can_health.get('irq0_call_rate', 0)
    cs.irq1CallRate = can_health.get('irq1_call_rate', 0)
    cs.irq2CallRate = can_health.get('irq2_call_rate', 0)
    cs.canCoreResetCnt = can_health.get('can_core_reset_count', 0)

  def process(self, engaged, pm)-> bool:
    ignition = False
    is_comma_three_red = (
      len(self.pandas) == 2 and
      self.hw_types[0] == log.PandaState.PandaType.dos and
      self.hw_types[1] == log.PandaState.PandaType.redPanda
    )

    msg = messaging.new_message('pandaStates', len(self.pandas))
    pss = msg.pandaStates
    for i, p in enumerate(self.pandas):
      with self.lock:
        health = p.health() or {}

      if SPOOFING_STARTED:
        health['ignition_line'] = 1
      elif is_comma_three_red and self.hw_types[i] == log.PandaState.PandaType.dos:
        health['ignition_line'] = 0

      ignition |= bool(health.get('ignition_line', 0) or health.get('ignition_can', 0))
      ps = pss[i]
      self._fill_state(ps, self.hw_types[i], health)

      for j, cs in enumerate((ps.init('canState0'), ps.init('canState1'), ps.init('canState2'))):
        with self.lock:
          can_helth = p.can_health(j)
        self._fill_can_state(cs, can_helth)

      fault_bits = health.get('faults', 0)
      fault_count = bin(fault_bits).count('1')
      faults = ps.init('faults', fault_count)
      idx = 0
      for f in range(log.PandaState.FaultType.relayMalfunction,
                     log.PandaState.FaultType.heartbeatLoopWatchdog + 1):
        if fault_bits & (1 << f):
          faults[idx] = f
          idx += 1

    with self.lock:
      for i, p in enumerate(self.pandas):
        ps = pss[i]
        if ps.safetyModel == car.CarParams.SafetyModel.silent or (
            not ignition and ps.safetyModel != car.CarParams.SafetyModel.noOutput):
            p.set_safety_mode(car.CarParams.SafetyModel.noOutput)

        if ps.powerSaveEnabled != (not ignition):
          p.set_power_save(not ignition)
        p.send_heartbeat(engaged)

    pm.send("pandaStates", msg)
    return ignition
