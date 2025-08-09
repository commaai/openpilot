import os
from cereal import car, log
import cereal.messaging as messaging
from panda.python import PANDA_BUS_CNT

SPOOFING_STARTED = os.getenv("STARTED") == "1"

LEC_ERROR_CODE = {
  "No error": 0, "Stuff error": 1, "Form error": 2, "AckError": 3,
  "Bit1Error": 4, "Bit0Error": 5, "CRCError": 6, "NoChange": 7,
}

class PandaStateManager:
  def __init__(self, pandas, hw_types):
    self.pandas = pandas
    self.hw_types = hw_types

    self.is_comma_three_red = (
      len(pandas) == 2 and
      self.hw_types[0] == log.PandaState.PandaType.dos and
      self.hw_types[1] == log.PandaState.PandaType.redPanda
    )

  def process(self, engaged, pm) -> bool:
    msg = messaging.new_message('pandaStates', len(self.pandas))
    msg.valid = True
    panda_states = msg.pandaStates
    ignition = False

    for i, panda in enumerate(self.pandas):
      health = panda.health() or {}

      if SPOOFING_STARTED:
        health['ignition_line'] = 1

      # on comma three setups with a red panda, the dos can
      # get false positive ignitions due to the harness box
      # without a harness connector, so ignore it
      if self.is_comma_three_red and i == 0:
        health['ignition_line'] = 0

      ignition |= bool(health['ignition_line']) or bool(health['ignition_can'])
      ps = panda_states[i]
      self._fill_state(ps, self.hw_types[i], health)

      # Fill can state
      for j in range(PANDA_BUS_CNT):
        can_health = panda.can_health(j)
        can_state = ps.init(f'canState{j}')
        self._fill_can_state(can_state, can_health)

      # Set faults
      fault_bits = int(health.get('faults', 0))
      faults_list = [f for f in range(log.PandaState.FaultType.relayMalfunction,
                                      log.PandaState.FaultType.heartbeatLoopWatchdog + 1)
                    if fault_bits & (1 << f)]
      faults = ps.init('faults', len(faults_list))
      for idx, fault in enumerate(faults_list):
        faults[idx] = fault

    for panda, ps in zip(self.pandas, panda_states, strict=False):
      if ps.safetyModel == car.CarParams.SafetyModel.silent or (
          not ignition and ps.safetyModel != car.CarParams.SafetyModel.noOutput):
        panda.set_safety_mode(car.CarParams.SafetyModel.noOutput)

      if ps.powerSaveEnabled != (not ignition):
        panda.set_power_save(not ignition)

      panda.send_heartbeat(engaged)

    pm.send("pandaStates", msg)
    return ignition

  def _fill_state(self, ps, hw_type, health):
    ps.voltage = health['voltage']
    ps.current = health['current']
    ps.uptime = health['uptime']
    ps.safetyTxBlocked = health['safety_tx_blocked']
    ps.safetyRxInvalid = health['safety_rx_invalid']
    ps.ignitionLine = bool(health['ignition_line'])
    ps.ignitionCan = bool(health['ignition_can'])
    ps.controlsAllowed = bool(health['controls_allowed'])
    ps.txBufferOverflow = health['tx_buffer_overflow']
    ps.rxBufferOverflow = health['rx_buffer_overflow']
    ps.pandaType = hw_type
    ps.safetyModel = health['safety_mode']
    ps.safetyParam = health['safety_param']
    ps.faultStatus = health['fault_status']
    ps.powerSaveEnabled = bool(health['power_save_enabled'])
    ps.heartbeatLost = bool(health['heartbeat_lost'])
    ps.alternativeExperience = health['alternative_experience']
    ps.harnessStatus = health['car_harness_status']
    ps.interruptLoad = health['interrupt_load']
    ps.fanPower = health['fan_power']
    ps.fanStallCount = health['fan_stall_count']
    ps.safetyRxChecksInvalid = bool(health['safety_rx_checks_invalid'])
    ps.spiErrorCount = health['spi_error_count']
    ps.sbu1Voltage = health['sbu1_voltage_mV'] / 1000.0
    ps.sbu2Voltage = health['sbu2_voltage_mV'] / 1000.0

  def _fill_can_state(self, cs, can_health):
    cs.busOff = bool(can_health['bus_off'])
    cs.busOffCnt = can_health['bus_off_cnt']
    cs.errorWarning = bool(can_health['error_warning'])
    cs.errorPassive = bool(can_health['error_passive'])
    cs.lastError = LEC_ERROR_CODE[can_health['last_error']]
    cs.lastStoredError = LEC_ERROR_CODE[can_health['last_stored_error']]
    cs.lastDataError = LEC_ERROR_CODE[can_health['last_data_error']]
    cs.lastDataStoredError = LEC_ERROR_CODE[can_health['last_data_stored_error']]
    cs.receiveErrorCnt = can_health['receive_error_cnt']
    cs.transmitErrorCnt = can_health['transmit_error_cnt']
    cs.totalErrorCnt = can_health['total_error_cnt']
    cs.totalTxLostCnt = can_health['total_tx_lost_cnt']
    cs.totalRxLostCnt = can_health['total_rx_lost_cnt']
    cs.totalTxCnt = can_health['total_tx_cnt']
    cs.totalRxCnt = can_health['total_rx_cnt']
    cs.totalFwdCnt = can_health['total_fwd_cnt']
    cs.canSpeed = can_health['can_speed']
    cs.canDataSpeed = can_health['can_data_speed']
    cs.canfdEnabled = bool(can_health['canfd_enabled'])
    cs.brsEnabled = bool(can_health['brs_enabled'])
    cs.canfdNonIso = bool(can_health['canfd_non_iso'])
    cs.irq0CallRate = can_health['irq0_call_rate']
    cs.irq1CallRate = can_health['irq1_call_rate']
    cs.irq2CallRate = can_health['irq2_call_rate']
    cs.canCoreResetCnt = can_health['can_core_reset_count']
