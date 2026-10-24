from opendbc.car.crc import CRC8J1850


def create_lka_steering(packer, frame, acm_lka_hba_cmd, apply_torque, enabled, active):
  # forward auto high beam and speed limit status and nothing else
  values = {s: acm_lka_hba_cmd[s] for s in (
    "ACM_hbaSysState",
    "ACM_hbaLamp",
    "ACM_hbaOnOffState",
    "ACM_slifOnOffState",
  )}

  values |= {
    "ACM_lkaHbaCmd_Counter": frame % 15,  # 0-14 counter (15 states), so % 15 is correct despite the 4-bit field
    "ACM_lkaStrToqReq": apply_torque,
    "ACM_lkaActToi": active,

    "ACM_lkaLaneRecogState": 3 if enabled else 0,
    "ACM_lkaSymbolState": 3 if enabled else 0,

    # static values
    "ACM_lkaElkRequest": 0,
    "ACM_ldwlkaOnOffState": 2,  # 2=LKAS+LDW on
    "ACM_elkOnOffState": 1,  # 1=LKAS on
    # TODO: what are these used for?
    "ACM_ldwWarnTypeState": 2,  # always 2
    "ACM_ldwWarnTimingState": 1,  # always 1
    #"ACM_lkaHandsoffDisplayWarning": 1,  # TODO: we can send this when openpilot wants you to pay attention
  }

  return packer.make_can_msg("ACM_lkaHbaCmd", 0, values)


def create_wheel_touch(packer, sccm_wheel_touch, enabled):
  values = {s: sccm_wheel_touch[s] for s in (
    "SCCM_WheelTouch_Counter",
    "SCCM_WheelTouch_HandsOn",
    "SCCM_WheelTouch_Calibration",
    "SCCM_WheelTouch_CapacitiveValue",
    "SCCM_WheelTouch_ResistiveValue",
  )}

  # When only using ACC without lateral, the ACM warns the driver to hold the steering wheel on engagement
  # Tell the ACM that the user is holding the wheel to avoid this warning
  if enabled:
    values["SCCM_WheelTouch_HandsOn"] = 1
    values["SCCM_WheelTouch_CapacitiveValue"] = 100  # only need to send this value, but both are set for consistency

  return packer.make_can_msg("SCCM_WheelTouch", 2, values)


def create_longitudinal(packer, frame, accel, enabled):
  values = {
    "ACM_longitudinalRequest_Counter": frame % 15,  # 0-14 counter (15 states), so % 15 is correct despite the 4-bit field
    "ACM_AccelerationRequest": accel,
    "ACM_PrndRequest": 0,
    "ACM_longInterfaceEnable": 1 if enabled else 0,
    "ACM_VehicleHoldRequest": 0,
  }

  return packer.make_can_msg("ACM_longitudinalRequest", 0, values)


def create_adas_status(packer, vdm_adas_status, interface_status):
  values = {s: vdm_adas_status[s] for s in (
    "VDM_AdasStatus_Counter",
    "VDM_AdasDecelLimit",
    "VDM_AdasDriverAccelPriorityStatus",
    "VDM_AdasFaultStatus",
    "VDM_AdasAccelLimit",
    "VDM_AdasDriverModeStatus",
    "VDM_AdasUnkown1",
    "VDM_AdasInterfaceStatus",
    "VDM_AdasVehicleHoldStatus",
    "VDM_UserAdasRequest",
  )}

  if interface_status is not None:
    values["VDM_AdasInterfaceStatus"] = interface_status

  return packer.make_can_msg("VDM_AdasSts", 2, values)


def rivian_checksum(address: int, sig, d: bytearray) -> int:
  # CRC-8 with the little-endian CAN address prepended to the payload.
  crc = 0
  for byte in address.to_bytes(2, "little") + d[1:]:
    crc = CRC8J1850[crc ^ byte]
  return crc
