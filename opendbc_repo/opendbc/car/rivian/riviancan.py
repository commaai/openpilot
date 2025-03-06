def checksum(data, poly, xor_output):
  crc = 0
  for byte in data:
    crc ^= byte
    for _ in range(8):
      if crc & 0x80:
        crc = (crc << 1) ^ poly
      else:
        crc <<= 1
      crc &= 0xFF
  return crc ^ xor_output


def create_lka_steering(packer, acm_lka_hba_cmd, apply_torque, enabled):
  values = {s: acm_lka_hba_cmd[s] for s in [
    "ACM_lkaHbaCmd_Counter",
    "ACM_lkaHbaCmd_Checksum",
    "ACM_HapticRequest",
    "ACM_lkaStrToqReq",
    "ACM_lkaSymbolState",
    "ACM_lkaToiFlt",
    "ACM_lkaActToi",
    "ACM_hbaSysState",
    "ACM_FailinfoAeb",
    "ACM_lkaRHWarning",
    "ACM_lkaLHWarning",
    "ACM_lkaLaneRecogState",
    "ACM_hbaOpt",
    "ACM_hbaLamp",
    "ACM_lkaHandsoffSoundWarning",
    "ACM_lkaHandsoffDisplayWarning",
    "ACM_unkown1",
    "ACM_unkown2",
    "ACM_unkown3",
    "ACM_unkown4",
    "ACM_unkown6",
  ]}

  if enabled:
    values["ACM_lkaActToi"] = 1
    values["ACM_lkaSymbolState"] = 3
    values["ACM_lkaLaneRecogState"] = 3
    values["ACM_lkaStrToqReq"] = apply_torque
    values["ACM_unkown2"] = 1
    values["ACM_unkown3"] = 4
    values["ACM_unkown4"] = 160
    values["ACM_unkown6"] = 1

  data = packer.make_can_msg("ACM_lkaHbaCmd", 0, values)[1]
  values["ACM_lkaHbaCmd_Checksum"] = checksum(data[1:], 0x1D, 0x63)
  return packer.make_can_msg("ACM_lkaHbaCmd", 0, values)


def create_wheel_touch(packer, sccm_wheel_touch, enabled):
  values = {s: sccm_wheel_touch[s] for s in (
    "SCCM_WheelTouch_Counter",
    "SCCM_WheelTouch_HandsOn",
    "SCCM_WheelTouch_CapacitiveValue",
    "SETME_X52",
  )}

  # When only using ACC without lateral, the ACM warns the driver to hold the steering wheel on engagement
  # Tell the ACM that the user is holding the wheel to avoid this warning
  if enabled:
    values["SCCM_WheelTouch_HandsOn"] = 1
    values["SCCM_WheelTouch_CapacitiveValue"] = 100  # only need to send this value, but both are set for consistency

  data = packer.make_can_msg("SCCM_WheelTouch", 2, values)[1]
  values["SCCM_WheelTouch_Checksum"] = checksum(data[1:], 0x1D, 0x97)
  return packer.make_can_msg("SCCM_WheelTouch", 2, values)


def create_longitudinal(packer, frame, accel, enabled):
  values = {
    "ACM_longitudinalRequest_Counter": frame % 15,
    "ACM_AccelerationRequest": accel if enabled else 0,
    "ACM_VehicleHoldRequired": 0,
    "ACM_PrndRequired": 0,
    "ACM_longInterfaceEnable": 1 if enabled else 0,
    "ACM_AccelerationRequestType": 0,
  }

  data = packer.make_can_msg("ACM_longitudinalRequest", 0, values)[1]
  values["ACM_longitudinalRequest_Checksum"] = checksum(data[1:], 0x1D, 0x12)
  return packer.make_can_msg("ACM_longitudinalRequest", 0, values)
