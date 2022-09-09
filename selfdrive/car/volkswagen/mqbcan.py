def create_steering_control(packer, bus, apply_steer, lkas_enabled):
  values = {
    "SET_ME_0X3": 0x3,
    "Assist_Torque": abs(apply_steer),
    "Assist_Requested": lkas_enabled,
    "Assist_VZ": 1 if apply_steer < 0 else 0,
    "HCA_Available": 1,
    "HCA_Standby": not lkas_enabled,
    "HCA_Active": lkas_enabled,
    "SET_ME_0XFE": 0xFE,
    "SET_ME_0X07": 0x07,
  }
  return packer.make_can_msg("HCA_01", bus, values)


def create_lka_hud_control(packer, bus, ldw_stock_values, enabled, steering_pressed, hud_alert, hud_control):
  values = ldw_stock_values.copy()

  values.update({
    "LDW_Status_LED_gelb": 1 if enabled and steering_pressed else 0,
    "LDW_Status_LED_gruen": 1 if enabled and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 if hud_control.leftLaneDepart else 1 + hud_control.leftLaneVisible,
    "LDW_Lernmodus_rechts": 3 if hud_control.rightLaneDepart else 1 + hud_control.rightLaneVisible,
    "LDW_Texte": hud_alert,
  })
  return packer.make_can_msg("LDW_02", bus, values)


def create_acc_buttons_control(packer, bus, gra_stock_values, idx, cancel=False, resume=False):
  values = gra_stock_values.copy()

  values.update({
    "COUNTER": idx,
    "GRA_Abbrechen": cancel,
    "GRA_Tip_Wiederaufnahme": resume,
  })

  return packer.make_can_msg("GRA_ACC_01", bus, values)


def acc_control_status_value(main_switch_on, acc_faulted, long_active):
  if acc_faulted:
    tsk_status = 6
  elif long_active:
    tsk_status = 3
  elif main_switch_on:
    tsk_status = 2
  else:
    tsk_status = 0

  return tsk_status


def acc_hud_status_value(main_switch_on, acc_faulted, long_active):
  if acc_faulted:
    hud_status = 6
  elif long_active:
    hud_status = 3
  elif main_switch_on:
    hud_status = 2
  else:
    hud_status = 0

  return hud_status


def create_acc_accel_control(packer, bus, enabled, acc_status, accel, stopping, starting, standstill):
  commands = []

  acc_06_values = {
    "ACC_Typ": 2,  # FIXME: require SnG during refactoring, re-enable FtS later
    "ACC_Status_ACC": acc_status,
    "ACC_StartStopp_Info": enabled,
    "ACC_Sollbeschleunigung_02": accel if enabled else 3.01,
    "ACC_zul_Regelabw_unten": 0.1,  # FIXME: reintroduce comfort band support
    "ACC_zul_Regelabw_oben": 0.1,  # FIXME: reintroduce comfort band support
    "ACC_neg_Sollbeschl_Grad_02": 4.0 if enabled else 0,
    "ACC_pos_Sollbeschl_Grad_02": 4.0 if enabled else 0,
    "ACC_Anfahren": starting,
    "ACC_Anhalten": stopping,
  }
  commands.append(packer.make_can_msg("ACC_06", bus, acc_06_values))

  if starting:
    acc_hold_type = 4  # hold release / startup
  elif standstill:
    acc_hold_type = 3  # hold standby
  elif stopping:
    acc_hold_type = 1  # hold
  else:
    acc_hold_type = 0

  acc_07_values = {
    "ACC_Distance_to_Stop": 1.0 if stopping else 20.46,
    "ACC_Hold_Request": stopping,
    "ACC_Freewheel_Type": 2 if enabled else 0,
    "ACC_Hold_Type": acc_hold_type,
    "ACC_Hold_Release": starting,
    "ACC_Accel_Secondary": 3.02,  # not using this unless and until we understand its impact
    "ACC_Accel_TSK": accel if enabled else 3.01,
  }
  commands.append(packer.make_can_msg("ACC_07", bus, acc_07_values))

  return commands


def create_acc_hud_control(packer, bus, acc_hud_status, set_speed, lead_visible):
  commands = []

  acc_02_values = {
    "ACC_Status_Anzeige": acc_hud_status,
    "ACC_Wunschgeschw": set_speed if set_speed < 250 else 327.36,
    "ACC_Gesetzte_Zeitluecke": 3,
    "ACC_Display_Prio": 3,
    "ACC_Abstandsindex": 512 if lead_visible else 0,  # FIXME: will break analog clusters during refactor
  }
  commands.append(packer.make_can_msg("ACC_02", bus, acc_02_values))

  return commands

