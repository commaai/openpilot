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


def create_acc_02_control(packer, bus, acc_status, set_speed, lead_distance):
  values = {
    "ACC_Status_Anzeige": 3 if acc_status == 5 else acc_status,
    "ACC_Wunschgeschw": set_speed if set_speed < 250 else 327.36,
    "ACC_Gesetzte_Zeitluecke": 3,
    "ACC_Display_Prio": 3,
    "ACC_Abstandsindex": lead_distance,
  }

  return packer.make_can_msg("ACC_02", bus, values)


def create_acc_04_control(packer, bus, acc_04_stock_values):
  values = acc_04_stock_values.copy()

  # Suppress disengagement alert from stock radar when OP long is in use, but passthru FCW/AEB alerts
  if values["ACC_Texte_braking_guard"] == 4:
    values["ACC_Texte_braking_guard"] = 0

  return packer.make_can_msg("ACC_04", bus, values)


def create_acc_06_control(packer, bus, enabled, acc_status, accel, acc_stopping, acc_starting, cb_pos, cb_neg, acc_type):
  values = {
    "ACC_Typ": acc_type,
    "ACC_Status_ACC": acc_status,
    "ACC_StartStopp_Info": enabled,
    "ACC_Sollbeschleunigung_02": accel if enabled else 3.01,
    "ACC_zul_Regelabw_unten": cb_neg,
    "ACC_zul_Regelabw_oben": cb_pos,
    "ACC_neg_Sollbeschl_Grad_02": 4.0 if enabled else 0,
    "ACC_pos_Sollbeschl_Grad_02": 4.0 if enabled else 0,
    "ACC_Anfahren": acc_starting,
    "ACC_Anhalten": acc_stopping,
  }

  return packer.make_can_msg("ACC_06", bus, values)


def create_acc_07_control(packer, bus, enabled, accel, acc_hold_request, acc_hold_release, acc_hold_type, stopping_distance):
  values = {
    "ACC_Distance_to_Stop": stopping_distance,
    "ACC_Hold_Request": acc_hold_request,
    "ACC_Freewheel_Type": 2 if enabled else 0,
    "ACC_Hold_Type": acc_hold_type,
    "ACC_Hold_Release": acc_hold_release,
    "ACC_Accel_Secondary": 3.02,  # not using this unless and until we understand its impact
    "ACC_Accel_TSK": accel if enabled else 3.01,
  }

  return packer.make_can_msg("ACC_07", bus, values)
