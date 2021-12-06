# CAN controls for MQB platform Volkswagen, Audi, Skoda and SEAT.
# PQ35/PQ46/NMS, and any future MLB, to come later.

def create_mqb_steering_control(packer, bus, apply_steer, idx, lkas_enabled):
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
  return packer.make_can_msg("HCA_01", bus, values, idx)

def create_mqb_hud_control(packer, bus, enabled, steering_pressed, hud_alert, left_lane_visible, right_lane_visible,
                           ldw_stock_values, left_lane_depart, right_lane_depart):
  # Lane color reference:
  # 0 (LKAS disabled) - off
  # 1 (LKAS enabled, no lane detected) - dark gray
  # 2 (LKAS enabled, lane detected) - light gray on VW, green or white on Audi depending on year or virtual cockpit.  On a color MFD on a 2015 A3 TDI it is white, virtual cockpit on a 2018 A3 e-Tron its green.
  # 3 (LKAS enabled, lane departure detected) - white on VW, red on Audi
  values = ldw_stock_values.copy()
  values.update({
    "LDW_Status_LED_gelb": 1 if enabled and steering_pressed else 0,
    "LDW_Status_LED_gruen": 1 if enabled and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 if left_lane_depart else 1 + left_lane_visible,
    "LDW_Lernmodus_rechts": 3 if right_lane_depart else 1 + right_lane_visible,
    "LDW_Texte": hud_alert,
  })
  return packer.make_can_msg("LDW_02", bus, values)

def create_mqb_acc_buttons_control(packer, bus, buttonStatesToSend, CS, idx):
  values = {
    "GRA_Hauptschalter": CS.graHauptschalter,
    "GRA_Abbrechen": buttonStatesToSend["cancel"],
    "GRA_Tip_Setzen": buttonStatesToSend["setCruise"],
    "GRA_Tip_Hoch": buttonStatesToSend["accelCruise"],
    "GRA_Tip_Runter": buttonStatesToSend["decelCruise"],
    "GRA_Tip_Wiederaufnahme": buttonStatesToSend["resumeCruise"],
    "GRA_Verstellung_Zeitluecke": 3 if buttonStatesToSend["gapAdjustCruise"] else 0,
    "GRA_Typ_Hauptschalter": CS.graTypHauptschalter,
    "GRA_Codierung": 2,
    "GRA_Tip_Stufe_2": CS.graTipStufe2,
    "GRA_ButtonTypeInfo": CS.graButtonTypeInfo
  }
  return packer.make_can_msg("GRA_ACC_01", bus, values, idx)

def create_mqb_acc_02_control(packer, bus, acc_status, set_speed, speed_visible, lead_visible, idx):
  values = {
    "ACC_Status_Anzeige": acc_status,
    "ACC_Wunschgeschw": 327.36 if not speed_visible else set_speed,
    "ACC_Gesetzte_Zeitluecke": 3,
    "ACC_Display_Prio": 3,
    "ACC_Abstandsindex": 637 if lead_visible else 0,
  }

  return packer.make_can_msg("ACC_02", bus, values, idx)

def create_mqb_acc_04_control(packer, bus, acc_04_stock_values, idx):
  values = acc_04_stock_values.copy()

  # Suppress disengagement alert from stock radar when OP long is in use, but passthru FCW/AEB alerts
  if values["ACC_Texte_braking_guard"] == 4:
    values["ACC_Texte_braking_guard"] = 0

  return packer.make_can_msg("ACC_04", bus, values, idx)

def create_mqb_acc_06_control(packer, bus, enabled, acc_status, accel, acc_stopping, acc_starting,
                              cb_pos, cb_neg, idx):
  values = {
    "ACC_Typ": 2,  # FIXME: locked to stop and go, need to tweak for cars that only support follow-to-stop
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

  return packer.make_can_msg("ACC_06", bus, values, idx)

def create_mqb_acc_07_control(packer, bus, enabled, accel, acc_hold_request, acc_hold_release,
                              acc_hold_type, stopping_distance, idx):
  values = {
    "ACC_Distance_to_Stop": stopping_distance,
    "ACC_Hold_Request": acc_hold_request,
    "ACC_Freewheel_Type": 2 if enabled else 0,
    "ACC_Hold_Type": acc_hold_type,
    "ACC_Hold_Release": acc_hold_release,
    "ACC_Accel_Secondary": 3.02,  # not using this unless and until we understand its impact
    "ACC_Accel_TSK": accel if enabled else 3.01,
  }

  return packer.make_can_msg("ACC_07", bus, values, idx)

def create_mqb_acc_13_control(packer, bus, enabled, acc_13_stock_values):
  values = acc_13_stock_values.copy()

  values.update({
    "Unknown_Status": 15,
    "ACC_Engaged": enabled,
  })

  return packer.make_can_msg("ACC_13", bus, values)
