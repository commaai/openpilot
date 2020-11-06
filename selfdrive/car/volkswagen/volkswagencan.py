# ----------------------------------------------------------------------- #
#                                                                         #
# CAN message packing for MQB vehicles                                    #
#                                                                         #
# ----------------------------------------------------------------------- #

def create_mqb_steering_control(packer, bus, apply_steer, idx, lkas_enabled):
  values = {
    "HCA_System_Status": 0x3,
    "LM_Offset": abs(apply_steer),
    "LM_Offset_Valid": lkas_enabled,
    "LM_Offsign": 1 if apply_steer < 0 else 0,
    "HCA_Control_Status": 0x5 if lkas_enabled else 0x3,
  }
  return packer.make_can_msg("HCA_01", bus, values, idx)

def create_mqb_hud_control(packer, bus, hca_enabled, steering_pressed, hud_alert, left_lane_visible, right_lane_visible,
                           ldw_lane_warning_left, ldw_lane_warning_right, ldw_side_dlc_tlc, ldw_dlc, ldw_tlc):
  if hca_enabled:
    left_lane_hud = 3 if left_lane_visible else 1
    right_lane_hud = 3 if right_lane_visible else 1
  else:
    left_lane_hud = 2 if left_lane_visible else 1
    right_lane_hud = 2 if right_lane_visible else 1

  values = {
    "LDW_Status_LED_gelb": 1 if hca_enabled and steering_pressed else 0,
    "LDW_Status_LED_gruen": 1 if hca_enabled and not steering_pressed else 0,
    "LDW_SW_Info_links": left_lane_hud,
    "LDW_SW_Info_rechts": right_lane_hud,
    "LDW_Texte": hud_alert,
    "LDW_SW_Warnung_links": ldw_lane_warning_left,
    "LDW_SW_Warnung_rechts": ldw_lane_warning_right,
    "LDW_Seite_DLCTLC": ldw_side_dlc_tlc,
    "LDW_DLC": ldw_dlc,
    "LDW_TLC": ldw_tlc
  }
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
    "GRA_Typ468": CS.graTyp468,
    "GRA_ButtonTypeInfo": CS.graButtonTypeInfo
  }

  return packer.make_can_msg("GRA_ACC_01", bus, values, idx)

# ----------------------------------------------------------------------- #
#                                                                         #
# CAN message packing for PQ35/PQ46/NMS vehicles                          #
#                                                                         #
# ----------------------------------------------------------------------- #

def create_pq_steering_control(packer, bus, apply_steer, idx, lkas_enabled):
  values = {
    "HCA_Zaehler": idx,
    "LM_Offset": abs(apply_steer),
    "LM_OffSign": 1 if apply_steer < 0 else 0,
    "HCA_Status": 5 if (lkas_enabled and apply_steer != 0) else 3,
    "Vib_Freq": 16,
  }

  dat = packer.make_can_msg("HCA_1", bus, values)[2]
  values["HCA_Checksumme"] = dat[1] ^ dat[2] ^ dat[3] ^ dat[4]
  return packer.make_can_msg("HCA_1", bus, values)

def create_pq_hud_control(packer, bus, hca_enabled, steering_pressed, hud_alert, leftLaneVisible, rightLaneVisible):
  if hca_enabled:
    leftlanehud = 3 if leftLaneVisible else 1
    rightlanehud = 3 if rightLaneVisible else 1
  else:
    leftlanehud = 2 if leftLaneVisible else 1
    rightlanehud = 2 if rightLaneVisible else 1

  values = {
    "Right_Lane_Status": rightlanehud,
    "Left_Lane_Status": leftlanehud,
    "SET_ME_X1": 1,
    "Kombi_Lamp_Orange": 1 if hca_enabled and steering_pressed else 0,
    "Kombi_Lamp_Green": 1 if hca_enabled and not steering_pressed else 0,
  }
  return packer.make_can_msg("LDW_1", bus, values)

pass

def create_pq_acc_buttons_control(packer, bus, buttonStatesToSend, CS, idx):
  pass
