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

def create_mqb_hud_control(packer, bus, lkas_enabled, hud_alert, leftLaneVisible, rightLaneVisible):

  if lkas_enabled:
    leftlanehud = 3 if leftLaneVisible else 1
    rightlanehud = 3 if rightLaneVisible else 1
  else:
    leftlanehud = 2 if leftLaneVisible else 1
    rightlanehud = 2 if rightLaneVisible else 1

  values = {
    "LDW_Unknown": 2, # FIXME: possible speed or attention relationship
    "Kombi_Lamp_Orange": 1 if lkas_enabled == 0 else 0,
    "Kombi_Lamp_Green": 1 if lkas_enabled == 1 else 0,
    "Left_Lane_Status": leftlanehud,
    "Right_Lane_Status": rightlanehud,
    "Alert_Message": hud_alert,
  }
  return packer.make_can_msg("LDW_02", bus, values)

def create_mqb_acc_buttons_control(packer, bus, gra_acc_buttons, gra_typ_hauptschalter, gra_buttontypeinfo, gra_tip_stufe_2, idx):
  values = {
    "GRA_Hauptschalter": gra_acc_buttons["main"],
    "GRA_Abbrechen": gra_acc_buttons["cancel"],
    "GRA_Tip_Setzen": gra_acc_buttons["set"],
    "GRA_Tip_Hoch": gra_acc_buttons["accel"],
    "GRA_Tip_Runter": gra_acc_buttons["decel"],
    "GRA_Tip_Wiederaufnahme": gra_acc_buttons["resume"],
    "GRA_Verstellung_Zeitluecke": 3 if gra_acc_buttons["timegap"] else 0,
    "GRA_Typ_Hauptschalter": gra_typ_hauptschalter,
    "GRA_Codierung": 2,
    "GRA_Tip_Stufe_2": gra_tip_stufe_2,
    "GRA_ButtonTypeInfo": gra_buttontypeinfo
  }

  return packer.make_can_msg("GRA_ACC_01", bus, values, idx)
