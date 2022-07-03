def create_pq_steering_control(packer, bus, apply_steer, idx, lkas_enabled):
  values = {
    "LM_Offset": abs(apply_steer),
    "LM_OffSign": 1 if apply_steer < 0 else 0,
    "HCA_Status": 5 if (lkas_enabled and apply_steer != 0) else 3,
    "Vib_Freq": 16,
  }

  return packer.make_can_msg("HCA_1", bus, values, idx)

def create_pq_hud_control(packer, bus, enabled, steering_pressed, hud_alert, left_lane_visible, right_lane_visible,
                          ldw_stock_values, left_lane_depart, right_lane_depart):
  values = ldw_stock_values.copy()
  values.update({
    "LDW_Lampe_gelb": 1 if enabled and steering_pressed else 0,
    "LDW_Lampe_gruen": 1 if enabled and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 if left_lane_depart else 1 + left_lane_visible,
    "LDW_Lernmodus_rechts": 3 if right_lane_depart else 1 + right_lane_visible,
    "LDW_Textbits": hud_alert,
  })
  return packer.make_can_msg("LDW_Status", bus, values)

def create_pq_acc_buttons_control(packer, bus, buttonStatesToSend, CS, idx):
  values = {
    "GRA_Sender": CS.graSenderCoding,
    "GRA_Abbrechen": 1 if (buttonStatesToSend["cancel"] or CS.buttonStates["cancel"]) else 0,
    "GRA_Hauptschalt": CS.graHauptschalter,
  }

  return packer.make_can_msg("GRA_Neu", bus, values, idx)
