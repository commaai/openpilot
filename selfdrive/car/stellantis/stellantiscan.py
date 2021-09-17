def create_lkas_command(packer, apply_steer, counter, steer_command_bit):
  values = {
    "LKAS_COMMAND": apply_steer,
    "LKAS_CONTROL_BIT": steer_command_bit,
    "COUNTER": counter
  }

  return packer.make_can_msg("DASM_LKAS_CMD", 0, values)

def create_cswc(packer, stock_cswc_values, cancel):
  if cancel:
    values = {
      "COUNTER": stock_cswc_values["CANCEL"],
      "CANCEL": 1,
    }
  else:
    values = stock_cswc_values.copy()

  return packer.make_can_msg("CSWC", 2, values)


def create_lkas_hud(packer, enabled, left_lane_visible, right_lane_visible, stock_lkas_hud_values):
  values = stock_lkas_hud_values.copy()  # Default to pass through auto high beam control, etc
  values.update({
    "LKAS_HUD_1": 0,  # TODO: go back and reanalyze the LKAS HUD lane/warning message logic
    "LKAS_HUD_2": 0,  # TODO: ditto
  })

  return packer.make_can_msg("DASM_LKAS_HUD", 0, values)
