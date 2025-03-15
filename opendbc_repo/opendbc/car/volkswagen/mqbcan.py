def create_steering_control(packer, bus, apply_torque, lkas_enabled):
  values = {
    "HCA_01_Status_HCA": 5 if lkas_enabled else 3,
    "HCA_01_LM_Offset": abs(apply_torque),
    "HCA_01_LM_OffSign": 1 if apply_torque < 0 else 0,
    "HCA_01_Vib_Freq": 18,
    "HCA_01_Sendestatus": 1 if lkas_enabled else 0,
    "EA_ACC_Wunschgeschwindigkeit": 327.36,
  }
  return packer.make_can_msg("HCA_01", bus, values)


def create_eps_update(packer, bus, eps_stock_values, ea_simulated_torque):
  values = {s: eps_stock_values[s] for s in [
    "COUNTER",                     # Sync counter value to EPS output
    "EPS_Lenkungstyp",             # EPS rack type
    "EPS_Berechneter_LW",          # Absolute raw steering angle
    "EPS_VZ_BLW",                  # Raw steering angle sign
    "EPS_HCA_Status",              # EPS HCA control status
  ]}

  values.update({
    # Absolute driver torque input and sign, with EA inactivity mitigation
    "EPS_Lenkmoment": abs(ea_simulated_torque),
    "EPS_VZ_Lenkmoment": 1 if ea_simulated_torque < 0 else 0,
  })

  return packer.make_can_msg("LH_EPS_03", bus, values)


def create_lka_hud_control(packer, bus, ldw_stock_values, lat_active, steering_pressed, hud_alert, hud_control):
  values = {}
  if len(ldw_stock_values):
    values = {s: ldw_stock_values[s] for s in [
      "LDW_SW_Warnung_links",   # Blind spot in warning mode on left side due to lane departure
      "LDW_SW_Warnung_rechts",  # Blind spot in warning mode on right side due to lane departure
      "LDW_Seite_DLCTLC",       # Direction of most likely lane departure (left or right)
      "LDW_DLC",                # Lane departure, distance to line crossing
      "LDW_TLC",                # Lane departure, time to line crossing
    ]}

  values.update({
    "LDW_Status_LED_gelb": 1 if lat_active and steering_pressed else 0,
    "LDW_Status_LED_gruen": 1 if lat_active and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 if hud_control.leftLaneDepart else 1 + hud_control.leftLaneVisible,
    "LDW_Lernmodus_rechts": 3 if hud_control.rightLaneDepart else 1 + hud_control.rightLaneVisible,
    "LDW_Texte": hud_alert,
  })
  return packer.make_can_msg("LDW_02", bus, values)


def create_acc_buttons_control(packer, bus, gra_stock_values, cancel=False, resume=False):
  values = {s: gra_stock_values[s] for s in [
    "GRA_Hauptschalter",           # ACC button, on/off
    "GRA_Typ_Hauptschalter",       # ACC main button type
    "GRA_Codierung",               # ACC button configuration/coding
    "GRA_Tip_Stufe_2",             # unknown related to stalk type
    "GRA_ButtonTypeInfo",          # unknown related to stalk type
  ]}

  values.update({
    "COUNTER": (gra_stock_values["COUNTER"] + 1) % 16,
    "GRA_Abbrechen": cancel,
    "GRA_Tip_Wiederaufnahme": resume,
  })

  return packer.make_can_msg("GRA_ACC_01", bus, values)


def acc_control_value(main_switch_on, acc_faulted, long_active):
  if acc_faulted:
    acc_control = 6
  elif long_active:
    acc_control = 3
  elif main_switch_on:
    acc_control = 2
  else:
    acc_control = 0

  return acc_control


def acc_hud_status_value(main_switch_on, acc_faulted, long_active):
  # TODO: happens to resemble the ACC control value for now, but extend this for init/gas override later
  return acc_control_value(main_switch_on, acc_faulted, long_active)


def create_acc_accel_control(packer, bus, acc_type, acc_enabled, accel, acc_control, stopping, starting, esp_hold):
  commands = []

  acc_06_values = {
    "ACC_Typ": acc_type,
    "ACC_Status_ACC": acc_control,
    "ACC_StartStopp_Info": acc_enabled,
    "ACC_Sollbeschleunigung_02": accel if acc_enabled else 3.01,
    "ACC_zul_Regelabw_unten": 0.2,  # TODO: dynamic adjustment of comfort-band
    "ACC_zul_Regelabw_oben": 0.2,  # TODO: dynamic adjustment of comfort-band
    "ACC_neg_Sollbeschl_Grad_02": 4.0 if acc_enabled else 0,  # TODO: dynamic adjustment of jerk limits
    "ACC_pos_Sollbeschl_Grad_02": 4.0 if acc_enabled else 0,  # TODO: dynamic adjustment of jerk limits
    "ACC_Anfahren": starting,
    "ACC_Anhalten": stopping,
  }
  commands.append(packer.make_can_msg("ACC_06", bus, acc_06_values))

  if starting:
    acc_hold_type = 4  # hold release / startup
  elif esp_hold:
    acc_hold_type = 3  # hold standby
  elif stopping:
    acc_hold_type = 1  # hold request
  else:
    acc_hold_type = 0

  acc_07_values = {
    "ACC_Anhalteweg": 0.3 if stopping else 20.46,  # Distance to stop (stopping coordinator handles terminal roll-out)
    "ACC_Freilauf_Info": 2 if acc_enabled else 0,
    "ACC_Folgebeschl": 3.02,  # Not using secondary controller accel unless and until we understand its impact
    "ACC_Sollbeschleunigung_02": accel if acc_enabled else 3.01,
    "ACC_Anforderung_HMS": acc_hold_type,
    "ACC_Anfahren": starting,
    "ACC_Anhalten": stopping,
  }
  commands.append(packer.make_can_msg("ACC_07", bus, acc_07_values))

  return commands


def create_acc_hud_control(packer, bus, acc_hud_status, set_speed, lead_distance, distance):
  values = {
    "ACC_Status_Anzeige": acc_hud_status,
    "ACC_Wunschgeschw_02": set_speed if set_speed < 250 else 327.36,
    "ACC_Gesetzte_Zeitluecke": distance + 2,
    "ACC_Display_Prio": 3,
    "ACC_Abstandsindex": lead_distance,
  }

  return packer.make_can_msg("ACC_02", bus, values)


# AWV = Stopping Distance Reduction
# Refer to Self Study Program 890253: Volkswagen Driver Assistance Systems, Design and Function


def create_aeb_control(packer, fcw_active, aeb_active, accel):
  values = {
    "AWV_Vorstufe": 0,  # Preliminary stage
    "AWV1_Anf_Prefill": 0,  # Brake pre-fill request
    "AWV1_HBA_Param": 0,  # Brake pre-fill level
    "AWV2_Freigabe": 0,  # Stage 2 braking release
    "AWV2_Ruckprofil": 0,  # Brake jerk level
    "AWV2_Priowarnung": 0,  # Suppress lane departure warning in favor of FCW
    "ANB_Notfallblinken": 0, # Hazard flashers request
    "ANB_Teilbremsung_Freigabe": 0,  # Target braking release
    "ANB_Zielbremsung_Freigabe": 0,  # Partial braking release
    "ANB_Zielbrems_Teilbrems_Verz_Anf": 0.0,   # Acceleration requirement for target/partial braking, m/s/s
    "AWV_Halten": 0,  # Vehicle standstill request
    "PCF_Time_to_collision": 0xFF,  # Pre Crash Front, populated only with a target, might be used on Audi only
  }

  return packer.make_can_msg("ACC_10", 0, values)


def create_aeb_hud(packer, aeb_supported, fcw_active):
  values = {
    "AWV_Texte": 5 if aeb_supported else 7,  # FCW/AEB system status, display text (from menu in VAL)
    "AWV_Status_Anzeige": 1 if aeb_supported else 2,  #  FCW/AEB system status, available or disabled
  }

  return packer.make_can_msg("ACC_15", 0, values)
