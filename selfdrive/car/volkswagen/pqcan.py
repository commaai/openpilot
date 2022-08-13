def create_steering_control(packer, bus, apply_steer, lkas_enabled):
  values = {
    "LM_Offset": abs(apply_steer),
    "LM_OffSign": 1 if apply_steer < 0 else 0,
    "HCA_Status": 5 if (lkas_enabled and apply_steer != 0) else 3,
    "Vib_Freq": 16,
  }

  return packer.make_can_msg("HCA_1", bus, values)


def create_lka_hud_control(packer, bus, ldw_stock_values, enabled, steering_pressed, hud_alert, hud_control):
  values = ldw_stock_values.copy()

  values.update({
    "LDW_Lampe_gelb": 1 if enabled and steering_pressed else 0,
    "LDW_Lampe_gruen": 1 if enabled and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 if hud_control.leftLaneDepart else 1 + hud_control.leftLaneVisible,
    "LDW_Lernmodus_rechts": 3 if hud_control.rightLaneDepart else 1 + hud_control.rightLaneVisible,
    "LDW_Textbits": hud_alert,
  })

  return packer.make_can_msg("LDW_Status", bus, values)


def create_acc_buttons_control(packer, bus, gra_stock_values, idx, cancel=False, resume=False):
  values = gra_stock_values.copy()

  values.update({
    "COUNTER": idx,
    "GRA_Abbrechen": cancel,
    "GRA_Recall": resume,
  })

  return packer.make_can_msg("GRA_Neu", bus, values)


def tsk_status_value(main_switch_on, acc_faulted, long_active):
  if long_active:
    tsk_status = 1
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


def create_acc_accel_control(packer, bus, adr_status, accel):
  values = {
    "ACS_Sta_ADR": adr_status,
    "ACS_StSt_Info": adr_status != 1,
    "ACS_Typ_ACC": 0,  # TODO: this is ACC "basic", find a way to detect FtS support (1)
    "ACS_Sollbeschl": accel if adr_status == 1 else 3.01,
    "ACS_zul_Regelabw": 0.2 if adr_status == 1 else 1.27,
    "ACS_max_AendGrad": 3.0 if adr_status == 1 else 5.08,
  }

  return packer.make_can_msg("ACC_System", bus, values)


def create_acc_hud_control(packer, bus, acc_status, set_speed, lead_visible):
  values = {
    "ACA_StaACC": acc_status,
    "ACA_Zeitluecke": 2,
    "ACA_V_Wunsch": set_speed,
    "ACA_gemZeitl": 8 if lead_visible else 0,
  }
  # TODO: ACA_ID_StaACC, ACA_AnzDisplay, ACA_kmh_mph, ACA_PrioDisp, ACA_Aend_Zeitluecke

  return packer.make_can_msg("ACC_GRA_Anziege", bus, values)
