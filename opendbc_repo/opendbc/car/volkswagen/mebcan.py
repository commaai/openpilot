def create_steering_control(packer, bus, apply_curvature, lkas_enabled, power=0):
  values = {
    "Curvature": abs(apply_curvature),  # in rad/m
    "Curvature_VZ": 1 if apply_curvature > 0 and lkas_enabled else 0,
    "Power": power if lkas_enabled else 0,
    "RequestStatus": 4 if lkas_enabled else 2,
    "HighSendRate": lkas_enabled,
  }
  return packer.make_can_msg("HCA_03", bus, values)


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


def create_lka_hud_control(packer, bus, ldw_stock_values, lat_active, steering_pressed, hud_alert, hud_control, sound_alert=False):
  display_mode = 1 if lat_active else 0  # travel assist style showing yellow lanes when op is active

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
    "LDW_Gong": sound_alert,
    "LDW_Status_LED_gelb": 1 if lat_active and steering_pressed else 0,
    "LDW_Status_LED_gruen": 1 if lat_active and not steering_pressed else 0,
    "LDW_Lernmodus_links": 3 + display_mode if hud_control.leftLaneDepart else 1 + hud_control.leftLaneVisible + display_mode,
    "LDW_Lernmodus_rechts": 3 + display_mode if hud_control.rightLaneDepart else 1 + hud_control.rightLaneVisible + display_mode,
    "LDW_Texte": hud_alert,
  })
  return packer.make_can_msg("LDW_02", bus, values)


def create_acc_buttons_control(packer, bus, gra_stock_values, cancel=False, resume=False, up=False, down=False):
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
    "GRA_Tip_Wiederaufnahme": resume or up,
    "GRA_Tip_Setzen": down,
  })
  return packer.make_can_msg("GRA_ACC_01", bus, values)


ACC_CTRL_ERROR    = 6
ACC_CTRL_OVERRIDE = 4
ACC_CTRL_ACTIVE   = 3
ACC_CTRL_ENABLED  = 2
ACC_CTRL_DISABLED = 0

ACC_HMS_RAMP_RELEASE = 5
ACC_HMS_RELEASE      = 4
ACC_HMS_HOLD         = 1
ACC_HMS_NO_REQUEST   = 0

ACC_HUD_ERROR    = 6
ACC_HUD_OVERRIDE = 4
ACC_HUD_ACTIVE   = 3
ACC_HUD_ENABLED  = 2
ACC_HUD_DISABLED = 0


def get_acc_control(CS, CC, long_override):
  if CS.accFaulted:
    acc_control = ACC_CTRL_ERROR  # error state
  elif CC.enabled:
    if long_override:
      acc_control = ACC_CTRL_OVERRIDE  # overriding
    else:
      acc_control = ACC_CTRL_ACTIVE  # active long control state
  elif CS.cruiseState.available:
    acc_control = ACC_CTRL_ENABLED  # long control ready
  else:
    acc_control = ACC_CTRL_DISABLED  # long control deactivated state

  return acc_control


def get_acc_hold_type(CS, CC, starting, stopping, esp_hold, long_override, long_override_begin, long_disabling):
  # warning: car is reacting to hold mechanic even with long control off
  if CS.accFaulted:
    acc_hold_type = ACC_HMS_NO_REQUEST  # no hold request
  elif not CC.enabled:
    if long_disabling:
      acc_hold_type = ACC_HMS_RAMP_RELEASE  # ramp release of requests right after disabling long control (prevents car error with EPB at low speed)
    else:
      acc_hold_type = ACC_HMS_NO_REQUEST  # no hold request
  elif long_override:
    if long_override_begin:
      acc_hold_type = ACC_HMS_RAMP_RELEASE  # ramp release of requests at the beginning of override (prevents car error with EPB at low speed)
    else:
      acc_hold_type = ACC_HMS_NO_REQUEST  # overriding / no request
  elif starting:
    acc_hold_type = ACC_HMS_RELEASE  # release request and startup
  elif stopping:
    acc_hold_type = ACC_HMS_HOLD  # hold while stopping/stopped
  else:
    acc_hold_type = ACC_HMS_NO_REQUEST  # no hold request

  return acc_hold_type


def create_acc_accel_control(packer, bus, CCP, acc_type, acc_enabled, accel, acc_control, acc_hold_type,
                             stopping, starting, esp_hold, speed, long_override, travel_assist_available):
  # active longitudinal control disables one pedal driving (regen mode) while using overriding mechanism
  # error mitigation when stopping or stopped: (newer gen cars can be very sensitive)
  # - send 0 m stopping distance for cars in kind of parameterized stopping mode (stopping accel -0.2 seen for those cars)
  # -> this mode is seen for different cars with same firmware radars so could be a coded operational mode
  # - jerk and control limits values set to 0 when fully stopped
  # - set accel to 0 / no stop accel for full stop (seems to be compatible with old (non 0 stop accel) and new gen, because HMS state holds the car anyways)
  # - stopping command sent as long as actually stopping
  commands = []

  # ACC_Anhalteweg: when stopping: MEB: values <> 0 the car can execute a hard brake probably if target is too close, MQBEvo: value 0 results in hard brake
  terminal_rollout = 0

  full_stop = stopping and esp_hold
  full_stop_no_start = esp_hold and not starting
  actually_stopping = stopping and not esp_hold

  if acc_enabled:
    if long_override:  # the car expects a non-inactive accel while overriding
      acceleration = CCP.ACCEL_OVERRIDE  # original ACC still sends active accel in this case (seamless experience)
    elif full_stop:
      acceleration = CCP.ACCEL_INACTIVE  # inactive accel, newer gen >2024 error of not neutral value
    else:
      acceleration = accel
  else:
    acceleration = CCP.ACCEL_INACTIVE  # inactive accel

  values = {
    "ACC_Typ":                    acc_type,
    "ACC_Status_ACC":             acc_control,
    "ACC_StartStopp_Info":        acc_enabled,
    "ACC_Sollbeschleunigung_02":  acceleration,
    "ACC_zul_Regelabw_unten":     0,
    "ACC_zul_Regelabw_oben":      0,
    "ACC_neg_Sollbeschl_Grad_02": CCP.JERK_LIMIT if acc_control in (ACC_CTRL_ACTIVE, ACC_CTRL_OVERRIDE) and not full_stop_no_start else 0,
    "ACC_pos_Sollbeschl_Grad_02": CCP.JERK_LIMIT if acc_control in (ACC_CTRL_ACTIVE, ACC_CTRL_OVERRIDE) and not full_stop_no_start else 0,
    "ACC_Anfahren":               0,  # always zero, stock uses ACC_Anforderung_HMS
    "ACC_Anhalten":               1 if actually_stopping else 0,
    "ACC_Anhalteweg":             terminal_rollout if actually_stopping else 20.46,
    "ACC_Anforderung_HMS":        acc_hold_type,
    "ACC_AKTIV_regelt":           1 if acc_control == ACC_CTRL_ACTIVE else 0,
    "Speed":                      speed,
    "SET_ME_0XFE":                0xFE,
    "SET_ME_0X1":                 0x1,
    "SET_ME_0X9":                 0x9,
  }

  commands.append(packer.make_can_msg("ACC_18", bus, values))

  if travel_assist_available:
    # satisfy car to prevent errors when pressing Travel Assist Button
    values_ta = {
       "Travel_Assist_Status":    4 if acc_enabled else 2,
       "Travel_Assist_Request":   0,
       "Travel_Assist_Available": 1,
    }

    commands.append(packer.make_can_msg("TA_01", bus, values_ta))

  return commands


def get_acc_hud_status(CS, CC, long_override):
  if CS.accFaulted:
    acc_hud_control = ACC_HUD_ERROR  # error state
  elif CC.enabled:
    if long_override:
      acc_hud_control = ACC_HUD_OVERRIDE  # overriding
    else:
      acc_hud_control = ACC_HUD_ACTIVE  # active
  elif CS.cruiseState.available:
    acc_hud_control = ACC_HUD_ENABLED  # inactive
  else:
    acc_hud_control = ACC_HUD_DISABLED  # deactivated

  return acc_hud_control


def get_desired_gap(distance_bars, desired_gap, current_gap_signal):
  # mapping desired gap to correct signal of corresponding distance bar
  gap = 0

  if distance_bars == current_gap_signal:
    gap = desired_gap

  return gap


def create_acc_hud_control(packer, bus, acc_control, set_speed, lead_visible, distance_bars, show_distance_bars, esp_hold, distance, desired_gap, fcw_alert):
  values = {
    "ACC_Status_ACC":                acc_control,
    "ACC_Tempolimit":                0,
    "ACC_Wunschgeschw_02":           set_speed if set_speed < 250 else 327.36,
    "ACC_Gesetzte_Zeitluecke":       distance_bars, # 5 distance bars available (3 are used by OP)
    "ACC_Display_Prio":              0 if fcw_alert and acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 1, # probably keeping warning in front
    "ACC_Optischer_Fahrerhinweis":   1 if fcw_alert and acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # enables optical warning
    "ACC_Akustischer_Fahrerhinweis": 3 if fcw_alert and acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # enables sound warning
    "ACC_Texte_Zusatzanz_02":        11 if fcw_alert and acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # type of warning: Break!
    "ACC_Abstandsindex_02":          569, # seems to be default for MEB but is not static in every case
    "ACC_EGO_Fahrzeug":              2 if fcw_alert and acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else
                                     (1 if acc_control == ACC_HUD_ACTIVE else 0), # red car warn symbol for fcw
    "Lead_Type_Detected":            1 if lead_visible else 0, # object should be displayed
    "Lead_Type":                     3 if lead_visible else 0, # displaying a car
    "Lead_Distance":                 distance if lead_visible else 0, # hud distance of object
    "ACC_Enabled":                   1 if acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0,
    "ACC_Standby_Override":          1 if acc_control != ACC_HUD_ACTIVE else 0,
    "Street_Color":                  1 if acc_control in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # light grey (1) or dark (0) street
    "Lead_Brightness":               3 if acc_control == ACC_HUD_ACTIVE else 0, # object shows in color
    "Zeitluecke_1":                  get_desired_gap(distance_bars, desired_gap, 1), # desired distance to lead object for distance bar 1
    "Zeitluecke_2":                  get_desired_gap(distance_bars, desired_gap, 2), # desired distance to lead object for distance bar 2
    "Zeitluecke_3":                  get_desired_gap(distance_bars, desired_gap, 3), # desired distance to lead object for distance bar 3
    "Zeitluecke_4":                  get_desired_gap(distance_bars, desired_gap, 4), # desired distance to lead object for distance bar 4
    "Zeitluecke_5":                  get_desired_gap(distance_bars, desired_gap, 5), # desired distance to lead object for distance bar 5
    "Zeitluecke_Farbe":              1 if acc_control in (ACC_HUD_ENABLED, ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # yellow (1) or white (0) time gap
    "ACC_Anzeige_Zeitluecke":        show_distance_bars if acc_control != ACC_HUD_DISABLED else 0, # show distance bar selection
    "SET_ME_0X1":                    0x1,    # unknown
    "SET_ME_0X6A":                   0x6A,   # unknown
    "SET_ME_0XFFFF":                 0xFFFF, # unknown
    "SET_ME_0X7FFF":                 0x7FFF, # unknown
  }

  return packer.make_can_msg("ACC_19", bus, values)


def create_capacitive_wheel_touch(packer, bus, lat_active, klr_stock_values):
  values = {s: klr_stock_values[s] for s in [
    "COUNTER",
    "KLR_Touchintensitaet_1",
    "KLR_Touchintensitaet_2",
    "KLR_Touchintensitaet_3",
    "KLR_Touchauswertung",
  ]}

  if lat_active:
    values.update({
      "COUNTER": (klr_stock_values["COUNTER"] + 1) % 16,
      "KLR_Touchintensitaet_1": 80,
      "KLR_Touchintensitaet_2": 200,
      "KLR_Touchintensitaet_3": 10,
      "KLR_Touchauswertung": 10,
    })
  return packer.make_can_msg("KLR_01", bus, values)
