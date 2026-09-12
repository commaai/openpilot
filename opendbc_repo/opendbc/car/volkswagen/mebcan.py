from opendbc.car import Bus, structs
from opendbc.can import CANDefine
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.volkswagen.values import DBC

LongCtrlState = structs.CarControl.Actuators.LongControlState


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


ACC_HUD_ERROR    = 6
ACC_HUD_OVERRIDE = 4
ACC_HUD_ACTIVE   = 3
ACC_HUD_ENABLED  = 2
ACC_HUD_DISABLED = 0


class MebLongStateMachine:
  HOLD_RELEASE_SPEED = 5 * CV.KPH_TO_MS

  def __init__(self, CP, CCP):
    self.CCP = CCP
    self.RAMP_FRAMES = 10 // CCP.ACC_CONTROL_STEP  # 100 ms

    self.disengage_ramp_counter = 0  # always ramp when disengaging

    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])
    self.acc_status_vals = {v: k for k, v in can_define.dv['ACC_18']['ACC_Status_ACC'].items()}
    self.acc_hold_type_vals = {v: k for k, v in can_define.dv['ACC_18']['ACC_Anforderung_HMS'].items()}

    self.prev_acc_hold_type = self.acc_hold_type_vals['KEINE_ANFORDERUNG']  # no request
    self.acc_status = self.acc_status_vals['ACC_OFF_HAUPTSCHALTER_AUS']  # last acc status, read by HUD msg

  def _get_acc_status(self, CS, CC) -> int:
    # stateless
    # NOTE: stock TSK and camera goes to 5 on disengage independently which we don't model, but hasn't been shown to fault without it
    if CS.out.accFaulted:
      return self.acc_status_vals['REVERSIBLER_FEHLER_IM_ACC_SYSTEM']
    elif CC.enabled:
      return self.acc_status_vals['ACC_OVERRIDE' if CC.cruiseControl.override else 'ACC_AKTIV_REGELT']
    elif CS.out.cruiseState.available:
      return self.acc_status_vals['ACC_STANDBY']
    else:
      return self.acc_status_vals['ACC_OFF_HAUPTSCHALTER_AUS']  # disabled

  def _get_hold_type(self, CS, CC) -> int:
    # warning: car is reacting to hold mechanic even with long control off
    # HALTEN -> KEINE_ANFORDERUNG causes the car to fault into park, so both branches below put a ramp in
    # between: disengaging always ramps, and while engaged a release ramps until 5 kph
    # NOTE: this allows KEINE_ANFORDERUNG -> ANFAHREN, but we haven't observed a fault due to this yet
    # TODO: camera can send 7 on disengage at a stop which we don't fully understand yet
    stopping = CC.actuators.longControlState == LongCtrlState.stopping
    starting = CC.actuators.longControlState == LongCtrlState.pid and CS.esp_hold_confirmation
    long_active = CC.longActive and not CS.out.accFaulted  # catches it one frame earlier, not sure if needed

    if not long_active:
      # Stock goes to RAMP for as long as TSK_Status is 5 usually, 100ms seems fine to mimic that behavior.
      # Stock stays active for gas press, but we go inactive
      if self.disengage_ramp_counter > 0:
        acc_hold_type = self.acc_hold_type_vals['LOESEN_UEBER_RAMPE']  # ramp
        self.disengage_ramp_counter -= 1
      else:
        acc_hold_type = self.acc_hold_type_vals['KEINE_ANFORDERUNG']  # no request

    else:
      was_engaged = self.disengage_ramp_counter == self.RAMP_FRAMES
      self.disengage_ramp_counter = self.RAMP_FRAMES  # prep ramp if we disengage

      if stopping:
        acc_hold_type = self.acc_hold_type_vals['HALTEN']  # stopping/stopped, allowed at any time
      elif starting:
        acc_hold_type = self.acc_hold_type_vals['ANFAHREN']  # resume after reaching full stop
      else:
        # After aborting a stop or finishing starting, we need to send RAMP until we hit 5 kph or go long inactive,
        # only if we didn't just re-engage
        releasing = was_engaged and self.prev_acc_hold_type in (self.acc_hold_type_vals['HALTEN'],
                                                                self.acc_hold_type_vals['ANFAHREN'],
                                                                self.acc_hold_type_vals['LOESEN_UEBER_RAMPE'])

        if releasing and CS.out.vEgo < self.HOLD_RELEASE_SPEED:
          acc_hold_type = self.acc_hold_type_vals['LOESEN_UEBER_RAMPE']  # ramp
        else:
          acc_hold_type = self.acc_hold_type_vals['KEINE_ANFORDERUNG']  # no request

    return acc_hold_type

  def update(self, CS, CC, accel) -> tuple[float, int, int, bool, bool]:
    acc_status = self._get_acc_status(CS, CC)
    acc_hold_type = self._get_hold_type(CS, CC)

    # transition to inactive accel and jerks as soon as we enter ESP standstill
    requesting_hold = acc_hold_type == self.acc_hold_type_vals['HALTEN']
    held = requesting_hold and CS.esp_hold_confirmation
    if not CC.enabled or held:
      accel = self.CCP.ACCEL_INACTIVE

    # hold requested but the car hasn't reached standstill yet
    braking_to_stop = requesting_hold and not CS.esp_hold_confirmation

    # driving off from a hold
    leaving_standstill = acc_hold_type == self.acc_hold_type_vals['ANFAHREN']

    self.prev_acc_hold_type = acc_hold_type
    self.acc_status = acc_status
    return accel, acc_status, acc_hold_type, braking_to_stop, leaving_standstill


def create_acc_accel_control(packer, bus, CCP, acc_type, acc_enabled, accel, acc_status, acc_hold_type,
                             braking_to_stop, leaving_standstill, speed, travel_assist_available):
  # active longitudinal control disables one pedal driving (regen mode) while using overriding mechanism
  # error mitigation when stopping or stopped: (newer gen cars can be very sensitive)
  # - send 0 m stopping distance for cars in kind of parameterized stopping mode (stopping accel -0.2 seen for those cars)
  # -> this mode is seen for different cars with same firmware radars so could be a coded operational mode
  # - jerk and control limits values set inactive together when fully stopped
  # - set accel to 0 / no stop accel for full stop (seems to be compatible with old (non 0 stop accel) and new gen, because HMS state holds the car anyways)
  # - stopping command sent while requesting stop but ESP is not in standstill
  commands = []

  # ACC_Anhalteweg: when stopping: MEB: values <> 0 the car can execute a hard brake probably if target is too close, MQBEvo: value 0 results in hard brake
  terminal_rollout = 0

  values = {
    "ACC_Typ":                    acc_type,
    "ACC_Status_ACC":             acc_status,
    "ACC_StartStopp_Info":        acc_enabled,
    "ACC_Sollbeschleunigung_02":  accel,
    "ACC_zul_Regelabw_unten":     0,
    "ACC_zul_Regelabw_oben":      0,
    "ACC_neg_Sollbeschl_Grad_02": CCP.JERK_LIMIT if accel != CCP.ACCEL_INACTIVE else 0,
    "ACC_pos_Sollbeschl_Grad_02": CCP.JERK_LIMIT if accel != CCP.ACCEL_INACTIVE else 0,
    # NOTE: gen1 stock sets this while launching, gen2 stock never does
    "ACC_Anfahren":               1 if leaving_standstill else 0,
    "ACC_Anhalten":               1 if braking_to_stop else 0,
    "ACC_Anhalteweg":             terminal_rollout if braking_to_stop else 20.46,
    "ACC_Anforderung_HMS":        acc_hold_type,
    "ACC_AKTIV_regelt":           0,  # always zero, stock uses ACC_Status_ACC
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


def create_acc_hud_control(packer, bus, acc_status, set_speed, lead_visible, distance_bars, show_distance_bars, distance, fcw_alert):
  values = {
    "ACC_Status_ACC":                acc_status,
    "ACC_Tempolimit":                0,
    "ACC_Wunschgeschw_02":           set_speed if set_speed < 250 else 327.36,
    "ACC_Gesetzte_Zeitluecke":       distance_bars, # 5 distance bars available (3 are used by OP)
    "ACC_Display_Prio":              0 if fcw_alert and acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 1, # probably keeping warning in front
    "ACC_Optischer_Fahrerhinweis":   1 if fcw_alert and acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # enables optical warning
    "ACC_Akustischer_Fahrerhinweis": 3 if fcw_alert and acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # enables sound warning
    "ACC_Texte_Zusatzanz_02":        11 if fcw_alert and acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # type of warning: Break!
    "ACC_Abstandsindex_02":          569, # seems to be default for MEB but is not static in every case
    "ACC_EGO_Fahrzeug":              2 if fcw_alert and acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else
                                     (1 if acc_status == ACC_HUD_ACTIVE else 0), # red car warn symbol for fcw
    "Lead_Type_Detected":            1 if lead_visible else 0, # object should be displayed
    "Lead_Type":                     3 if lead_visible else 0, # displaying a car
    "Lead_Distance":                 distance if lead_visible else 0, # hud distance of object
    "ACC_Enabled":                   1 if acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0,
    "ACC_Standby_Override":          1 if acc_status != ACC_HUD_ACTIVE else 0,
    "Street_Color":                  1 if acc_status in (ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # light grey (1) or dark (0) street
    "Lead_Brightness":               3 if acc_status == ACC_HUD_ACTIVE else 0, # object shows in color
    # TODO: a nice speed dependent bar distance
    "Zeitluecke_1":                  0, # desired distance to lead object for distance bar 1
    "Zeitluecke_2":                  0, # desired distance to lead object for distance bar 2
    "Zeitluecke_3":                  0, # desired distance to lead object for distance bar 3
    "Zeitluecke_4":                  0, # desired distance to lead object for distance bar 4
    "Zeitluecke_5":                  0, # desired distance to lead object for distance bar 5
    "Zeitluecke_Farbe":              1 if acc_status in (ACC_HUD_ENABLED, ACC_HUD_ACTIVE, ACC_HUD_OVERRIDE) else 0, # yellow (1) or white (0) time gap
    "ACC_Anzeige_Zeitluecke":        show_distance_bars if acc_status != ACC_HUD_DISABLED else 0, # show distance bar selection
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
