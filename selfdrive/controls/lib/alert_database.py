alerts = []
keys = ["id",
        "chime",
        "beep",
        "hud_alert",
        "screen_chime",
        "priority",
        "text_line_1",
        "text_line_2",
        "duration_sound",
        "duration_hud_alert",
        "duration_text"]


#car chimes: enumeration from dbc file. Chimes are for alerts and warnings
class CM:
  MUTE = 0
  SINGLE = 3
  DOUBLE = 4
  REPEATED = 1
  CONTINUOUS = 2


#car beepss: enumeration from dbc file. Beeps are for activ and deactiv
class BP:
  MUTE = 0
  SINGLE = 3
  TRIPLE = 2
  REPEATED = 1


# lert ids
class AI:
  ENABLE = 0
  DISABLE = 1
  SEATBELT = 2
  DOOR_OPEN = 3
  PEDAL_PRESSED = 4
  COMM_ISSUE = 5
  ESP_OFF = 6
  FCW = 7
  STEER_ERROR = 8
  BRAKE_ERROR = 9
  CALIB_INCOMPLETE = 10
  CALIB_INVALID = 11
  GEAR_NOT_D = 12
  MAIN_OFF = 13
  STEER_SATURATED = 14
  PCM_LOW_SPEED = 15
  THERMAL_DEAD = 16
  OVERHEAT = 17
  HIGH_SPEED = 18
  CONTROLSD_LAG = 19
  STEER_ERROR_ID = 100
  BRAKE_ERROR_ID = 101
  PCM_MISMATCH_ID = 102
  CTRL_MISMATCH_ID = 103
  SEATBELT_SD = 200
  DOOR_OPEN_SD = 201
  COMM_ISSUE_SD = 202
  ESP_OFF_SD = 203
  THERMAL_DEAD_SD = 204
  OVERHEAT_SD = 205
  CONTROLSD_LAG_SD = 206
  CALIB_INCOMPLETE_SD = 207
  CALIB_INVALID_SD = 208
  DRIVER_DISTRACTED = 300

class AH:
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 0x8]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]

class ET:
  ENABLE = 0
  NO_ENTRY = 1
  WARNING = 2
  SOFT_DISABLE = 3
  IMMEDIATE_DISABLE = 4
  USER_DISABLE = 5

def process_alert(alert_id, alert, cur_time, sound_exp, hud_exp, text_exp, alert_p):
  # INPUTS:
  # alert_id is mapped to the alert properties in alert_database
  # cur_time is current time
  # sound_exp is when the alert beep/chime is supposed to end 
  # hud_exp is when the hud visual is supposed to end 
  # text_exp is when the alert text is supposed to disappear 
  # alert_p is the priority of the current alert 
  # CM, BP, AH are classes defined in alert_database and they respresents chimes, beeps and hud_alerts
  if len(alert_id) > 0:
    # take the alert with higher priority
    alerts_present = filter(lambda a_id: a_id['id'] in alert_id, alerts)
    alert = sorted(alerts_present, key=lambda k: k['priority'])[-1]
    # check if we have a more important alert
    if alert['priority'] > alert_p:
      alert_p = alert['priority']
      sound_exp = cur_time + alert['duration_sound']
      hud_exp = cur_time + alert['duration_hud_alert']
      text_exp = cur_time + alert['duration_text']

  chime = CM.MUTE
  beep = BP.MUTE
  if cur_time < sound_exp:
    chime = alert['chime']
    beep = alert['beep']

  hud_alert = AH.NONE
  if cur_time < hud_exp:
    hud_alert = alert['hud_alert']

  alert_text = ["", ""]
  if cur_time < text_exp:
    alert_text = [alert['text_line_1'], alert['text_line_2']]

  if chime == CM.MUTE and beep == BP.MUTE and hud_alert == AH.NONE:   #and alert_text[0] is None and alert_text[1] is None:
    alert_p = 0
  return alert, chime, beep, hud_alert, alert_text, sound_exp, hud_exp, text_exp, alert_p

def process_hud_alert(hud_alert):
  # initialize to no alert
  fcw_display = 0
  steer_required = 0
  acc_alert = 0
  if hud_alert == AH.NONE:          # no alert 
    pass
  elif hud_alert == AH.FCW:         # FCW
    fcw_display = hud_alert[1]
  elif hud_alert == AH.STEER:       # STEER
    steer_required = hud_alert[1]
  else:                             # any other ACC alert
    acc_alert = hud_alert[1]

  return fcw_display, steer_required, acc_alert

def app_alert(alert_add):
  alerts.append(dict(zip(keys, alert_add)))

app_alert([AI.ENABLE,              CM.MUTE,     BP.SINGLE, AH.NONE,           ET.ENABLE,            2, "",                         "",                        .2, 0., 0.])
app_alert([AI.DISABLE,             CM.MUTE,     BP.SINGLE, AH.NONE,           ET.USER_DISABLE,      2, "",                         "",                        .2, 0., 0.])
app_alert([AI.SEATBELT,            CM.DOUBLE,   BP.MUTE,   AH.SEATBELT,       ET.NO_ENTRY,          1, "Comma Unavailable",        "Seatbelt Unlatched",      .4, 2., 3.])
app_alert([AI.DOOR_OPEN,           CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Door Open",               .4, 0., 3.])
app_alert([AI.PEDAL_PRESSED,       CM.DOUBLE,   BP.MUTE,   AH.BRAKE_PRESSED,  ET.NO_ENTRY,          1, "Comma Unavailable",        "Pedal Pressed",           .4, 2., 3.])
app_alert([AI.COMM_ISSUE,          CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Communcation Issues",     .4, 0., 3.])
app_alert([AI.ESP_OFF,             CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "ESP Off",                 .4, 0., 3.])
app_alert([AI.FCW,                 CM.REPEATED, BP.MUTE,   AH.FCW,            ET.WARNING,           3, "Risk of Collision",        "",                        1., 2., 3.])
app_alert([AI.STEER_ERROR,         CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Steer Error",             .4, 0., 3.])
app_alert([AI.BRAKE_ERROR,         CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Brake Error",             .4, 0., 3.])
app_alert([AI.CALIB_INCOMPLETE,    CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Calibration in Progress", .4, 0., 3.])
app_alert([AI.CALIB_INVALID,       CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Calibration Error",       .4, 0., 3.])
app_alert([AI.GEAR_NOT_D,          CM.DOUBLE,   BP.MUTE,   AH.GEAR_NOT_D,     ET.NO_ENTRY,          1, "Comma Unavailable",        "Gear not in D",           .4, 2., 3.])
app_alert([AI.MAIN_OFF,            CM.MUTE,     BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Main Switch Off",         .4, 0., 3.])
app_alert([AI.STEER_SATURATED,     CM.SINGLE,   BP.MUTE,   AH.STEER,          ET.WARNING,           2, "Take Control",             "Steer Control Saturated", 1., 2., 3.])
app_alert([AI.PCM_LOW_SPEED,       CM.MUTE,     BP.SINGLE, AH.STEER,          ET.WARNING,           2, "Comma disengaged",         "Speed too low",           .2, 2., 3.])
app_alert([AI.THERMAL_DEAD,        CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Thermal Unavailable",     .4, 0., 3.])
app_alert([AI.OVERHEAT,            CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "System Overheated",       .4, 0., 3.])
app_alert([AI.HIGH_SPEED,          CM.DOUBLE,   BP.MUTE,   AH.SPEED_TOO_HIGH, ET.NO_ENTRY,          1, "Comma Unavailable",        "Speed Too High",          .4, 2., 3.])
app_alert([AI.CONTROLSD_LAG,       CM.DOUBLE,   BP.MUTE,   AH.NONE,           ET.NO_ENTRY,          1, "Comma Unavailable",        "Controls Lagging",        .4, 0., 3.])
app_alert([AI.STEER_ERROR_ID,      CM.REPEATED, BP.MUTE,   AH.STEER,          ET.IMMEDIATE_DISABLE, 3, "Take Control Immediately", "Steer Error",             1., 3., 3.])
app_alert([AI.BRAKE_ERROR_ID,      CM.REPEATED, BP.MUTE,   AH.STEER,          ET.IMMEDIATE_DISABLE, 3, "Take Control Immediately", "Brake Error",             1., 3., 3.])
app_alert([AI.PCM_MISMATCH_ID,     CM.REPEATED, BP.MUTE,   AH.STEER,          ET.IMMEDIATE_DISABLE, 3, "Take Control Immediately", "Pcm Mismatch",            1., 3., 3.])
app_alert([AI.CTRL_MISMATCH_ID,    CM.REPEATED, BP.MUTE,   AH.STEER,          ET.IMMEDIATE_DISABLE, 3, "Take Control Immediately", "Ctrl Mismatch",           1., 3., 3.])
app_alert([AI.SEATBELT_SD,         CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Seatbelt Unlatched",      1., 3., 3.])
app_alert([AI.DOOR_OPEN_SD,        CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Door Open",               1., 3., 3.])
app_alert([AI.COMM_ISSUE_SD,       CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Technical Issues",        1., 3., 3.])
app_alert([AI.ESP_OFF_SD,          CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "ESP Off",                 1., 3., 3.])
app_alert([AI.THERMAL_DEAD_SD,     CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Thermal Unavailable",     1., 3., 3.])
app_alert([AI.OVERHEAT_SD,         CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "System Overheated",       1., 3., 3.])
app_alert([AI.CONTROLSD_LAG_SD,    CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Controls Lagging",        1., 3., 3.])
app_alert([AI.CALIB_INCOMPLETE_SD, CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Calibration in Progress", 1., 3., 3.])
app_alert([AI.CALIB_INVALID_SD,    CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      3, "Take Control Immediately", "Calibration Error",       1., 3., 3.])
app_alert([AI.DRIVER_DISTRACTED,   CM.REPEATED, BP.MUTE,   AH.STEER,          ET.SOFT_DISABLE,      2, "Take Control to Regain Speed", "User Distracted",     1., 1., 1.])
