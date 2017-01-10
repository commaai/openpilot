from cereal import car

class ET:
  ENABLE = 0
  NO_ENTRY = 1
  WARNING = 2
  SOFT_DISABLE = 3
  IMMEDIATE_DISABLE = 4
  USER_DISABLE = 5

class alert(object):
  def __init__(self, alert_text_1, alert_text_2, alert_type, visual_alert, audible_alert, duration_sound, duration_hud_alert, duration_text):
    self.alert_text_1 = alert_text_1
    self.alert_text_2 = alert_text_2
    self.alert_type = alert_type
    self.visual_alert = visual_alert if visual_alert is not None else "none"
    self.audible_alert = audible_alert if audible_alert is not None else "none"

    self.duration_sound = duration_sound
    self.duration_hud_alert = duration_hud_alert
    self.duration_text = duration_text

    # typecheck that enums are valid on startup
    tst = car.CarControl.new_message()
    tst.hudControl.visualAlert = self.visual_alert
    tst.hudControl.audibleAlert = self.audible_alert
  
  def __str__(self):
    return self.alert_text_1 + "/" + self.alert_text_2 + " " + str(self.alert_type) + "  " + str(self.visual_alert) + " " + str(self.audible_alert)

  def __gt__(self, alert2):
    return self.alert_type > alert2.alert_type

class AlertManager(object):
  alerts = {
    "enable":             alert("", "",                                            ET.ENABLE,       None,               "beepSingle", .2, 0., 0.),
    "disable":            alert("", "",                                            ET.USER_DISABLE, None,               "beepSingle", .2, 0., 0.),
    "pedalPressed":       alert("Comma Unavailable",        "Pedal Pressed",       ET.NO_ENTRY,     "brakePressed",     "chimeDouble", .4, 2., 3.),
    "driverDistracted":   alert("Take Control to Regain Speed", "User Distracted", ET.WARNING,      "steerRequired", "chimeRepeated", 1., 1., 1.),
    "steerSaturated":     alert("Take Control", "Steer Control Saturated",         ET.WARNING,      "steerRequired", "chimeSingle", 1., 2., 3.),
    "overheat":           alert("Take Control Immediately", "System Overheated",   ET.SOFT_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "controlsMismatch":   alert("Take Control Immediately", "Controls Mismatch",   ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "radarCommIssue":     alert("Take Control Immediately", "Radar Comm Issue",    ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "modelCommIssue":     alert("Take Control Immediately", "Model Comm Issue",    ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "controlsFailed":     alert("Take Control Immediately", "Controls Failed",     ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    # car errors
    "commIssue":          alert("Take Control Immediately","Communication Issues", ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "steerUnavailable":   alert("Take Control Immediately","Steer Error",          ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "steerTemporarilyUnavailable": alert("Take Control", "Steer Temporarily Unavailable",  ET.WARNING,   "steerRequired", "chimeRepeated", 1., 2., 3.),
    "brakeUnavailable":   alert("Take Control Immediately","Brake Error",          ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "gasUnavailable":     alert("Take Control Immediately","Gas Error",            ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "wrongGear":          alert("Take Control Immediately","Gear not D",           ET.SOFT_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "reverseGear":        alert("Take Control Immediately","Car in Reverse",       ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "doorOpen":           alert("Take Control Immediately","Door Open",            ET.SOFT_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "seatbeltNotLatched": alert("Take Control Immediately","Seatbelt Unlatched",   ET.SOFT_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "espDisabled":        alert("Take Control Immediately","ESP Off",              ET.SOFT_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
    "wrongCarMode":       alert("Comma Unavailable","Main Switch Off",             ET.NO_ENTRY,     None, "chimeDouble", .4, 0., 3.),
    "outOfSpace":         alert("Comma Unavailable","Out of Space",                ET.NO_ENTRY,     None, "chimeDouble", .4, 0., 3.),
    "ethicalDilemma":     alert("Take Control Immediately","Ethical Dilemma Detected", ET.IMMEDIATE_DISABLE, "steerRequired", "chimeRepeated", 1., 3., 3.),
  }
  def __init__(self):
    self.activealerts = []
    self.current_alert = None

  def alertPresent(self):
    return len(self.activealerts) > 0

  def alertShouldSoftDisable(self):
    return len(self.activealerts) > 0 and self.activealerts[0].alert_type == ET.SOFT_DISABLE

  def alertShouldDisable(self):
    return len(self.activealerts) > 0 and self.activealerts[0].alert_type >= ET.IMMEDIATE_DISABLE

  def add(self, alert_type, enabled = True):
    this_alert = self.alerts[str(alert_type)]

    # downgrade the alert if we aren't enabled
    if not enabled and this_alert.alert_type > ET.NO_ENTRY:
      this_alert = alert("Comma Unavailable" if this_alert.alert_text_1 != "" else "", this_alert.alert_text_2, ET.NO_ENTRY, None, "chimeDouble", .4, 0., 3.)

    # ignore no entries if we are enabled 
    if enabled and this_alert.alert_type < ET.WARNING:
      return

    self.activealerts.append(this_alert)
    self.activealerts.sort()

  def process_alerts(self, cur_time):
    if self.alertPresent():
      self.alert_start_time = cur_time
      self.current_alert = self.activealerts[0]
      print self.current_alert

    alert_text_1 = ""
    alert_text_2 = ""
    visual_alert = "none"
    audible_alert = "none"

    if self.current_alert is not None:
      # ewwwww
      if self.alert_start_time + self.current_alert.duration_sound > cur_time:
        audible_alert = self.current_alert.audible_alert

      if self.alert_start_time + self.current_alert.duration_hud_alert > cur_time:
        visual_alert = self.current_alert.visual_alert

      if self.alert_start_time + self.current_alert.duration_text > cur_time:
        alert_text_1 = self.current_alert.alert_text_1
        alert_text_2 = self.current_alert.alert_text_2

    # reset
    self.activealerts = []

    return alert_text_1, alert_text_2, visual_alert, audible_alert

