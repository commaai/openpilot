alert_file_path = "/data/openpilot/selfdrive/car/tesla/alert.msg"
alert_file_rw = "w"

def custom_alert_message(message,CS,duration):
  fo = open(alert_file_path, alert_file_rw)
  fo.write(message)
  fo.close()
  CS.custom_alert_counter = duration


def update_custom_alert(CS):
  if (CS.custom_alert_counter > 0):
    CS.custom_alert_counter -= 1
    if (CS.custom_alert_counter ==0):
      custom_alert_message("",CS,0)
      CS.custom_alert_counter = -1
