
alert_file_path = "/data/openpilot/selfdrive/car/tesla/alert.msg"
alert_file_rw = "w"

def custom_alert_message(message):
  fo = open(alert_file_path, alert_file_rw)
  fo.write(message)
  fo.close()
