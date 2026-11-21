from openpilot.cereal import messaging


def generate_deviceMotion():
  msg = messaging.new_message('deviceMotion')
  meas = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'xStd': 0.0, 'yStd': 0.0, 'zStd': 0.0, 'valid': True}
  msg.deviceMotion.orientationNED = meas
  msg.deviceMotion.velocityDevice = meas
  msg.deviceMotion.angularVelocityDevice = meas
  msg.deviceMotion.accelerationDevice = meas
  msg.deviceMotion.inputsOK = True
  msg.deviceMotion.posenetOK = True
  msg.deviceMotion.sensorsOK = True
  return msg
