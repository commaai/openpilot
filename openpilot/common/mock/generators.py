from openpilot.cereal import messaging


def generate_devicePose():
  msg = messaging.new_message('devicePose')
  meas = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'xStd': 0.0, 'yStd': 0.0, 'zStd': 0.0, 'valid': True}
  msg.devicePose.orientationNED = meas
  msg.devicePose.velocityDevice = meas
  msg.devicePose.angularVelocityDevice = meas
  msg.devicePose.accelerationDevice = meas
  msg.devicePose.inputsOK = True
  msg.devicePose.posenetOK = True
  msg.devicePose.sensorsOK = True
  return msg
