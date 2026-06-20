from cereal import messaging


def generate_livePose():
  msg = messaging.new_message('livePose')
  meas = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'xStd': 0.0, 'yStd': 0.0, 'zStd': 0.0, 'valid': True}
  msg.livePose.orientationNED = meas
  msg.livePose.velocityDevice = meas
  msg.livePose.angularVelocityDevice = meas
  msg.livePose.accelerationDevice = meas
  msg.livePose.inputsOK = True
  msg.livePose.posenetOK = True
  msg.livePose.sensorsOK = True
  return msg
