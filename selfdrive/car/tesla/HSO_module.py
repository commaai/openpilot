#human steer override module
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import custom_alert as customAlert
import time

def _current_time_millis():
  return int(round(time.time() * 1000))

class HSOController(object):
    def __init__(self,carcontroller):
        self.CC = carcontroller
        self.frame_humanSteered = 0
    

    def update_stat(self,CS,enabled,actuators,frame):
        human_control = False
        if (CS.steer_override>0):
            self.frame_humanSteered = frame
            customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
        else:
            if (frame - self.frame_humanSteered < 5.): # Need more human testing of handoff timing
                if (CS.cstm_btns.get_button_status("steer") >0) and enabled:
                    if (frame - self.frame_humanSteered < 50): # Need more human testing of handoff timing
                    # Find steering difference between visiond model and human (no need to do every frame if we run out of CPU):
                        steer_current=(CS.angle_steers)  # Formula to convert current steering angle to match apply_steer calculated number
                        apply_steer = int(-actuators.steerAngle)
                        angle = abs(apply_steer-steer_current)
                        if angle > 5.:
                            self.frame_humanSteered = frame
                            customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
        if enabled:
            if CS.cstm_btns.get_button_status("steer") > 0:
              if (frame - self.frame_humanSteered < 50):
                human_control = True
                CS.cstm_btns.set_button_status("steer",3)
                customAlert.custom_alert_message("Manual Steering Enabled",CS,50)
              else:
                CS.cstm_btns.set_button_status("steer",2)
        else:
            if CS.cstm_btns.get_button_status("steer") > 0:
              CS.cstm_btns.set_button_status("steer",1)
        return human_control and enabled
