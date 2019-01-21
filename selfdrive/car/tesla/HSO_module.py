#human steer override module
from selfdrive.config import Conversions as CV
import selfdrive.messaging as messaging
import time

def _current_time_millis():
  return int(round(time.time() * 1000))

class HSOController(object):
    def __init__(self,carcontroller):
        self.CC = carcontroller
        self.frame_humanSteered = 0
    

    def update_stat(self,CS,enabled,actuators,frame):
        human_control = False
        if CS.enableHSO and enabled:
          #if steering but not by ALCA
          if (CS.right_blinker_on or CS.left_blinker_on) and (self.CC.ALCA.laneChange_enabled <= 1):
            self.frame_humanSteered = frame
          if (CS.steer_override > 0): 
            self.frame_humanSteered = frame
          else:
            if (frame - self.frame_humanSteered < 50): # Need more human testing of handoff timing
              # Find steering difference between visiond model and human (no need to do every frame if we run out of CPU):
              steer_current=(CS.angle_steers)  # Formula to convert current steering angle to match apply_steer calculated number
              apply_steer = int(-actuators.steerAngle)
              angle = abs(apply_steer-steer_current)
              if angle > 5.:
                self.frame_humanSteered = frame
        if enabled:
            if CS.enableHSO:
              if (frame - self.frame_humanSteered < 50):
                human_control = True
                #CS.cstm_btns.set_button_status("steer",3)
                CS.UE.custom_alert_message(3,"Manual Steering Enabled",51,4)
              #else:
              #  CS.cstm_btns.set_button_status("steer",2)
        #else:
        #    if CS.cstm_btns.get_button_status("steer") > 0:
        #      CS.cstm_btns.set_button_status("steer",1)
        return human_control and enabled
