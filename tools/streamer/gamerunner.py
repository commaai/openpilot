import random
import threading

from common.params import Params
import cereal.messaging as messaging
from tools.streamer.tasks import Tasks 
from selfdrive.test.helpers import set_params_enabled
from common.realtime import config_realtime_process, Priority

from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

#from tools.sim.bridge.common import VehicleState 
#from common.realtime import DT_DMON, Ratekeeper

sm = messaging.SubMaster(['carControl', 'controlsState'])


class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = None
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True
    self.cruise_sp = 0.0


c_W = 1928
c_H = 1208
W, H = round(c_W//1.4), round(c_H//1.4)

class myBridge():
  def __init__(self):
    self._threads = []
    set_params_enabled()
    self.params = Params()
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())
    #self.params.remove("CalibrationParams")
    self.params.put_bool("WideCameraOnly", False)
    self.params.put_bool("ObdMultiplexingDisabled", True)
    self._exit_event = threading.Event()
    self._exit_event.clear()
    self.sm = sm
    self.vs = VehicleState()
    self.W = W
    self.H = H

  def exit(self):
    self._exit_event.set()
      
  def is_exiting(self):
    return self._exit_event.is_set()

  def sub_threads(self):
    self._threads.append(threading.Thread(target=Tasks.panda, args=(self,)))
    self._threads.append(threading.Thread(target=Tasks.peripherals, args=(self,)))
    self._threads.append(threading.Thread(target=Tasks.dm, args=(self,)))
    self._threads.append(threading.Thread(target=Tasks.can, args=(self,)))
    
    for t in self._threads:
      t.start()      
      

  def game_runner(self):
    config = dict(
      window_size=(W, H),
      use_render=True, # need this on to get the camera
      offscreen_render=True,
      streamer=True,
      streamer_ipc="ipc:///tmp/metadrive_window",
      mouse_look=False,
      manual_control=True,
      rgb_clip=True,
      
      random_lane_width=True,
      random_lane_num=True,
      openpilot_control=True,
      random_agent_model=True,
      environment_num=6,
      traffic_density=0.01,
      camera_dist= 0.0,
      camera_pitch= 0,
      camera_heading= 0,
      camera_yaw= 0,
      camera_height= 0.6,
      camera_fov= 120,
      camera_aspect_ratio= (W/H),
      camera_smooth= False,
      show_interface= False,
      physics_world_step_size = 0.02, #.03
      map="SSSSSSSSCSSSSS",
      start_seed=random.randint(0, 1000),
      vehicle_config = dict(image_source="rgb_camera", 
                            rgb_camera= (1,1),
                            stack_size=1,
                            rgb_clip=True,
                            show_navi_mark=False,
                            max_engine_force=2500,
                            max_brake_force=300,
                            max_steering = 60,
                            wheel_friction=4.0,
                            ),
    )

    env = MetaDriveEnv(config)
    #rk = Ratekeeper(4, print_delay_threshold=None)
    try:

      o = env.reset()
      print(HELP_MESSAGE)
      env.vehicle.expert_takeover = False
      assert isinstance(o, dict)
      while not self._exit_event.is_set():
        sm.update(0)
        _,_,_,step_infos = env.step([0,0])
        self.vs.speed = step_infos['velocity']
        self.vs.angle = step_infos['steering']
        self.vs.cruise_sp = step_infos['policy_cruise_sp']
        self.vs.is_engaged = step_infos['policy_engaged']
        self.vs.ignition = step_infos['ignition']
    except Exception as e:
      raise e
    finally:
      env.close()
        
def main():
  config_realtime_process(1, Priority.CTRL_HIGH)
  bridge = myBridge() # this is my bridge
  bridge.sub_threads() # this part is nice
  bridge.game_runner() # this is my favorite part

if __name__ == "__main__":
  main()