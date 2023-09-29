import numpy as np

from multiprocessing.connection import Connection
from metadrive.envs.metadrive_env import MetaDriveEnv


def metadrive_process(dual_camera: bool, config: dict, camera_conn: Connection, control_conn: Connection):
  env = MetaDriveEnv(config)

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    img = cam.perceive(env.vehicle, clip=False)
    if type(img) != np.ndarray:
      img = img.get() # convert cupy array to numpy
    return img

  