import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import Texture, GraphicsOutput


class CopyRamRGBCamera(RGBCamera):
  """Camera which copies its content into RAM during the render process, for faster image grabbing."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

  def get_rgb_array_cpu(self):
    origin_img = self.cpu_texture
    img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)
    img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
    img = img[:,:,:3] # RGBA to RGB
    # img = np.swapaxes(img, 1, 0)
    img = img[::-1] # Flip on vertical axis
    return img


class RGBCameraWide(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    lens = self.get_lens()
    lens.setFov(120)
    lens.setNear(0.1)


class RGBCameraRoad(CopyRamRGBCamera):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    lens = self.get_lens()
    lens.setFov(40)
    lens.setNear(0.1)
