import os

import numpy as np

from metadrive.component.sensors.rgb_camera import RGBCamera
from panda3d.core import Texture, GraphicsOutput


class CopyRamRGBCamera(RGBCamera):
  """Camera which copies its content into RAM during the render process, for faster image grabbing."""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cpu_texture = Texture()
    self.buffer.addRenderTexture(self.cpu_texture, GraphicsOutput.RTMCopyRam)

  def _setup_effect(self):
    if not os.environ.get("METADRIVE_SIMPLE_RENDER"):
      return super()._setup_effect()

    from metadrive.constants import CameraTagStateKey, Semantics
    from metadrive.engine.core.terrain import Terrain
    from panda3d.core import NodePath, Shader

    camera = self.get_cam().node()
    camera.setTagStateKey(CameraTagStateKey.RGB)
    if os.environ.get("METADRIVE_FLAT_TERRAIN_CARD"):
      here = os.path.dirname(os.path.abspath(__file__))
      dummy = NodePath("Dummy")
      dummy.setShader(Shader.load(Shader.SL_GLSL, os.path.join(here, "terrain_card.vert.glsl"),
                                  os.path.join(here, "terrain_ci.frag.glsl")))
      terrain_state = dummy.getState()
    else:
      terrain_state = Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain.frag.glsl")
    camera.setTagState(Semantics.TERRAIN.label, terrain_state)

  def get_rgb_array_cpu(self):
    origin_img = self.cpu_texture
    img = np.frombuffer(origin_img.getRamImageAs("RGB").getData(), dtype=np.uint8)
    img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 3))
    img = img[::-1]  # Flip on vertical axis
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

