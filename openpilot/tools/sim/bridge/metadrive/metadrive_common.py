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
    if os.environ.get("METADRIVE_SIMPLE_RENDER"):
      # the default RGBCamera pipeline renders the scene with 16x MSAA into a
      # float HDR buffer plus a tonemap post-pass, which is prohibitively slow
      # on software rasterizers (CI); render straight into the 8-bit buffer,
      # keeping the terrain shader tag so the road still draws correctly
      from metadrive.constants import CameraTagStateKey, Semantics
      from metadrive.engine.core.terrain import Terrain
      cam = self.get_cam().node()
      cam.setTagStateKey(CameraTagStateKey.RGB)
      if os.environ.get("METADRIVE_FLAT_TERRAIN_CARD"):
        # flat quad terrain + flat-lit shader, ~2 texture taps per fragment
        from panda3d.core import NodePath, Shader
        here = os.path.dirname(os.path.abspath(__file__))
        dummy_np = NodePath("Dummy")
        dummy_np.setShader(Shader.load(Shader.SL_GLSL,
                                        os.path.join(here, "terrain_card.vert.glsl"),
                                        os.path.join(here, "terrain_ci.frag.glsl")))
        terrain_state = dummy_np.getState()
      elif os.environ.get("METADRIVE_CHEAP_TERRAIN"):
        # flat-lit terrain shader on the stock terrain mesh
        from metadrive.engine.asset_loader import AssetLoader
        from panda3d.core import NodePath, Shader
        vert = AssetLoader.file_path("../shaders", "terrain.vert.glsl")
        frag = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_ci.frag.glsl")
        dummy_np = NodePath("Dummy")
        dummy_np.setShader(Shader.load(Shader.SL_GLSL, vert, frag))
        terrain_state = dummy_np.getState()
      else:
        terrain_state = Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain.frag.glsl")
      cam.setTagState(Semantics.TERRAIN.label, terrain_state)
    else:
      super()._setup_effect()

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
