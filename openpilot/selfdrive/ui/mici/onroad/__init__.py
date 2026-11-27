import numpy as np
import raylib_nb as rl
from openpilot.system.ui.lib.application import gui_app, ASSETS_DIR

SIDE_PANEL_WIDTH = 72
DIMMED_ICON_ALPHA = 0.32  # side panel icons while disengaged


def blend_colors(a: rl.Color, b: rl.Color, f: float) -> rl.Color:
  h0, s0, v0 = (hsv0 := rl.color_to_hsv(a)).x, hsv0.y, hsv0.z
  h1, s1, v1 = (hsv1 := rl.color_to_hsv(b)).x, hsv1.y, hsv1.z
  dh = ((h1 - h0 + 180) % 360) - 180  # shortest hue delta
  color = rl.color_from_hsv((h0 + f * dh) % 360,
                            s0 + f * (s1 - s0),
                            v0 + f * (v1 - v0))
  color.a = round(a.a + f * (b.a - a.a))
  return color


def glow_highlight_texture(asset: str, size: int, center: tuple[float, float] = (0.5, 0.5),
                           radius: tuple[float, float] = (0.5, 0.5)) -> rl.Texture:
  image = gui_app._load_image_from_path(str(ASSETS_DIR / asset), size, size)
  w, h = image.width, image.height
  alpha = np.frombuffer(image.to_bytes(w * h * 4), dtype=np.uint8).reshape(h, w, 4)[..., 3] / 255.0
  rl.unload_image(image)

  yy, xx = np.mgrid[:h, :w]
  r = np.hypot(((xx + 0.5) / w - center[0]) / radius[0], ((yy + 0.5) / h - center[1]) / radius[1])
  rgba = np.full((h, w, 4), 255, dtype=np.uint8)
  rgba[..., 3] = np.clip(alpha * 128 * (1 - r), 0, 255)

  texture = rl.load_texture_from_image(rl.Image(None, w, h, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8))
  rl.update_texture(texture, np.ascontiguousarray(rgba))
  rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  texture.width, texture.height = size, size
  return texture
