import math
import numpy as np
import raylib_nb as rl

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


def gaussian_blur(alpha: np.ndarray, sigma: float) -> np.ndarray:
  """Blurs a coverage mask, padded by 3 sigma so nothing is clipped"""
  pad = math.ceil(3 * sigma)
  out = np.pad(alpha, pad)
  x = np.arange(-pad, pad + 1)
  kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
  kernel /= kernel.sum()
  for axis in (0, 1):
    out = np.apply_along_axis(np.convolve, axis, out, kernel, mode='same')
  return out


def alpha_texture(alpha: np.ndarray, px: float = 1.0, color: tuple[int, int, int] = (255, 255, 255)) -> rl.Texture:
  """A flat color texture from a coverage mask rendered at px pixels per point"""
  rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
  rgba[..., :3] = color
  rgba[..., 3] = np.clip(alpha * 255, 0, 255)
  return texture_from_rgba(rgba, px)


def texture_from_rgba(rgba: np.ndarray, px: float = 1.0) -> rl.Texture:
  texture = rl.load_texture_from_image(rl.Image(None, rgba.shape[1], rgba.shape[0], 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8))
  rl.update_texture(texture, np.ascontiguousarray(rgba))
  rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  texture.width, texture.height = round(rgba.shape[1] / px), round(rgba.shape[0] / px)
  return texture
