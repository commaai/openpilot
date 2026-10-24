import pyray as rl

SIDE_PANEL_WIDTH = 60

# [experiment] night mode: multiplied over the onroad view, indexed by NightModeTint
NIGHT_TINTS: list[tuple[str, rl.Color]] = [
  ("off", rl.Color(255, 255, 255, 255)),
  ("dim", rl.Color(150, 150, 150, 255)),
  ("warm", rl.Color(255, 180, 110, 255)),
  ("no blue", rl.Color(255, 255, 0, 255)),
  ("amber", rl.Color(255, 130, 0, 255)),
  ("dim amber", rl.Color(150, 75, 0, 255)),
  ("red", rl.Color(255, 0, 0, 255)),
  ("dim red", rl.Color(150, 0, 0, 255)),
]


def blend_colors(a: rl.Color, b: rl.Color, f: float) -> rl.Color:
  h0, s0, v0 = (hsv0 := rl.color_to_hsv(a)).x, hsv0.y, hsv0.z
  h1, s1, v1 = (hsv1 := rl.color_to_hsv(b)).x, hsv1.y, hsv1.z
  dh = ((h1 - h0 + 180) % 360) - 180  # shortest hue delta
  return rl.color_from_hsv((h0 + f * dh) % 360,
                           s0 + f * (s1 - s0),
                           v0 + f * (v1 - v0))
