import pyray as rl


def get_roundness(rect: rl.Rectangle, border_radius: int) -> float:
  """Calculate the roundness of a rectangle based on its width and height, given a border radius value in pixels.
  This is used to standardize the roundness across rectangle with different sizes, since `draw_rectangle_rounded` doesn't use pixels.
  """
  return border_radius / (min(rect.width, rect.height) / 2)
