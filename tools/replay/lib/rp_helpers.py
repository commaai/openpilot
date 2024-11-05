import numpy as np
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

# Color palette used for rerun AnnotationContext
rerunColorPalette = [(96, "red", (255, 0, 0)),
                     (100, "pink", (255, 36, 0)),
                     (124, "yellow", (255, 255, 0)),
                     (230, "vibrantpink", (255, 36, 170)),
                     (240, "orange", (255, 146, 0)),
                     (255, "white", (255, 255, 255)),
                     (110, "carColor", (255,0,127)),
                     (0, "background", (0, 0, 0))]


class UIParams:
  lidar_x, lidar_y, lidar_zoom = 384, 960, 6
  lidar_car_x, lidar_car_y = lidar_x / 2., lidar_y / 1.1
  car_hwidth = 1.7272 / 2 * lidar_zoom
  car_front = 2.6924 * lidar_zoom
  car_back = 1.8796 * lidar_zoom
  car_color = rerunColorPalette[6][0]
UP = UIParams


def to_topdown_pt(y, x):
  px, py = x * UP.lidar_zoom + UP.lidar_car_x, -y * UP.lidar_zoom + UP.lidar_car_y
  if px > 0 and py > 0 and px < UP.lidar_x and py < UP.lidar_y:
    return int(px), int(py)
  return -1, -1


def draw_path(path, lid_overlay, lid_color=None):
  x, y = np.asarray(path.x), np.asarray(path.y)
  # draw lidar path point on lidar
  if lid_color is not None and lid_overlay is not None:
    for i in range(len(x)):
      px, py = to_topdown_pt(x[i], y[i])
      if px != -1:
        lid_overlay[px, py] = lid_color


def plot_model(m, lid_overlay):
  if lid_overlay is None:
    return
  for lead in m.leadsV3:
    if lead.prob < 0.5:
      continue
    x, y = lead.x[0], lead.y[0]
    x_std = lead.xStd[0]
    x -= RADAR_TO_CAMERA
    _, py_top = to_topdown_pt(x + x_std, y)
    px, py_bottom = to_topdown_pt(x - x_std, y)
    lid_overlay[int(round(px - 4)):int(round(px + 4)), py_top:py_bottom] = rerunColorPalette[2][0]

  for path in m.laneLines:
    draw_path(path, lid_overlay, rerunColorPalette[2][0])
  for edge in m.roadEdges:
    draw_path(edge, lid_overlay, rerunColorPalette[0][0])
  draw_path(m.position, lid_overlay, rerunColorPalette[0][0])


def plot_lead(rs, lid_overlay):
  for lead in [rs.leadOne, rs.leadTwo]:
    if not lead.status:
      continue
    x = lead.dRel
    px_left, py = to_topdown_pt(x, -10)
    px_right, _ = to_topdown_pt(x, 10)
    lid_overlay[px_left:px_right, py] = rerunColorPalette[0][0]


def update_radar_points(lt, lid_overlay):
  ar_pts = []
  if lt is not None:
    ar_pts = {}
    for track in lt:
      ar_pts[track.trackId] = [track.dRel, track.yRel, track.vRel, track.aRel, track.oncoming, track.stationary]
  for ids, pt in ar_pts.items():
    # negative here since radar is left positive
    px, py = to_topdown_pt(pt[0], -pt[1])
    if px != -1:
      if pt[-1]:
        color = rerunColorPalette[4][0]
      elif pt[-2]:
        color = rerunColorPalette[3][0]
      else:
        color = rerunColorPalette[5][0]
      if int(ids) == 1:
        lid_overlay[px - 2:px + 2, py - 10:py + 10] = rerunColorPalette[1][0]
      else:
        lid_overlay[px - 2:px + 2, py - 2:py + 2] = color


def get_blank_lid_overlay(UP):
  lid_overlay = np.zeros((UP.lidar_x, UP.lidar_y), 'uint8')
  # Draw the car.
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y -
                                                      UP.car_front))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y +
                                                      UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x + UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  return lid_overlay
