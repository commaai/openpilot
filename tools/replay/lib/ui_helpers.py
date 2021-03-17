import itertools
from typing import Any, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame  # pylint: disable=import-error

from common.transformations.camera import (eon_f_frame_size, eon_f_focal_length,
                                           tici_f_frame_size, tici_f_focal_length,
                                           get_view_frame_from_calib_frame)
from selfdrive.config import UIParams as UP
from selfdrive.config import RADAR_TO_CAMERA


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_FULL_FRAME_SIZE = {
}

_BB_TO_FULL_FRAME = {}
_FULL_FRAME_TO_BB = {}
_INTRINSICS = {}
cams = [(eon_f_frame_size[0], eon_f_frame_size[1], eon_f_focal_length),
        (tici_f_frame_size[0], tici_f_frame_size[1], tici_f_focal_length)]
for width, height, focal in cams:
  sz = width * height
  _BB_SCALE = width / 640.
  _BB_TO_FULL_FRAME[sz] = np.asarray([
      [_BB_SCALE, 0., 0.],
      [0., _BB_SCALE, 0.],
      [0., 0., 1.]])
  _FULL_FRAME_TO_BB[sz] = np.linalg.inv(_BB_TO_FULL_FRAME[sz])
  _FULL_FRAME_SIZE[sz] = (width, height)
  _INTRINSICS[sz] = np.array([
    [focal, 0., width / 2.],
    [0., focal, height / 2.],
    [0., 0., 1.]])


METER_WIDTH = 20

class Calibration:
  def __init__(self, num_px, rpy, intrinsic):
    self.intrinsic = intrinsic
    self.extrinsics_matrix = get_view_frame_from_calib_frame(rpy[0], rpy[1], rpy[2], 0.0)[:,:3]
    self.zoom = _BB_TO_FULL_FRAME[num_px][0, 0]

  def car_space_to_ff(self, x, y, z):
    car_space_projective = np.column_stack((x, y, z)).T

    ep = self.extrinsics_matrix.dot(car_space_projective)
    kep = self.intrinsic.dot(ep)
    return (kep[:-1, :] / kep[-1, :]).T

  def car_space_to_bb(self, x, y, z):
    pts = self.car_space_to_ff(x, y, z)
    return pts / self.zoom


_COLOR_CACHE : Dict[Tuple[int, int, int], Any] = {}
def find_color(lidar_surface, color):
  if color in _COLOR_CACHE:
    return _COLOR_CACHE[color]
  tcolor = 0
  ret = 255
  for x in lidar_surface.get_palette():
    if x[0:3] == color:
      ret = tcolor
      break
    tcolor += 1
  _COLOR_CACHE[color] = ret
  return ret


def to_topdown_pt(y, x):
  px, py = x * UP.lidar_zoom + UP.lidar_car_x, -y * UP.lidar_zoom + UP.lidar_car_y
  if px > 0 and py > 0 and px < UP.lidar_x and py < UP.lidar_y:
    return int(px), int(py)
  return -1, -1


def draw_path(path, color, img, calibration, top_down, lid_color=None, z_off=0):
  x, y, z = np.asarray(path.x), np.asarray(path.y), np.asarray(path.z) + z_off
  pts = calibration.car_space_to_bb(x, y, z)
  pts = np.round(pts).astype(int)

  # draw lidar path point on lidar
  # find color in 8 bit
  if lid_color is not None and top_down is not None:
    tcolor = find_color(top_down[0], lid_color)
    for i in range(len(x)):
      px, py = to_topdown_pt(x[i], y[i])
      if px != -1:
        top_down[1][px, py] = tcolor

  height, width = img.shape[:2]
  for x, y in pts:
    if 1 < x < width - 1 and 1 < y < height - 1:
      for a, b in itertools.permutations([-1, 0, -1], 2):
        img[y + a, x + b] = color


def init_plots(arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=False):
  color_palette = { "r": (1, 0, 0),
                    "g": (0, 1, 0),
                    "b": (0, 0, 1),
                    "k": (0, 0, 0),
                    "y": (1, 1, 0),
                    "p": (0, 1, 1),
                    "m": (1, 0, 1)}

  if bigplots:
    fig = plt.figure(figsize=(6.4, 7.0))
  else:
    fig = plt.figure()

  fig.set_facecolor((0.2, 0.2, 0.2))

  axs = []
  for pn in range(len(plot_ylims)):
    ax = fig.add_subplot(len(plot_ylims), 1, len(axs)+1)
    ax.set_xlim(plot_xlims[pn][0], plot_xlims[pn][1])
    ax.set_ylim(plot_ylims[pn][0], plot_ylims[pn][1])
    ax.patch.set_facecolor((0.4, 0.4, 0.4))
    axs.append(ax)

  plots, idxs, plot_select = [], [], []
  for i, pl_list in enumerate(plot_names):
    for j, item in enumerate(pl_list):
      plot, = axs[i].plot(arr[:, name_to_arr_idx[item]],
                          label=item,
                          color=color_palette[plot_colors[i][j]],
                          linestyle=plot_styles[i][j])
      plots.append(plot)
      idxs.append(name_to_arr_idx[item])
      plot_select.append(i)
    axs[i].set_title(", ".join("%s (%s)" % (nm, cl)
                               for (nm, cl) in zip(pl_list, plot_colors[i])), fontsize=10)
    axs[i].tick_params(axis="x", colors="white")
    axs[i].tick_params(axis="y", colors="white")
    axs[i].title.set_color("white")

    if i < len(plot_ylims) - 1:
      axs[i].set_xticks([])

  fig.canvas.draw()

  renderer = fig.canvas.get_renderer()

  if matplotlib.get_backend() == "MacOSX":
    fig.draw(renderer)

  def draw_plots(arr):
    for ax in axs:
      ax.draw_artist(ax.patch)
    for i in range(len(plots)):
      plots[i].set_ydata(arr[:, idxs[i]])
      axs[plot_select[i]].draw_artist(plots[i])

    if matplotlib.get_backend() == "QT4Agg":
      fig.canvas.update()
      fig.canvas.flush_events()

    raw_data = renderer.tostring_rgb()
    x, y = fig.canvas.get_width_height()

    # Handle 2x scaling
    if len(raw_data) == 4 * x * y * 3:
      plot_surface = pygame.image.frombuffer(raw_data, (2*x, 2*y), "RGB").convert()
      plot_surface = pygame.transform.scale(plot_surface, (x, y))
    else:
      plot_surface = pygame.image.frombuffer(raw_data, fig.canvas.get_width_height(), "RGB").convert()
    return plot_surface

  return draw_plots


def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()


def plot_model(m, img, calibration, top_down):
  if calibration is None or top_down is None:
    return

  for lead in m.leads:
    if lead.prob < 0.5:
      continue

    x, y, _, _ = lead.xyva
    x_std, _, _, _ = lead.xyvaStd
    x -= RADAR_TO_CAMERA

    _, py_top = to_topdown_pt(x + x_std, y)
    px, py_bottom = to_topdown_pt(x - x_std, y)
    top_down[1][int(round(px - 4)):int(round(px + 4)), py_top:py_bottom] = find_color(top_down[0], YELLOW)

  for path, prob, _ in zip(m.laneLines, m.laneLineProbs, m.laneLineStds):
    color = (0, int(255 * prob), 0)
    draw_path(path, color, img, calibration, top_down, YELLOW)

  for edge, std in zip(m.roadEdges, m.roadEdgeStds):
    prob = max(1 - std, 0)
    color = (int(255 * prob), 0, 0)
    draw_path(edge, color, img, calibration, top_down, RED)

  color = (255, 0, 0)
  draw_path(m.position, color, img, calibration, top_down, RED, 1.22)


def plot_lead(rs, top_down):
  for lead in [rs.leadOne, rs.leadTwo]:
    if not lead.status:
      continue

    x = lead.dRel
    px_left, py = to_topdown_pt(x, -10)
    px_right, _ = to_topdown_pt(x, 10)
    top_down[1][px_left:px_right, py] = find_color(top_down[0], RED)


def maybe_update_radar_points(lt, lid_overlay):
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
        color = 240
      elif pt[-2]:
        color = 230
      else:
        color = 255
      if int(ids) == 1:
        lid_overlay[px - 2:px + 2, py - 10:py + 10] = 100
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
