import platform
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame

from tools.lib.lazy_property import lazy_property
from selfdrive.config import UIParams as UP
from selfdrive.config import RADAR_TO_CAMERA
from selfdrive.controls.lib.lane_planner import (compute_path_pinv,
                                                 model_polyfit)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_PATH_X = np.arange(192.)
_PATH_XD = np.arange(192.)
_PATH_PINV = compute_path_pinv(50)
#_BB_OFFSET = 290, 332
_BB_OFFSET = 0,0
_BB_SCALE = 1164/640.
_BB_TO_FULL_FRAME = np.asarray([
    [_BB_SCALE, 0., _BB_OFFSET[0]],
    [0., _BB_SCALE, _BB_OFFSET[1]],
    [0., 0.,   1.]])
_FULL_FRAME_TO_BB = np.linalg.inv(_BB_TO_FULL_FRAME)

METER_WIDTH = 20

ModelUIData = namedtuple("ModelUIData", ["cpath", "lpath", "rpath", "lead", "lead_future"])

_COLOR_CACHE = {}
def find_color(lidar_surface, color):
  if color in _COLOR_CACHE:
    return _COLOR_CACHE[color]
  tcolor = 0
  ret = 255
  for x in lidar_surface.get_palette():
    #print tcolor, x
    if x[0:3] == color:
      ret = tcolor
      break
    tcolor += 1
  _COLOR_CACHE[color] = ret
  return ret

def warp_points(pt_s, warp_matrix):
  # pt_s are the source points, nxm array.
  pt_d = np.dot(warp_matrix[:, :-1], pt_s.T) + warp_matrix[:, -1, None]

  # Divide by last dimension for representation in image space.
  return (pt_d[:-1, :] / pt_d[-1, :]).T

def to_lid_pt(y, x):
  px, py = -x * UP.lidar_zoom + UP.lidar_car_x, -y * UP.lidar_zoom + UP.lidar_car_y
  if px > 0 and py > 0 and px < UP.lidar_x and py < UP.lidar_y:
    return int(px), int(py)
  return -1, -1


def draw_path(y, x, color, img, calibration, top_down, lid_color=None):
  # TODO: Remove big box.
  uv_model_real = warp_points(np.column_stack((x, y)), calibration.car_to_model)
  uv_model = np.round(uv_model_real).astype(int)

  uv_model_dots = uv_model[np.logical_and.reduce((np.all(  # pylint: disable=no-member
    uv_model > 0, axis=1), uv_model[:, 0] < img.shape[1] - 1, uv_model[:, 1] <
                                                  img.shape[0] - 1))]

  for i, j  in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
    img[uv_model_dots[:, 1] + i, uv_model_dots[:, 0] + j] = color

  # draw lidar path point on lidar
  # find color in 8 bit
  if lid_color is not None and top_down is not None:
    tcolor = find_color(top_down[0], lid_color)
    for i in range(len(x)):
      px, py = to_lid_pt(x[i], y[i])
      if px != -1:
        top_down[1][px, py] = tcolor

def draw_steer_path(speed_ms, curvature, color, img,
                    calibration, top_down, VM, lid_color=None):
  path_x = np.arange(101.)
  path_y =  np.multiply(path_x, np.tan(np.arcsin(np.clip(path_x * curvature, -0.999, 0.999)) / 2.))

  draw_path(path_y, path_x, color, img, calibration, top_down, lid_color)

def draw_lead_car(closest, top_down):
  if closest != None:
    closest_y = int(round(UP.lidar_car_y - closest * UP.lidar_zoom))
    if closest_y > 0:
      top_down[1][int(round(UP.lidar_car_x - METER_WIDTH * 2)):int(
        round(UP.lidar_car_x + METER_WIDTH * 2)), closest_y] = find_color(
          top_down[0], (255, 0, 0))

def draw_lead_on(img, closest_x_m, closest_y_m, calibration, color, sz=10, img_offset=(0, 0)):
  uv = warp_points(np.asarray([closest_x_m, closest_y_m]), calibration.car_to_bb)[0]
  u, v = int(uv[0] + img_offset[0]), int(uv[1] + img_offset[1])
  if u > 0 and u < 640 and v > 0 and v < 480 - 5:
    img[v - 5 - sz:v - 5 + sz, u] = color
    img[v - 5, u - sz:u + sz] = color
  return u, v


if platform.system() != 'Darwin':
  matplotlib.use('QT4Agg')


def init_plots(arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=False):
  color_palette = { "r": (1,0,0),
                    "g": (0,1,0),
                    "b": (0,0,1),
                    "k": (0,0,0),
                    "y": (1,1,0),
                    "p": (0,1,1),
                    "m": (1,0,1) }

  if bigplots == True:
    fig = plt.figure(figsize=(6.4, 7.0))
  elif bigplots == False:
    fig = plt.figure()
  else:
    fig = plt.figure(figsize=bigplots)

  fig.set_facecolor((0.2,0.2,0.2))

  axs = []
  for pn in range(len(plot_ylims)):
    ax = fig.add_subplot(len(plot_ylims),1,len(axs)+1)
    ax.set_xlim(plot_xlims[pn][0], plot_xlims[pn][1])
    ax.set_ylim(plot_ylims[pn][0], plot_ylims[pn][1])
    ax.patch.set_facecolor((0.4, 0.4, 0.4))
    axs.append(ax)

  plots = [] ;idxs = [] ;plot_select = []
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


def draw_mpc(liveMpc, top_down):
  mpc_color = find_color(top_down[0], (0, 255, 0))
  for p in zip(liveMpc.x, liveMpc.y):
    px, py = to_lid_pt(*p)
    top_down[1][px, py] = mpc_color



class CalibrationTransformsForWarpMatrix(object):
  def __init__(self, model_to_full_frame, K, E):
    self._model_to_full_frame = model_to_full_frame
    self._K = K
    self._E = E

  @property
  def model_to_bb(self):
    return _FULL_FRAME_TO_BB.dot(self._model_to_full_frame)

  @lazy_property
  def model_to_full_frame(self):
    return self._model_to_full_frame

  @lazy_property
  def car_to_model(self):
    return np.linalg.inv(self._model_to_full_frame).dot(self._K).dot(
      self._E[:, [0, 1, 3]])

  @lazy_property
  def car_to_bb(self):
    return _BB_TO_FULL_FRAME.dot(self._K).dot(self._E[:, [0, 1, 3]])


def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()

def draw_var(y, x, var, color, img, calibration, top_down):
  # otherwise drawing gets stupid
  var = max(1e-1, min(var, 0.7))

  varcolor = tuple(np.array(color)*0.5)
  draw_path(y - var, x, varcolor, img, calibration, top_down)
  draw_path(y + var, x, varcolor, img, calibration, top_down)


class ModelPoly(object):
  def __init__(self, model_path):
    if len(model_path.points) == 0 and len(model_path.poly) == 0:
      self.valid = False
      return

    if len(model_path.poly):
      self.poly = np.array(model_path.poly)
    else:
      self.poly = model_polyfit(model_path.points, _PATH_PINV)

    self.prob = model_path.prob
    self.std = model_path.std
    self.y = np.polyval(self.poly, _PATH_XD)
    self.valid = True

def extract_model_data(md):
  return ModelUIData(
    cpath=ModelPoly(md.path),
    lpath=ModelPoly(md.leftLane),
    rpath=ModelPoly(md.rightLane),
    lead=md.lead,
    lead_future=md.leadFuture,
    )

def plot_model(m, VM, v_ego, curvature, imgw, calibration, top_down, d_poly, top_down_color=216):
  if calibration is None or top_down is None:
    return

  for lead in [m.lead, m.lead_future]:
    if lead.prob < 0.5:
      continue

    lead_dist_from_radar = lead.dist - RADAR_TO_CAMERA
    _, py_top = to_lid_pt(lead_dist_from_radar + lead.std, lead.relY)
    px, py_bottom = to_lid_pt(lead_dist_from_radar - lead.std, lead.relY)
    top_down[1][int(round(px - 4)):int(round(px + 4)), py_top:py_bottom] = top_down_color

  color = (0, int(255 * m.lpath.prob), 0)
  for path in [m.cpath, m.lpath, m.rpath]:
    if path.valid:
      draw_path(path.y, _PATH_XD, color, imgw, calibration, top_down, YELLOW)
      draw_var(path.y, _PATH_XD, path.std, color, imgw, calibration, top_down)

  if d_poly is not None:
    dpath_y = np.polyval(d_poly, _PATH_X)
    draw_path(dpath_y, _PATH_X, RED, imgw, calibration, top_down, RED)

  # draw user path from curvature
  draw_steer_path(v_ego, curvature, BLUE, imgw, calibration, top_down, VM, BLUE)


def maybe_update_radar_points(lt, lid_overlay):
  ar_pts = []
  if lt is not None:
    ar_pts = {}
    for track in lt:
      ar_pts[track.trackId] = [track.dRel, track.yRel, track.vRel, track.aRel, track.oncoming, track.stationary]
  for ids, pt in ar_pts.items():
    px, py = to_lid_pt(pt[0], pt[1])
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
