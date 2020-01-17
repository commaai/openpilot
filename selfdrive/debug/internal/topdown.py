#!/usr/bin/env python3
import sys
import math
import pygame
import pyproj

import zmq
import cereal.messaging as messaging
from cereal.services import service_list
import numpy as np

METER = 25
YSCALE = 1

def to_grid(pt):
  return (int(round(pt[0] * METER + 100)), int(round(pt[1] * METER * YSCALE + 500)))

def gps_latlong_to_meters(gps_values, zero):
  inProj = pyproj.Proj(init='epsg:4326')
  outProj = pyproj.Proj(("+proj=tmerc +lat_0={:f} +lon_0={:f} +units=m"
                         " +k=1. +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +no_defs"
                         "+towgs84=-90.7,-106.1,-119.2,4.09,0.218,-1.05,1.37").format(*zero))
  gps_x, gps_y = pyproj.transform(inProj, outProj, gps_values[1], gps_values[0])
  return gps_x, gps_y

def rot(hrad):
  return [[math.cos(hrad), -math.sin(hrad)],
          [math.sin(hrad),  math.cos(hrad)]]

class Car():
  CAR_WIDTH = 2.0
  CAR_LENGTH = 4.5

  def __init__(self, c):
    self.car = pygame.Surface((METER*self.CAR_LENGTH*YSCALE, METER*self.CAR_LENGTH))
    self.car.set_alpha(64)
    self.car.fill((0,0,0))
    self.car.set_colorkey((0,0,0))
    pygame.draw.rect(self.car, c, (METER*1.25*YSCALE, 0, METER*self.CAR_WIDTH*YSCALE, METER*self.CAR_LENGTH), 1)

    self.x = 0.0
    self.y = 0.0
    self.heading = 0.0

  def from_car_frame(self, pts):
    ret = []
    for x, y in pts:
      rx, ry = np.dot(rot(math.radians(self.heading)), [x,y])
      ret.append((self.x + rx, self.y + ry))
    return ret

  def draw(self, screen):
    cars = pygame.transform.rotate(self.car, 90-self.heading)
    pt = (self.x - self.CAR_LENGTH/2, self.y - self.CAR_LENGTH/2)
    screen.blit(cars, to_grid(pt))


def ui_thread(addr="127.0.0.1"):
  #from selfdrive.radar.nidec.interface import RadarInterface
  #RI = RadarInterface()

  pygame.display.set_caption("comma top down UI")
  size = (1920,1000)
  screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

  liveLocation = messaging.sub_sock('liveLocation', addr=addr)

  #model = messaging.sub_sock('testModel', addr=addr)
  model = messaging.sub_sock('model', addr=addr)

  plan = messaging.sub_sock('plan', addr=addr)
  frame = messaging.sub_sock('frame', addr=addr)
  liveTracks = messaging.sub_sock('liveTracks', addr=addr)

  car = Car((255,0,255))

  base = None

  lb = []

  ts_map = {}

  while 1:
    lloc = messaging.recv_sock(liveLocation, wait=True)
    lloc_ts = lloc.logMonoTime
    lloc = lloc.liveLocation

    # 50 ms of lag
    lb.append(lloc)
    if len(lb) < 2:
      continue
    lb = lb[-1:]

    lloc = lb[0]

    # spacebar reset
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        base = None

    # offscreen reset
    rp = to_grid((car.x, car.y))
    if rp[0] > (size[0] - 100) or rp[1] > (size[1] - 100) or rp[0] < 0 or rp[1] < 100:
      base = None


    if base == None:
      screen.fill((10,10,10))
      base = lloc

    # transform pt into local
    pt = gps_latlong_to_meters((lloc.lat, lloc.lon), (base.lat, base.lon))
    hrad = math.radians(270+base.heading)
    pt = np.dot(rot(hrad), pt)

    car.x, car.y = pt[0], -pt[1]
    car.heading = lloc.heading - base.heading

    #car.draw(screen)
    pygame.draw.circle(screen, (192,64,192,128), to_grid((car.x, car.y)), 4)

    """
    lt = messaging.recv_sock(liveTracks, wait=False)
    if lt is not None:
      for track in lt.liveTracks:
        pt = car.from_car_frame([[track.dRel, -track.yRel]])[0]
        if track.stationary:
          pygame.draw.circle(screen, (192,128,32,64), to_grid(pt), 1)
    """


    """
    rr = RI.update()
    for pt in rr.points:
      cpt = car.from_car_frame([[pt.dRel + 2.7, -pt.yRel]])[0]
      if (pt.vRel + lloc.speed) < 1.0:
        pygame.draw.circle(screen, (192,128,32,64), to_grid(cpt), 1)
    """


    for f in messaging.drain_sock(frame):
      ts_map[f.frame.frameId] = f.frame.timestampEof

    def draw_model_data(mm, c):
      pts = car.from_car_frame(zip(np.arange(0.0, 50.0), -np.array(mm)))
      lt = 255
      for pt in pts:
        screen.set_at(to_grid(pt), (c[0]*lt,c[1]*lt,c[2]*lt,lt))
        lt -= 2
      #pygame.draw.lines(screen, (c[0]*lt,c[1]*lt,c[2]*lt,lt), False, map(to_grid, pts), 1)

    md = messaging.recv_sock(model, wait=False)
    if md:
      if md.model.frameId in ts_map:
        f_ts = ts_map[md.model.frameId]
        print((lloc_ts - f_ts) * 1e-6,"ms")

      #draw_model_data(md.model.path.points, (1,0,0))
      if md.model.leftLane.prob > 0.3:
        draw_model_data(md.model.leftLane.points, (0,1,0))
      if md.model.rightLane.prob > 0.3:
        draw_model_data(md.model.rightLane.points, (0,1,0))
      #if md.model.leftLane.prob > 0.3 and md.model.rightLane.prob > 0.3:
      #  draw_model_data([(x+y)/2 for x,y in zip(md.model.leftLane.points, md.model.rightLane.points)], (1,1,0))

    tplan = messaging.recv_sock(plan, wait=False)
    if tplan:
      pts = np.polyval(tplan.plan.dPoly, np.arange(0.0, 50.0))
      draw_model_data(pts, (1,1,1))

    pygame.display.flip()

if __name__ == "__main__":
  if len(sys.argv) > 1:
    ui_thread(sys.argv[1])
  else:
    ui_thread()

