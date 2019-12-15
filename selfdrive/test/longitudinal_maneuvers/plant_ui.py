#!/usr/bin/env python3
import pygame # pylint: disable=import-error
from selfdrive.test.longitudinal_maneuvers.plant import Plant
from selfdrive.car.honda.values import CruiseButtons
import numpy as np
import selfdrive.messaging as messaging
import math

CAR_WIDTH = 2.0
CAR_LENGTH = 4.5

METER = 8

def rot_center(image, angle):
  """rotate an image while keeping its center and size"""
  orig_rect = image.get_rect()
  rot_image = pygame.transform.rotate(image, angle)
  rot_rect = orig_rect.copy()
  rot_rect.center = rot_image.get_rect().center
  rot_image = rot_image.subsurface(rot_rect).copy()
  return rot_image

def car_w_color(c):
  car = pygame.Surface((METER*CAR_LENGTH, METER*CAR_LENGTH))  # pylint: disable=too-many-function-args
  car.set_alpha(0)
  car.fill((10,10,10))
  car.set_alpha(128)
  pygame.draw.rect(car, c, (METER*1.25, 0, METER*CAR_WIDTH, METER*CAR_LENGTH), 1)
  return car

if __name__ == "__main__":
  pygame.init()
  display = pygame.display.set_mode((1000, 1000))
  pygame.display.set_caption('Plant UI')

  car = car_w_color((255,0,255))
  leadcar = car_w_color((255,0,0))

  carx, cary, heading = 10.0, 50.0, 0.0

  plant = Plant(100, distance_lead = 40.0)

  control_offset = 2.0
  control_pts = list(zip(np.arange(0, 100.0, 10.0), [50.0 + control_offset]*10))

  def pt_to_car(pt):
    x,y = pt
    x -= carx
    y -= cary
    rx = x * math.cos(-heading) + y * -math.sin(-heading)
    ry = x * math.sin(-heading) + y * math.cos(-heading)
    return rx, ry

  def pt_from_car(pt):
    x,y = pt
    rx = x * math.cos(heading) + y * -math.sin(heading)
    ry = x * math.sin(heading) + y * math.cos(heading)
    rx += carx
    ry += cary
    return rx, ry

  while 1:
    if plant.rk.frame%100 >= 20 and plant.rk.frame%100 <= 25:
      cruise_buttons = CruiseButtons.RES_ACCEL
    else:
      cruise_buttons = 0

    md = messaging.new_message()
    md.init('model')
    md.model.frameId = 0
    for x in [md.model.path, md.model.leftLane, md.model.rightLane]:
      x.points = [0.0]*50
      x.prob = 0.0
      x.std = 1.0

    car_pts = [pt_to_car(pt) for pt in control_pts]

    print(car_pts)

    car_poly = np.polyfit([x[0] for x in car_pts], [x[1] for x in car_pts], 3)
    md.model.path.points = np.polyval(car_poly, np.arange(0, 50)).tolist()
    md.model.path.prob = 1.0
    Plant.model.send(md.to_bytes())

    plant.step(cruise_buttons = cruise_buttons, v_lead = 2.0, publish_model = False)

    display.fill((10,10,10))

    carx += plant.speed * plant.ts * math.cos(heading)
    cary += plant.speed * plant.ts * math.sin(heading)

    # positive steering angle = steering right
    print(plant.angle_steer)
    heading += plant.angle_steer * plant.ts
    print(heading)

    # draw my car
    display.blit(pygame.transform.rotate(car, 90-math.degrees(heading)), (carx*METER, cary*METER))

    # draw control pts
    for x,y in control_pts:
      pygame.draw.circle(display, (255,255,0), (int(x * METER),int(y * METER)), 2)

    # draw path
    path_pts = zip(np.arange(0, 50), md.model.path.points)

    for x,y in path_pts:
      x,y = pt_from_car((x,y))
      pygame.draw.circle(display, (0,255,0), (int(x * METER),int(y * METER)), 1)

    """
    # draw lead car
    dl = (plant.distance_lead - plant.distance) + 4.5
    lx = carx + dl * math.cos(heading)
    ly = cary + dl * math.sin(heading)

    display.blit(pygame.transform.rotate(leadcar, 90-math.degrees(heading)), (lx*METER, ly*METER))
    """

    pygame.display.flip()
