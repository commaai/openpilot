import numpy as np
import cv2  # pylint: disable=import-error

def rot_matrix(roll, pitch, yaw):
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rr = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  rp = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  ry = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return ry.dot(rp.dot(rr))

def draw_pose(img, pose, loc, W=160, H=320, xyoffset=(0, 0), faceprob=0):
  rcmat = np.zeros((3, 4))
  rcmat[:, :3] = rot_matrix(*pose[0:3]) * 0.5
  rcmat[0, 3] = (loc[0]+0.5) * W
  rcmat[1, 3] = (loc[1]+0.5) * H
  rcmat[2, 3] = 1.0
  # draw nose
  p1 = np.dot(rcmat, [0, 0, 0, 1])[0:2]
  p2 = np.dot(rcmat, [0, 0, 100, 1])[0:2]
  tr = tuple([int(round(x + xyoffset[i])) for i, x in enumerate(p1)])
  pr = tuple([int(round(x + xyoffset[i])) for i, x in enumerate(p2)])
  if faceprob > 0.4:
    color = (255, 255, 0)
    cv2.line(img, tr, pr, color=(255, 255, 0), thickness=3)
  else:
    color = (64, 64, 64)
  cv2.circle(img, tr, 7, color=color)
