#!/usr/bin/env python3
from extras.transformations.camera import transform_img, eon_intrinsics
from extras.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
from tools.lib.framereader import FrameReader, MP4FrameReader, MKVFrameReader 
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from modelparser import parser
import cv2
import sys
camerafile = sys.argv[1]
supercombo = load_model('../../models/supercombo.keras')

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

fr = FrameReader(camerafile)
cap = cv2.VideoCapture(camerafile)

imgs = []
raw_imgs = []
for i in tqdm(range(1000)):
  imgs.append(fr.get(i, pix_fmt='yuv420p')[0].reshape((874*3//2, 1164)))
  ret, frame = cap.read()
  raw_imgs.append(frame)

raw_imgs = np.array(raw_imgs)
print(raw_imgs.shape)
def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


state = np.zeros((1,512))
desire = np.zeros((1,8))

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
  outs = supercombo.predict(inputs)
  parsed = parser(outs)
  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]

  # Show raw camera image
  cv2.imshow("modeld", raw_imgs[i])
  # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")
  # lll = left lane line
  plt.scatter(parsed["lll"], range(0,192), c="b")
  # rll = right lane line
  plt.scatter(parsed["rll"], range(0, 192), c="r")
  # path = path cool isn't it ?
  plt.scatter(parsed["path"], range(0, 192), c="g")
  print(np.array(pose[0,:3]).shape)
  #plt.scatter(pose[0,:3], range(3), c="y")
  
  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.05)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

plt.show()
  


