from tools.streamer.tasks import Camerad
import struct
import numpy as np
import zmq
import cv2
from selfdrive.test.helpers import set_params_enabled
from common.params import Params
#import cereal.messaging as messaging


def streamer():
  set_params_enabled()
  params = Params()
  #msg = messaging.new_message('liveCalibration')
  #msg.liveCalibration.validBlocks = 20
  #msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
  #params.put("CalibrationParams", msg.to_bytes())
  params.remove("CalibrationParams")
  params.put_bool("WideCameraOnly", False)
  context = zmq.Context().instance()
  socket = context.socket(zmq.PULL)

  socket.setsockopt(zmq.CONFLATE, 1)
  socket.set_hwm(5)
  socket.connect("ipc:///tmp/metadrive_window") #configured in gamerunner.py
  c_W = 1928
  c_H = 1208
  _camerad = Camerad(c_W, c_H)
  while True:

    image_buffer_msg = socket.recv()
    # read the first 32bit int as W
    # second as H to handle different resolutions
    W, H = struct.unpack('ii', image_buffer_msg[:8]) 
    # read the rest of the message as the image buffer
    image_buffer = image_buffer_msg[8:]
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((H, W, 4))
    
    image = image[:, :, :3] # remove the alpha channel
    image = np.flip(image, 0) # flip the image with numpy
    cropped_img = image[H//5:H-(H//5), W//5:W-(W//5)]

    road_img = cv2.resize(cropped_img, (c_W, c_H))
    wide_img = cv2.resize(image, (c_W, c_H))

    _camerad.cam_callback_wide_road(wide_img)
    _camerad.cam_callback_road(road_img)

    #cv2.imshow('wide_img', wide_img)
    #cv2.imshow("cropped", cropped_img)
    #cv2.imshow('road_img', road_img)
    #cv2.waitKey(1)


def main():
  streamer()

if __name__ == "__main__":
  main()