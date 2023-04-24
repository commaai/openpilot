#!/usr/bin/env python3
import time
import threading
from flask import Flask, Response, render_template

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType

IMG_H, IMG_W = 540, 960

app = Flask(__name__)
pm = messaging.PubMaster(['testJoystick'])

@app.route("/")
def hello_world():
  return render_template('index.html')


#camera.py
# import the necessary packages
import cv2
import numpy as np

class VideoCamera(object):
  def __init__(self):
    self.vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)


  def __del__(self):
    pass

  def get_frame(self):
    if not self.vipc_client.is_connected():
      self.vipc_client.connect(True)
    yuv_img_raw = self.vipc_client.recv()

    if yuv_img_raw is None or not yuv_img_raw.any():
      return np.zeros((IMG_H, IMG_W, 3), np.uint8)

    imgff = np.frombuffer(yuv_img_raw, dtype=np.uint8)
    imgff = imgff.reshape((self.vipc_client.height * 3 // 2, self.vipc_client.width))
    frame = cv2.cvtColor(imgff, cv2.COLOR_YUV2BGR_NV12)
    frame = cv2.resize(frame, (IMG_W, IMG_H))

    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()
      

def gen(camera):
  while True:
      
    #get camera frame
    frame = camera.get_frame()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
  return Response(gen(VideoCamera()),
                  mimetype='multipart/x-mixed-replace; boundary=frame')


last_send_time = time.monotonic()
@app.route("/control/<y>/<x>")
def control(x, y):
  global last_send_time
  x,y = float(x), float(y)
  x = max(-1, min(1, x))
  y = max(-1, min(1, y))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [y,x]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)
  last_send_time = time.monotonic()
  return ""

def handle_timeout():
  while 1:
    this_time = time.monotonic()
    if (last_send_time+0.5) < this_time:
      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [0,0]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
    time.sleep(0.1)

def main():
  threading.Thread(target=handle_timeout, daemon=True).start()
  app.run(host="0.0.0.0")

if __name__ == '__main__':
  main()
