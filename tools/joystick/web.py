#!/usr/bin/env python3
import time
import os
from flask import Flask, render_template, Response

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from system.camerad.snapshot.snapshot import  extract_image
from flask_socketio import SocketIO

IMG_H, IMG_W = 540, 960

app = Flask(__name__)
pm = messaging.PubMaster(['testJoystick'])
socketio = SocketIO(app, async_mode='threading')

@app.route("/")
def hello_world():
  return render_template('index.html')


#camera.py
# import the necessary packages
import cv2
import numpy as np

class VideoCamera(object):
  def __init__(self):
    self.vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)
    self.cnt = 0


  def __del__(self):
    pass

  def get_frame(self):
    if os.environ.get('FAKE_CAMERA') == '1':
      frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
      frame[self.cnt:self.cnt+10, :, :] = 255
      self.cnt = (self.cnt + 10)%IMG_H
      _, jpeg = cv2.imencode('.jpg', frame)
      return jpeg.tobytes()
 
    if not self.vipc_client.is_connected():
      self.vipc_client.connect(True)
    yuv_img_raw = self.vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.any():
      frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)
      time.sleep(0.05)
    else:
      #imgff = np.frombuffer(yuv_img_raw, dtype=np.uint8)

      #imgff = imgff[:3493536].reshape((self.vipc_client.height * 3 // 2, self.vipc_client.width))
      #frame = cv2.cvtColor(imgff, cv2.COLOR_YUV2BGR_NV12)
      c = self.vipc_client
      frame = extract_image(c.recv(), c.width, c.height, c.stride, c.uv_offset)
      #frame = np.zeros((IMG_H, IMG_W, 3), np.uint8)

      frame = cv2.resize(frame, (IMG_W, IMG_H))

    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()
      

def gen():
  camera = VideoCamera()
  while True:
    #get camera frame
    frame = camera.get_frame()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
  return Response(gen(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')


last_send_time = time.monotonic()
@socketio.on('control_command')
def hand_control_command(data):
  print(data)
  x = data['x']
  y = data['y']
  global last_send_time
  x,y = float(x), float(y)
  x = max(-1, min(1, x))
  y = max(-1, min(1, y))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [x,y]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)
  last_send_time = time.monotonic()

def handle_timeout():
  while 1:
    this_time = time.monotonic()
    if (last_send_time+0.5) < this_time:
      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [0,0]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
    time.sleep(0.1)

import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

 
audio1 = pyaudio.PyAudio()

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def gen_audio():
  while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    data = np.frombuffer(data, dtype=np.int16)
    socketio.emit('stream', data.tolist())
    time.sleep(0.0001)


def main():
  #threading.Thread(target=handle_timeout, daemon=True).start()
  socketio.start_background_task(gen)
  socketio.start_background_task(target=gen_audio)
  socketio.run(app, host="0.0.0.0")


if __name__ == '__main__':
  main()