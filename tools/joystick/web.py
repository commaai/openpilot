#!/usr/bin/env python3
import asyncio
import sys
import time
import threading
from flask import Flask
from struct import unpack

sys.path.append('/data/openpilot/third_party/websockets/src')
from websockets import serve

import cereal.messaging as messaging

app = Flask(__name__)
pm = messaging.PubMaster(['testJoystick'])

index = """
<html>
<head>
<script src="https://github.com/bobboteck/JoyStick/releases/download/v1.1.6/joy.min.js"></script>
</head>
<body>
<div id="joyDiv" style="width:100%;height:100%"></div>
<script type="text/javascript">
// Set up gamepad handlers
let gamepad = null;
function ws_gen() {
  var ws_ = new WebSocket('ws://' + location.hostname + ':5001');
  ws_.onclose = e => setTimeout(() => { console.log("reconnecting"); ws = ws_gen(); }, 50);
  return ws_;
}
var ws = ws_gen();

window.addEventListener("gamepadconnected", function(e) {
  gamepad = e.gamepad;
});
window.addEventListener("gamepaddisconnected", function(e) {
  gamepad = null;
});
// Create JoyStick object into the DIV 'joyDiv'
var joy = new JoyStick('joyDiv');
var axes = new Float32Array([0.0, 0.0]);
setInterval(function(){
  var x = -joy.GetX()/100;
  var y = joy.GetY()/100;
  if (x === 0 && y === 0 && gamepad !== null) {
    let gamepadstate = navigator.getGamepads()[gamepad.index];
    x = -gamepadstate.axes[0];
    y = -gamepadstate.axes[1];
  }
  axes[0] = x;
  axes[1] = y;
  ws.send(axes);
}, 50);
</script>
"""

@app.route("/")
def hello_world():
  return index

last_send_time = time.monotonic()
ws_print_every_xth_message = 8
ws_print_counter = 0
async def handle_message(ws):
  async for message in ws:
    global last_send_time
    global ws_print_counter
    global ws_print_every_xth_message
    if not len(message) == 8:
      pass
    else:
      x,y = unpack('ff', message)
      x = max(-1, min(1, x))
      y = max(-1, min(1, y))
      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [y,x]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
      last_lst = last_send_time
      last_send_time = time.monotonic()
      ws_delta_time_ms = (last_send_time - last_lst) * 1000.
      if (ws_print_counter % ws_print_every_xth_message) == 0:
        print((
          "[testJoystick] "
          "Δt: " f"{ws_delta_time_ms:1.0f}" ", "
          "axes: {"
            "x: " + f"{x:.2f}" + ", "
            "y: " + f"{y:.2f}"
          "}"
        ))
      ws_print_counter += 1

async def maine():
  async with serve(handle_message, "0.0.0.0", 5001):
    await asyncio.Future() # run forever
def websocket_thread():
  asyncio.run(maine())

def handle_timeout():
  while 1:
    this_time = time.monotonic()
    if (last_send_time+0.5) < this_time:
      #print("timeout, no web in %.2f s" % (this_time-last_send_time))
      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [0,0]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
    time.sleep(0.1)

def main():
  threading.Thread(target=handle_timeout, daemon=True).start()
  threading.Thread(target=websocket_thread, daemon=True).start()
  app.run(host="0.0.0.0", port="5000")

if __name__ == '__main__':
  main()
