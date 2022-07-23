#!/usr/bin/env python3
import time
import threading
from flask import Flask

import cereal.messaging as messaging

app = Flask(__name__)
pm = messaging.PubMaster(['testJoystick'])

index = """
<html>
<head>
<style>
* {box-sizing: border-box;}
.column {float: left;width: 50%;padding: 10px;height: 100%;text-align: center;}
.row:after {content: "";display: table;clear: both;}
@media screen and (orientation: portrait) {.column { width: 100%;height: 50%;}}
</style>
<script src="https://github.com/bobboteck/JoyStick/releases/download/v1.1.6/joy.min.js"></script>
</head>
<body>
<div class="row">
<div class="column" id="joy1Div"></div>
<div class="column" id="joy2Div"></div>
</div>
<script type="text/javascript">
// Set up gamepad handlers
let gamepad = null;
window.addEventListener("gamepadconnected", function(e) {
  gamepad = e.gamepad;
});
window.addEventListener("gamepaddisconnected", function(e) {
  gamepad = null;
});
const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)
var scr_min = Math.min(vw, vh)/2.5
// Create JoyStick object into the DIV 'joyDiv'
var joy1 = new JoyStick('joy1Div', { "height": scr_min, "width": scr_min });
var joy2 = new JoyStick('joy2Div', { "height": scr_min, "width": scr_min, "autoReturnToCenter": false });
setInterval(function(){
  var x1 = -joy1.GetX()/100;
  var y1 = joy1.GetY()/100;
  var x2 = -joy2.GetX()/100;
  var y2 = joy2.GetY()/100;
  if (x1 === 0 && y1 === 0 && gamepad !== null) {
    // Support only two axes with real gamepad for now
    let gamepadstate = navigator.getGamepads()[gamepad.index];
    x1 = -gamepadstate.axes[0];
    y1 = -gamepadstate.axes[1];
  }
  let xhr = new XMLHttpRequest();
  xhr.open("GET", "/control/"+x1+"/"+y1+"/"+x2+"/"+y2);
  xhr.send();
}, 50);
</script>
"""

@app.route("/")
def hello_world():
  return index

last_send_time = time.monotonic()
@app.route("/control/<x1>/<y1>/<x2>/<y2>")
def control(x1, y1, x2, y2):
  global last_send_time
  x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
  x1 = max(-1, min(1, x1))
  y1 = max(-1, min(1, y1))
  x2 = max(-1, min(1, x2))
  y2 = max(-1, min(1, y2))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [y1,x1,x2,y2]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)
  last_send_time = time.monotonic()
  return ""

def handle_timeout():
  while 1:
    this_time = time.monotonic()
    if (last_send_time+0.5) < this_time:
      #print("timeout, no web in %.2f s" % (this_time-last_send_time))
      dat = messaging.new_message('testJoystick')
      dat.testJoystick.axes = [0,0,0,0]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
    time.sleep(0.1)

def main():
  threading.Thread(target=handle_timeout, daemon=True).start()
  app.run(host="0.0.0.0")

if __name__ == '__main__':
  main()
