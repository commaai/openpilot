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
.column {float: left;width: 50%;padding: 10px;height: 100%;display: flex;align-items: center;justify-content: center;}
#inner-column {width: calc(100vw/2);height:calc(100vh/4);}
.row:after {content: "";display: table;clear: both;}
@media screen and (orientation: portrait) {
  .column { width: 100%;height: 50%;}
  #inner-column {width: 100vw;height:calc(100vh/8);}}
</style>
<script src="https://github.com/bobboteck/JoyStick/releases/download/v1.1.6/joy.min.js"></script>
</head>
<body>
<div class="row">
<div class="column" id="joyDiv"></div>
<div class="column">
  <div class="row">
  <div class="column" id="inner-column">Hip angle<div id="output1"> : 180</div></div>
  <div class="column" id="inner-column">0 <input type="range" id="slider1" min="0" max="360" value="180" step="1" autocomplete="off" style="width:90%;"> 360</div>
  <div class="column" id="inner-column">Knee angle<div id="output2"> : 180</div></div>
  <div class="column" id="inner-column">0 <input type="range" id="slider2" min="0" max="360" value="180" step="1" autocomplete="off" style="width:90%;"> 360</div>
</div>
</div>
</div>
<script type="text/javascript">
// Add range sliders for knee
var slider1 = document.getElementById("slider1");
var slider2 = document.getElementById("slider2");
window.addEventListener("load", function(){
  slider1.addEventListener("change", function(){
      document.getElementById("output1").innerHTML = " : " + this.value;
 });
  slider2.addEventListener("change", function(){
      document.getElementById("output2").innerHTML = " : " + this.value;
 });
});
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
var joy = new JoyStick('joyDiv', { "height": scr_min, "width": scr_min });
setInterval(function(){
  var x = -joy.GetX()/100;
  var y = joy.GetY()/100;
  if (x === 0 && y === 0 && gamepad !== null) {
    let gamepadstate = navigator.getGamepads()[gamepad.index];
    x = -gamepadstate.axes[0];
    y = -gamepadstate.axes[1];
  }
  let xhr = new XMLHttpRequest();
  xhr.open("GET", "/control/"+x+"/"+y+"/"+slider1.value+"/"+slider2.value);
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
      dat.testJoystick.axes = [0,0,180,180]
      dat.testJoystick.buttons = [False]
      pm.send('testJoystick', dat)
    time.sleep(0.1)

def main():
  threading.Thread(target=handle_timeout, daemon=True).start()
  app.run(host="0.0.0.0")

if __name__ == '__main__':
  main()
