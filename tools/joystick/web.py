#!/usr/bin/env python3
from flask import Flask
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
// Create JoyStick object into the DIV 'joyDiv'
var joy = new JoyStick('joyDiv');
setInterval(function(){
  var x = -joy.GetX()/100;
  var y = joy.GetY()/100;
  let xhr = new XMLHttpRequest();
  xhr.open("GET", "/control/"+x+"/"+y);
  xhr.send();
}, 50);
</script>
"""

@app.route("/")
def hello_world():
  return index

@app.route("/control/<x>/<y>")
def control(x, y):
  x,y = float(x), float(y)
  x = max(-1, min(1, x))
  y = max(-1, min(1, y))
  dat = messaging.new_message('testJoystick')
  dat.testJoystick.axes = [y,x]
  dat.testJoystick.buttons = [False]
  pm.send('testJoystick', dat)
  return ""

if __name__ == '__main__':
  app.run(host="0.0.0.0")

