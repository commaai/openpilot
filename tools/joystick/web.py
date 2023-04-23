#!/usr/bin/env python3
import time
import threading
from flask import Flask

import cereal.messaging as messaging

app = Flask(__name__)
pm = messaging.PubMaster(['testJoystick'])

index = """
<html>
<body>
<span style="font-size:18px;">Press wasd to control body</span><p id="output"></p>

<script type="text/javascript">
var keyEnum = { W_Key:0, A_Key:1, S_Key:2, D_Key:3 };
var keyArray = new Array(4).fill(false);


window.addEventListener('keydown', keydown_ivent);
window.addEventListener('keyup', keyup_ivent);

function keydown_ivent(e) {
	let key = '';
	switch (e.key) {
		case 'w':
                        keyArray[keyEnum.W_Key] = true;
                        break;
	}
	switch (e.key) {
		case 'a':
                        keyArray[keyEnum.A_Key] = true;
                        break;
	}
	switch (e.key) {
		case 's':
                        keyArray[keyEnum.S_Key] = true;
                        break;
	}
	switch (e.key) {
		case 'd':
                        keyArray[keyEnum.D_Key] = true;
                        break;
	}
}
function keyup_ivent(e) {
	let key = '';
	switch (e.key) {
		case 'w':
                        keyArray[keyEnum.W_Key] = false;
                        break;
	}
	switch (e.key) {
		case 'a':
                        keyArray[keyEnum.A_Key] = false;
                        break;
	}
	switch (e.key) {
		case 's':
                        keyArray[keyEnum.S_Key] = false;
                        break;
	}
	switch (e.key) {
		case 'd':
                        keyArray[keyEnum.D_Key] = false;
                        break;
	}
}

function isKeyDown(key)
{
    return keyArray[key];
}

setInterval(function(){
  var x = 0;
  var y = 0;
  if (isKeyDown(keyEnum.W_Key))
    x += 1;
  if (isKeyDown(keyEnum.S_Key))
    x -= 1;
  if (isKeyDown(keyEnum.D_Key))
    y -= 1;
  if (isKeyDown(keyEnum.A_Key))
    y += 1;
  let xhr = new XMLHttpRequest();
  xhr.open("GET", "/control/"+x+"/"+y);
  xhr.send();
}, 50);
</script>
"""

@app.route("/")
def hello_world():
  return index

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
      #print("timeout, no web in %.2f s" % (this_time-last_send_time))
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
