#!/usr/bin/python3
import sys
import os
import stat
import subprocess
import json
from common.text_window import TextWindow
import time

# Required for sensord not to bus-error on startup
# commaai/cereal#22
try:
  os.mkdir("/dev/shm")
except FileExistsError:
  pass
except PermissionError:
  print("WARNING: failed to make /dev/shm")

try:
  with open('/tmp/test-results.json', 'r') as infile:
    data = json.load(infile)
except:
  data = {'sensor-pass': 0, 'sensor-fail': 0}

STARTUP_SCRIPT = "/data/data/com.termux/files/continue.sh"
try:
  with open(STARTUP_SCRIPT, 'w') as startup_script:
    startup_script.write("#!/usr/bin/bash\n\n/data/openpilot/selfdrive/test/sensor_test_bootloop.py\n")
  os.chmod(STARTUP_SCRIPT, stat.S_IRWXU)
except:
  print("Failed to install new startup script -- aborting")
  sys.exit(-1)

sensord_env = {**os.environ, 'SENSOR_TEST': '1'}
process = subprocess.run("./sensord", cwd="/data/openpilot/selfdrive/sensord", env=sensord_env)

if process.returncode == 40:
  text = "Current run: SUCCESS\n"
  data['sensor-pass'] += 1
else:
  text = "Current run: FAIL\n"
  data['sensor-fail'] += 1

  timestr = str(int(time.time()))
  with open('/tmp/dmesg-' + timestr + '.log', 'w') as dmesg_out:
    subprocess.call('dmesg', stdout=dmesg_out, shell=False)
  with open("/tmp/logcat-" + timestr + '.log', 'w') as logcat_out:
    subprocess.call(['logcat','-d'], stdout=logcat_out, shell=False)

text += "Sensor pass history: " + str(data['sensor-pass']) + "\n"
text += "Sensor fail history: " + str(data['sensor-fail']) + "\n"

print(text)

with open('/tmp/test-results.json', 'w') as outfile:
  json.dump(data, outfile, indent=4)

with TextWindow(text) as status:
  for _ in range(100):
    if status.get_status() == 1:
      with open(STARTUP_SCRIPT, 'w') as startup_script:
        startup_script.write("#!/usr/bin/bash\n\ncd /data/openpilot\nexec ./launch_openpilot.sh\n")
      os.chmod(STARTUP_SCRIPT, stat.S_IRWXU)
      break
    time.sleep(0.1)

subprocess.Popen("reboot")
