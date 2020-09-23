import os
import time
import subprocess

def test_ubloxd():
  # unlogger_command = '~/openpilot/tools/replay/unlogger.py "202fc0c905c39dd8|2020-09-09--10-50-37"'
  subprocess.Popen(["python", "~/openpilot/tools/replay/unlogger.py", "202fc0c905c39dd8|2020-09-09--10-50-37"])

  time.sleep(5)

  output = os.popen("valgrind --leak-check=full ~/openpilot/selfdrive/locationd/ubloxd & sleep 10; kill $!")
  print("\n\n\n\n")
  while True:
    s = output.read()
    if s == "":
      break
    print(s)
  print("Lol")


if __name__ == "__main__":
  test_ubloxd()
