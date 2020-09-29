import os
import time
from multiprocessing import Process
import cereal.messaging as messaging
import random

def valgrindlauncher(arg, cwd):
  os.chdir(cwd)

  # Run valgrind on a process
  command = "valgrind " + arg
  print(command)
  output = os.popen(command)
  while True:
    s = output.read()
    if s == "":
      break
    print(s)

def test_ubloxd():
  running = Process(name="ublox", target=valgrindlauncher, args=("./ubloxd & sleep 10; kill $!", "../locationd"))
  running.start()

def random_carstate():
  fields = ["vEgo", "aEgo", "gas", "steeringAngle"]
  msg = messaging.new_message("carState")
  cs = msg.carState
  for f in fields:
    setattr(cs, f, random.random() * 10)
  return msg

def send_gps():
  pub_sock = messaging.pub_sock("ubloxRaw")
  time.sleep(3)

  msg = random_carstate()
  pub_sock.send(msg.to_bytes())

# python test_loggerd in tests kind of works, with VALGRIND=1 in ENV
def test_loggerd():
  output = os.popen("valgrind --leak-check=full ~/openpilot/selfdrive/loggerd/loggerd & sleep 30; kill $!")
  print(2 * "\n")
  while True:
    s = output.read()
    if s == "":
      break
    print(s)
  print("Lol")


if __name__ == "__main__":
  # pm = messaging.PubMaster(['ubloxRaw'])
  send_gps()
  # test_ubloxd()
