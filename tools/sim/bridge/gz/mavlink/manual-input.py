import sys
sys.path.append("/home/kevinm/Documents/projects/drone/openpilot/cereal")
import socket
import termios
import threading
import cv2
import capnp
import numpy as np
MAX_BUFFER_SIZE = 10*1024*1024  # 10MB
import log_capnp
import car_capnp
from collections import namedtuple
import time

stop = False
class CarControl:
  def __init__(self, steering, gas):
    self.steering = steering
    self.gas = gas

carState = CarControl(0, 0)

def poll():
  while True and not stop:
    data = bytearray(MAX_BUFFER_SIZE)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 4069))
        s.send(bytes([0]))
        offset = 0
        while offset < len(data):
            recv = s.recv(len(data) - offset)
            data[offset:offset+len(recv)] = recv
            offset += len(recv)
            if len(recv) == 0:
                break

    thumbnail = log_capnp.Thumbnail.from_bytes_packed(data)
    original = np.frombuffer(thumbnail.thumbnail, dtype=np.uint8).reshape((1080, 1920, 3))
    im = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ims = cv2.resize(im, (800, 600))
    cv2.imshow("road", ims)
    cv2.waitKey(1)

def start_vehicle():
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.connect(("127.0.0.1", 4069))
          s.send(bytes([3]))
          confirmed = s.recv(1)
          if len(confirmed) == 0:
            raise Exception("Failed to start vehicle")

          if confirmed[0] != 0:
            raise Exception("Failed to start vehicle")

def apply_controls():
  actuators = car_capnp.CarControl.Actuators.new_message(
          steeringAngleDeg = carState.steering,
          gas = carState.gas)
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.connect(("127.0.0.1", 4069))
      s.send(bytes([2]))
      s.sendall(actuators.to_bytes_packed())

def control_loop():
  while not stop:
    apply_controls()
    time.sleep(0.1)

def read_keyboard():
  return sys.stdin.read(1)

def setup_terminal():
  attributes = termios.tcgetattr(sys.stdin)
  attributes[3] &= ~(termios.ECHO | termios.ICANON)
  termios.tcsetattr(sys.stdin, termios.TCSANOW, attributes)


def main():
  setup_terminal()
  poll_thread = threading.Thread(target=poll)
  poll_thread.start()
  command_thread = threading.Thread(target=control_loop)
  command_thread.start()
  start_vehicle()
  while True:
    read = read_keyboard()
    if read == "w":
      carState.gas += 1
    elif read == "a":
      carState.steering += 5
    elif read == "d":
      carState.steering -= 5
    elif read == "s":
      carState.gas -= 1
    elif read == "q":
      print("quit")
      stop = True
      break
    print(f"gas: {carState.gas}, steering: {carState.steering}")
  command_thread.join()
  poll_thread.join()

if __name__ == "__main__":
  main()
