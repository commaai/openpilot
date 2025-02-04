import capnp
import os
import sys
sys.path.append("/home/kevinm/Documents/projects/drone/openpilot/cereal")
import log_capnp
import socket
import numpy as np
import matplotlib.pyplot as plt

def main():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 4069))
    print("connected")
    s.send(bytes([0]))
    data = bytearray(10*1024*1024)  # 10MB
    offset = 0
    print("receiving")
    while True:
      received = s.recv(len(data) - offset)
      data[offset:offset+len(received)] = received
      offset += len(received)
      if len(received) == 0:
        break

    print(f"{offset} bytes received")
    print("deserializing")
    thumbnail = log_capnp.Thumbnail.from_bytes_packed(data)
    flat = np.frombuffer(thumbnail.thumbnail, dtype=np.uint8)
    pixels = flat.reshape(720, 1280, 3)

    plt.imshow(pixels)
    plt.show()



if __name__ == "__main__":
  main()
