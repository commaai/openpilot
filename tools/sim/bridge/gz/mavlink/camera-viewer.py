from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import zlib
import socket
import capnp
import sys
sys.path.append("/home/kevinm/Documents/projects/drone/openpilot/cereal")
import log_capnp


def nv12_to_rgb(nv12: bytes | bytearray, size: tuple[int, int]) -> Image:
    w, h = size
    n = w * h
    y, u, v = nv12[:n], nv12[n + 0::2], nv12[n + 1::2]
    yuv = bytearray(3 * n)
    yuv[0::3] = y
    yuv[1::3] = Image.frombytes(
        'L', (w // 2, h // 2), u).resize(size).tobytes()
    yuv[2::3] = Image.frombytes(
        'L', (w // 2, h // 2), v).resize(size).tobytes()
    return Image.frombuffer('YCbCr', size, yuv).convert('RGB')


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
        data = zlib.decompress(data)
        thumbnail = log_capnp.Thumbnail.from_bytes_packed(data)
        rgb = nv12_to_rgb(thumbnail.thumbnail, (1280, 720))

        plt.imshow(rgb)
        plt.show()


if __name__ == "__main__":
    main()
