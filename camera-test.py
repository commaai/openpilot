import cv2
import numpy as np
from beamngpy import BeamNGpy
from beamngpy.sensors import Camera

def main():
  bng = BeamNGpy('localhost', 64256)
  bng.open(launch=False)

  cars = bng.get_current_vehicles()
  if not cars:
    print("No cars.")
    return

  id = list(cars.keys())[0]
  car = cars[id]
  car.connect(bng)

  pos = (0, 0, 1.5)
  dir = (0, -1, 0)
  res = (1280, 720) # test for 720p, OP uses larger

  cam_narrow = Camera('narrow', bng, car, dir=dir, resolution=res, fov=30, requested_update_time=0.05)
  cam_wide = Camera('wide', bng, car, dir=dir, resolution=res, fov=60, requested_update_time=0.05)
  cam_ultrawide = Camera('ultrawide', bng, car, dir=dir, resolution=res, fov=120, requested_update_time=0.05)

  while True:
    data_narrow = cam_narrow.poll()
    if "colour" in data_narrow:
      img_narrow = cv2.cvtColor(np.array(data_narrow["colour"].convert("RGB")), cv2.COLOR_RGB2BGR)
      cv2.imshow("Narrow 30 deg", img_narrow)

    data_wide = cam_wide.poll()
    if "colour" in data_wide:
      img_wide = cv2.cvtColor(np.array(data_wide["colour"].convert("RGB")), cv2.COLOR_RGB2BGR)
      cv2.imshow("Wide 60 deg", img_wide)

    data_ultrawide= cam_ultrawide.poll()
    if "colour" in data_ultrawide:
      img_ultrawide = cv2.cvtColor(np.array(data_ultrawide["colour"].convert("RGB")), cv2.COLOR_RGB2BGR)
      cv2.imshow("Ultrawide 120 deg", img_ultrawide)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()