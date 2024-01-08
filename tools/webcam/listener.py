from openpilot.system.camerad.snapshot.snapshot import get_snapshots
import cv2 as cv

def main():
  print("LISTENING...")
  while True:
    rear, front = get_snapshots()
    rear= cv.cvtColor(rear, cv.COLOR_RGB2BGR)
    cv.imshow("preview", rear)
    cv.waitKey(1)

if __name__ == "__main__":
  main()
