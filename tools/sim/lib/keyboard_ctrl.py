import time
import sys
import tty, termios

def getch():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

def keyboard_poll_thread(q):
  while True:
    c = getch()
    print("got %s" % c)
    if c == '1':
      q.put(str("cruise_up"))
    if c == '2':
      q.put(str("cruise_down"))
    if c == '3':
      q.put(str("cruise_cancel"))
    if c == 'q':
      exit(0)

if __name__ == '__main__':
  keyboard_poll_thread()

