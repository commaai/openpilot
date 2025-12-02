#!/usr/bin/env python3
import sys
import termios
import atexit
from select import select

STDIN_FD = sys.stdin.fileno()


class KBHit:
  def __init__(self) -> None:
    self.set_kbhit_terminal()

  def set_kbhit_terminal(self) -> None:
    # Save the terminal settings
    self.old_term = termios.tcgetattr(STDIN_FD)
    self.new_term = self.old_term.copy()

    # New terminal setting unbuffered
    self.new_term[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, self.new_term)

    # Support normal-terminal reset at exit
    atexit.register(self.set_normal_term)

  def set_normal_term(self) -> None:
    termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, self.old_term)

  @staticmethod
  def getch() -> str:
    return sys.stdin.read(1)

  @staticmethod
  def getarrow() -> int:
    c = sys.stdin.read(3)[2]
    vals = [65, 67, 66, 68]
    return vals.index(ord(c))

  @staticmethod
  def kbhit():
    ''' Returns True if keyboard character was hit, False otherwise.
    '''
    return select([sys.stdin], [], [], 0)[0] != []


if __name__ == "__main__":

  kb = KBHit()

  print('Hit any key, or ESC to exit')

  while True:

    if kb.kbhit():
      c = kb.getch()
      if c == '\x1b':  # ESC
        break
      print(c)

  kb.set_normal_term()
