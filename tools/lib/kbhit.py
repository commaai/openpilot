#!/usr/bin/env python
import sys
import termios
import atexit
from select import select

STDIN_FD = sys.stdin.fileno()

class KBHit:
  def __init__(self) -> None:
    ''' Creates a KBHit object that you can call to do various keyboard things.
    '''

    self.set_kbhit_terminal()

  def set_kbhit_terminal(self) -> None:
    ''' Save old terminal settings for closure, remove ICANON & ECHO flags.
    '''

    # Save the terminal settings
    self.old_term = termios.tcgetattr(STDIN_FD)
    self.new_term = self.old_term.copy()

    # New terminal setting unbuffered
    self.new_term[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, self.new_term)

    # Support normal-terminal reset at exit
    atexit.register(self.set_normal_term)

  def set_normal_term(self) -> None:
    ''' Resets to normal terminal. On Windows this is a no-op.
    '''

    termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, self.old_term)

  @staticmethod
  def getch() -> str:
    ''' Returns a keyboard character after kbhit() has been called.
      Should not be called in the same program as getarrow().
    '''
    return sys.stdin.read(1)

  @staticmethod
  def getarrow() -> int:
    ''' Returns an arrow-key code after kbhit() has been called. Codes are
    0 : up
    1 : right
    2 : down
    3 : left
    Should not be called in the same program as getch().
    '''

    c = sys.stdin.read(3)[2]
    vals = [65, 67, 66, 68]

    return vals.index(ord(c))

  @staticmethod
  def kbhit():
    ''' Returns True if keyboard character was hit, False otherwise.
    '''
    return select([sys.stdin], [], [], 0)[0] != []


# Test
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
