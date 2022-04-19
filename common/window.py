import sys
import pygame  # pylint: disable=import-error
import cv2  # pylint: disable=import-error

class Window():
  def __init__(self, w, h, caption="window", double=False, halve=False):
    self.w = w
    self.h = h
    pygame.display.init()
    pygame.display.set_caption(caption)
    self.double = double
    self.halve = halve
    if self.double:
      self.rw, self.rh = w*2, h*2
    elif self.halve:
      self.rw, self.rh = w//2, h//2
    else:
      self.rw, self.rh = w, h
    self.screen = pygame.display.set_mode((self.rw, self.rh))
    pygame.display.flip()

    # hack for xmonad, it shrinks the window by 6 pixels after the display.flip
    if self.screen.get_width() != self.rw:
      self.screen = pygame.display.set_mode((self.rw+(self.rw-self.screen.get_width()), self.rh+(self.rh-self.screen.get_height())))
      pygame.display.flip()

  def draw(self, out):
    pygame.event.pump()
    if self.double:
      out2 = cv2.resize(out, (self.w*2, self.h*2))
      pygame.surfarray.blit_array(self.screen, out2.swapaxes(0, 1))
    elif self.halve:
      out2 = cv2.resize(out, (self.w//2, self.h//2))
      pygame.surfarray.blit_array(self.screen, out2.swapaxes(0, 1))
    else:
      pygame.surfarray.blit_array(self.screen, out.swapaxes(0, 1))
    pygame.display.flip()

  def getkey(self):
    while 1:
      event = pygame.event.wait()
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
      if event.type == pygame.KEYDOWN:
        return event.key

  def getclick(self):
    for event in pygame.event.get():
      if event.type == pygame.MOUSEBUTTONDOWN:
        mx, my = pygame.mouse.get_pos()
        return mx, my

if __name__ == "__main__":
  import numpy as np
  win = Window(200, 200, double=True)
  img = np.zeros((200, 200, 3), np.uint8)
  while 1:
    print("draw")
    img += 1
    win.draw(img)
