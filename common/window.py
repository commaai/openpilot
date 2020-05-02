import sys
import pygame

class Window():
  def __init__(self, w, h, caption="window", double=False):
    self.w = w
    self.h = h
    pygame.init()
    pygame.display.set_caption(caption)
    self.double = double
    if self.double:
      self.screen = pygame.display.set_mode((w*2,h*2), pygame.DOUBLEBUF)
    else:
      self.screen = pygame.display.set_mode((w,h), pygame.DOUBLEBUF)
    self.camera_surface = pygame.surface.Surface((w,h), 0, 24).convert()

  def draw(self, out):
    pygame.surfarray.blit_array(self.camera_surface, out.swapaxes(0,1))
    if self.double:
      camera_surface_2x = pygame.transform.scale2x(self.camera_surface)
      self.screen.blit(camera_surface_2x, (0, 0))
    else:
      self.screen.blit(self.camera_surface, (0, 0))
    pygame.display.flip()
  
  def getkey(self):
    while 1:
      event = pygame.event.wait()
      if event.type == QUIT:
        pygame.quit()
        sys.exit()
      if event.type == KEYDOWN:
        return event.key

  def getclick(self):
    for event in pygame.event.get():
      if event.type == pygame.MOUSEBUTTONDOWN:
        mx, my = pygame.mouse.get_pos()
        return mx, my

if __name__ == "__main__":
  import numpy as np
  win = Window(200, 200)
  img = np.zeros((200,200,3), np.uint8)
  while 1:
    print("draw")
    img += 1
    win.draw(img)

