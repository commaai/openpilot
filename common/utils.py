class Freezable:
  FROZEN: bool = False

  def freeze(self):
    if not self.FROZEN:
      self.FROZEN = True

  def __setattr__(self, *args, **kwargs):
    if self.FROZEN:
      raise Exception("cannot modify frozen object")
    super().__setattr__(*args, **kwargs)
