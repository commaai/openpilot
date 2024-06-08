import signal


class Freezable:
  _frozen: bool = False

  def freeze(self):
    if not self._frozen:
      self._frozen = True

  def __setattr__(self, *args, **kwargs):
    if self._frozen:
      raise Exception("cannot modify frozen object")
    super().__setattr__(*args, **kwargs)


class ExitHandler:
  def __init__(self):
    self.signal = 0
    self._do_exit = False

    signal.signal(signal.SIGINT, self._set_do_exit)
    signal.signal(signal.SIGTERM, self._set_do_exit)
    # signal.signal(signal.SIGHUP, self._set_do_exit)

  def _set_do_exit(self, signum, frame):
    self.signal = signum
    self._do_exit = True

  def __bool__(self):
    return self._do_exit
