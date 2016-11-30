import os
import time


class Logger(object):
  def __init__(self, root, init_data):
    self.root = root
    self.init_data = init_data

    self.part = None
    self.data_dir = None
    self.cur_dir = None
    self.log_file = None
    self.started = False
    self.log_path = None
    self.lock_path = None
    self.log_file = None

  def open(self):
    self.data_dir = self.cur_dir + "--" + str(self.part)

    try:
      os.makedirs(self.data_dir)
    except OSError:
      pass

    self.log_path = os.path.join(self.data_dir, "rlog")
    self.lock_path = self.log_path + ".lock"

    open(self.lock_path, "wb").close()
    self.log_file = open(self.log_path, "wb")
    self.log_file.write(self.init_data)

  def start(self):
    self.part = 0
    self.cur_dir = self.root + time.strftime("%Y-%m-%d--%H-%M-%S")

    self.open()

    self.started = True

    return self.data_dir, self.part

  def stop(self):
    if not self.started:
      return
    self.log_file.close()
    os.unlink(self.lock_path)
    self.started = False

  def rotate(self):
    old_lock_path = self.lock_path
    old_log_file = self.log_file
    self.part += 1
    self.open()

    old_log_file.close()
    os.unlink(old_lock_path)

    return self.data_dir, self.part

  def log_data(self, d):
    if not self.started:
      return
    self.log_file.write(d)
