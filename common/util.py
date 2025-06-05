import os
import subprocess

def sudo_write(val: str, path: str) -> None:
  try:
    with open(path, 'w') as f:
      f.write(str(val))
  except PermissionError:
    os.system(f"sudo chmod a+w {path}")
    try:
      with open(path, 'w') as f:
        f.write(str(val))
    except PermissionError:
      # fallback for debugfs files
      os.system(f"sudo su -c 'echo {val} > {path}'")

def sudo_read(path: str) -> str:
  try:
    return subprocess.check_output(f"sudo cat {path}", shell=True, encoding='utf8').strip()
  except Exception:
    return ""

class MovingAverage:
  def __init__(self, window_size: int):
    self.window_size: int = window_size
    self.buffer: list[float] = [0.0] * window_size
    self.index: int = 0
    self.count: int = 0
    self.sum: float = 0.0

  def add_value(self, new_value: float):
    # Update the sum: subtract the value being replaced and add the new value
    self.sum -= self.buffer[self.index]
    self.buffer[self.index] = new_value
    self.sum += new_value

    # Update the index in a circular manner
    self.index = (self.index + 1) % self.window_size

    # Track the number of added values (for partial windows)
    self.count = min(self.count + 1, self.window_size)

  def get_average(self) -> float:
    if self.count == 0:
      return float('nan')
    return self.sum / self.count
