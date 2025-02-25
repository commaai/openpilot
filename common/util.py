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
