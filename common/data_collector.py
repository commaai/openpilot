from common.travis_checker import travis
from selfdrive.swaglog import cloudlog
import threading
import time
import os
from common.op_params import opParams

op_params = opParams()

class DataCollector:
  def __init__(self, file_path, keys, write_frequency=60, write_threshold=2):
    """
    This class provides an easy way to set up your own custom data collector to gather custom data.
    Parameters:
      file_path (str): The path you want your custom data to be written to.
      keys: (list): A string list containing the names of the values you want to collect.
                    Your data list needs to be in this order.
      write_frequency (int/float): The rate at which to write data in seconds.
      write_threshold (int): The length of the data list we need to collect before considering writing.
    Example:
      data_collector = DataCollector('/data/openpilot/custom_data', ['v_ego', 'a_ego', 'custom_dict'], write_frequency=120)
    """

    self.log_data = op_params.get('log_data', False)
    self.file_path = file_path
    self.keys = keys
    self.write_frequency = write_frequency
    self.write_threshold = write_threshold
    self.data = []
    self.last_write_time = time.time()
    self.thread_running = False
    self.is_initialized = False

  def initialize(self):  # add keys to top of data file
    if not os.path.exists(self.file_path) and not travis:
      with open(self.file_path, "w") as f:
        f.write('{}\n'.format(self.keys))
    self.is_initialized = True

  def append(self, sample):
    """
    Appends your sample to a central self.data variable that gets written to your specified file path every n seconds.
    Parameters:
      sample: Can be any type of data. List, dictionary, numbers, strings, etc.
      Or a combination: dictionaries, booleans, and floats in a list
    Continuing from the example above, we assume that the first value is your velocity, and the second
    is your acceleration. IMPORTANT: If your values and keys are not in the same order, you will have trouble figuring
    what data is what when you want to process it later.
    Example:
      data_collector.append([17, 0.5, {'a': 1}])
    """

    if self.log_data:
      if len(sample) != len(self.keys):
        raise Exception("You need the same amount of data as you specified in your keys")
      if not self.is_initialized:
        self.initialize()
      self.data.append(sample)
      self.check_if_can_write()

  def reset(self, reset_type=None):
    if reset_type in ['data', 'all']:
      self.data = []
    if reset_type in ['time', 'all']:
      self.last_write_time = time.time()

  def check_if_can_write(self):
    """
    You shouldn't ever need to call this. It checks if we should write, then calls a thread to do so
    with a copy of the current gathered data. Then it clears the self.data variable so that new data
    can be added and it won't be duplicated in the next write.
    If the thread is still writing by the time of the next write, which shouldn't ever happen unless
    you set a low write frequency, it will skip creating another write thread. If this occurs,
    something is wrong with writing.
    """

    if (time.time() - self.last_write_time) >= self.write_frequency and len(self.data) >= self.write_threshold and not travis:
      if not self.thread_running:
        write_thread = threading.Thread(target=self.write, args=(self.data,))
        write_thread.daemon = True
        write_thread.start()
        # self.write(self.data)  # non threaded approach
        self.reset(reset_type='all')
      elif self.write_frequency > 30:
        cloudlog.warning('DataCollector write thread is taking a while to write data.')

  def write(self, current_data):
    """
    Only write data that has been added so far in background. self.data is still being appended to in
    foreground so in the next write event, new data will be written. This eliminates lag causing openpilot
    critical processes to pause while a lot of data is being written.
    """

    self.thread_running = True
    with open(self.file_path, "a") as f:
      f.write('{}\n'.format('\n'.join(map(str, current_data))))  # json takes twice as long to write
    self.reset(reset_type='time')
    self.thread_running = False
