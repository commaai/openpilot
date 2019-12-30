from common.op_params import opParams
import ast
import time


class opTune:
  def __init__(self):
    self.op_params = opParams()
    self.sleep_time = 1.0
    self.start()

  def start(self):
    print('Welcome to the opParams command line live tuner!')
    editable = [p for p in self.op_params.get() if p in self.op_params.default_params and self.op_params.default_params[p]['live']]
    while True:
      print('Choose a parameter to tune:')
      print('\n'.join(['{}. {}'.format(idx + 1, p) for idx, p in enumerate(editable)]))
      choice = input('>> ')
      if not choice:
        print('Exiting opTune!')
        break
      choice = ast.literal_eval(choice) - 1
      if choice not in range(len(editable)):
        self.message('Error, not in range!')
        continue
      self.chosen(editable[choice])

  def chosen(self, param):
    allowed_types = self.op_params.default_params[param]['allowed_types']
    print('\nChosen parameter: {}'.format(param))
    print('Current value: {}'.format(self.op_params.get(param)))
    print('\n- Description: {}'.format(self.op_params.default_params[param]['description']))
    print('- Allowed types: {}\n'.format(', '.join([str(i).split("'")[1] for i in allowed_types])))
    while True:
      value = input('Enter value: ')
      if value == '':
        self.message('Exiting this parameter...')
        break

      status, value = self.parse_input(value)
      if not status:
        self.message('Cannot parse input!')
        continue

      if not any([isinstance(value, typ) for typ in allowed_types]):
        self.message('The type of data you entered ({}) is not allowed with this parameter!\n'.format(str(type(value)).split("'")[1]))
        continue
      self.op_params.put(param, value)
      print('Saved {} with value: {}! (type: {})\n'.format(param, value, str(type(value)).split("'")[1]))

  def message(self, msg):
    print('--------\n{}\n--------'.format(msg), flush=True)
    time.sleep(self.sleep_time)
    print()

  def parse_input(self, dat):
    dat = dat.replace("'", '"')
    try:
      dat = ast.literal_eval(dat)
    except:
      try:
        dat = ast.literal_eval('"{}"'.format(dat))
      except ValueError:
        return False, dat
    return True, dat


opTune()
