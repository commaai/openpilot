from common.op_params import opParams
import time
import ast


class opEdit:  # use by running `python /data/openpilot/op_edit.py`
  def __init__(self):
    self.op_params = opParams()
    self.params = None
    self.sleep_time = 1.0
    print('Welcome to the opParams command line editor!')
    print('Here are your parameters:\n')
    self.run_loop()

  def run_loop(self):
    print('Welcome to the opParams command line editor!')
    print('Here are your parameters:\n')
    while True:
      self.params = self.op_params.get()
      values_list = [self.params[i] if len(str(self.params[i])) < 20 else '{} ... {}'.format(str(self.params[i])[:30], str(self.params[i])[-15:]) for i in self.params]
      to_print = ['{}. {}: {} (type: {})'.format(idx + 1, i, values_list[idx], str(type(self.params[i])).split("'")[1]) for idx, i in enumerate(self.params)]
      to_print.append('{}. Add new parameter!'.format(len(self.params) + 1))
      to_print.append('{}. Delete parameter!'.format(len(self.params) + 2))
      print('\n'.join(to_print))
      print('\nChoose a parameter to explore (by integer index): ')
      choice = input('>> ')
      parsed, choice = self.parse_choice(choice)
      if parsed == 'continue':
        continue
      elif parsed == 'add':
        if self.add_parameter() == 'error':
          return
      elif parsed == 'change':
        if self.change_parameter(choice) == 'error':
          return
      elif parsed == 'delete':
        if self.delete_parameter() == 'error':
          return
      elif parsed == 'error':
        return

  def parse_choice(self, choice):
    if choice.isdigit():
      choice = int(choice)
    elif choice == '':
      print('Exiting...')
      return 'error', choice
    else:
      print('\nNot an integer!\n', flush=True)
      time.sleep(self.sleep_time)
      return 'retry', choice
    if choice not in range(1, len(self.params) + 3):  # three for add/delete parameter
      print('Not in range!\n', flush=True)
      time.sleep(self.sleep_time)
      return 'continue', choice

    if choice == len(self.params) + 1:  # add new parameter
      return 'add', choice

    if choice == len(self.params) + 2:  # delete parameter
      return 'delete', choice

    return 'change', choice

  def change_parameter(self, choice):
    chosen_key = list(self.params)[choice - 1]
    extra_info = False
    if chosen_key in self.op_params.default_params:
      extra_info = True
      param_allowed_types = self.op_params.default_params[chosen_key]['allowed_types']
      param_description = self.op_params.default_params[chosen_key]['description']

    old_value = self.params[chosen_key]
    print('Chosen parameter: {}'.format(chosen_key))
    print('Current value: {} (type: {})'.format(old_value, str(type(old_value)).split("'")[1]))
    if extra_info:
      print('\nDescription: {}'.format(param_description))
      print('Allowed types: {}\n'.format(', '.join([str(i).split("'")[1] for i in param_allowed_types])))
    print('Enter your new value:')
    new_value = input('>> ')
    if len(new_value) == 0:
      print('Entered value cannot be empty!')
      return 'error'
    status, new_value = self.parse_input(new_value)
    if not status:
      print('Cannot parse input, exiting!')
      return 'error'

    if extra_info and not any([isinstance(new_value, typ) for typ in param_allowed_types]):
      print('The type of data you entered ({}) is not allowed with this parameter!\n'.format(str(type(new_value)).split("'")[1]))
      time.sleep(self.sleep_time)
      return

    print('\nOld value: {} (type: {})'.format(old_value, str(type(old_value)).split("'")[1]))
    print('New value: {} (type: {})'.format(new_value, str(type(new_value)).split("'")[1]))
    print('Do you want to save this?')
    choice = input('[Y/n]: ').lower()
    if choice == 'y':
      self.op_params.put(chosen_key, new_value)
      print('\nSaved!\n')
    else:
      print('\nNot saved!\n', flush=True)
    time.sleep(self.sleep_time)

  def parse_input(self, dat):
    try:
      dat = ast.literal_eval(dat)
    except:
      try:
        dat = ast.literal_eval('"{}"'.format(dat))
      except ValueError:
        return False, dat
    return True, dat

  def delete_parameter(self):
    print('Enter the name of the parameter to delete:')
    key = input('>> ')
    status, key = self.parse_input(key)
    if not status:
      print('Cannot parse input, exiting!')
      return 'error'
    if not isinstance(key, str):
      print('Input must be a string!')
      return 'error'
    if key not in self.params:
      print("Parameter doesn't exist!")
      return 'error'

    value = self.params.get(key)
    print('Parameter name: {}'.format(key))
    print('Parameter value: {} (type: {})'.format(value, str(type(value)).split("'")[1]))
    print('Do you want to delete this?')

    choice = input('[Y/n]: ').lower()
    if choice == 'y':
      self.op_params.delete(key)
      print('\nDeleted!\n')
    else:
      print('\nNot saved!\n', flush=True)
    time.sleep(self.sleep_time)

  def add_parameter(self):
    print('Type the name of your new parameter:')
    key = input('>> ')
    if len(key) == 0:
      print('Entered key cannot be empty!')
      return 'error'
    status, key = self.parse_input(key)
    if not status:
      print('Cannot parse input, exiting!')
      return 'error'
    if not isinstance(key, str):
      print('Input must be a string!')
      return 'error'

    print("Enter the data you'd like to save with this parameter:")
    value = input('>> ')
    status, value = self.parse_input(value)
    if not status:
      print('Cannot parse input, exiting!')
      return 'error'

    print('Parameter name: {}'.format(key))
    print('Parameter value: {} (type: {})'.format(value, str(type(value)).split("'")[1]))
    print('Do you want to save this?')

    choice = input('[Y/n]: ').lower()
    if choice == 'y':
      self.op_params.put(key, value)
      print('\nSaved!\n')
    else:
      print('\nNot saved!\n', flush=True)
    time.sleep(self.sleep_time)


opEdit()
