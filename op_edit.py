#!/usr/bin/env python3
import time
import ast
import difflib

from common.op_params import opParams
from common.colors import COLORS
from collections import OrderedDict


class opEdit:  # use by running `python /data/openpilot/op_edit.py`
  def __init__(self):
    self.op_params = opParams()
    self.params = None
    self.sleep_time = 0.75
    self.live_tuning = self.op_params.get('op_edit_live_mode')
    self.username = self.op_params.get('username')
    self.type_colors = {int: COLORS.BASE(179), float: COLORS.BASE(179),
                        bool: {False: COLORS.RED, True: COLORS.OKGREEN},
                        type(None): COLORS.BASE(177),
                        str: COLORS.BASE(77)}

    self.last_choice = None

    self.run_init()

  def run_init(self):
    if self.username is None:
      self.success('\nWelcome to the opParams command line editor!', sleep_time=0)
      self.prompt('Parameter \'username\' is missing! Would you like to add your Discord username for easier crash debugging?')

      username_choice = self.input_with_options(['Y', 'n', 'don\'t ask again'], default='n')[0]
      if username_choice == 0:
        self.prompt('Please enter your Discord username so the developers can reach out if a crash occurs:')
        username = ''
        while username == '':
          username = input('>> ').strip()
        self.success('Thanks! Saved your Discord username\n'
                     'Edit the \'username\' parameter at any time to update', sleep_time=3.0)
        self.op_params.put('username', username)
        self.username = username
      elif username_choice == 2:
        self.op_params.put('username', False)
        self.info('Got it, bringing you into opEdit\n'
                  'Edit the \'username\' parameter at any time to update', sleep_time=3.0)
    else:
      self.success('\nWelcome to the opParams command line editor, {}!'.format(self.username), sleep_time=0)

    self.run_loop()

  def run_loop(self):
    while True:
      if not self.live_tuning:
        self.info('Here are your parameters:', end='\n', sleep_time=0)
      else:
        self.info('Here are your live parameters:', sleep_time=0)
        self.info('(changes take effect within {} seconds)'.format(self.op_params.read_frequency), end='\n', sleep_time=0)
      self.params = self.op_params.get(force_live=True)
      if self.live_tuning:  # only display live tunable params
        self.params = {k: v for k, v in self.params.items() if self.op_params.param_info(k).live}

      self.params = OrderedDict(sorted(self.params.items(), key=self.sort_params))

      values_list = []
      for k, v in self.params.items():
        if len(str(v)) < 30 or len(str(v)) <= len('{} ... {}'.format(str(v)[:30], str(v)[-15:])):
          v_color = ''
          if type(v) in self.type_colors:
            v_color = self.type_colors[type(v)]
            if isinstance(v, bool):
              v_color = v_color[v]
          v = '{}{}{}'.format(v_color, v, COLORS.ENDC)
        else:
          v = '{} ... {}'.format(str(v)[:30], str(v)[-15:])
        values_list.append(v)

      live = [COLORS.INFO + '(live!)' + COLORS.ENDC if self.op_params.param_info(k).live else '' for k in self.params]

      to_print = []
      blue_gradient = [33, 39, 45, 51, 87]
      last_key = ''
      last_info = None
      shown_dots = False
      for idx, param in enumerate(self.params):
        info = self.op_params.param_info(param)
        indent = self.get_sort_key(param).count(',')
        line = ''
        if not info.depends_on or param in last_info.children and \
          self.op_params.get(last_key) and self.op_params.get(info.depends_on):
          line = '{}. {}: {}  {}'.format(idx + 1, param, values_list[idx], live[idx])
          line = indent * '.' + line
        elif not shown_dots and last_info and param in last_info.children:
          line = '...'
          shown_dots = True
        if line:
          if idx == self.last_choice and self.last_choice is not None:
            line = COLORS.OKGREEN + line
          else:
            _color = blue_gradient[min(round(idx / len(self.params) * len(blue_gradient)), len(blue_gradient) - 1)]
            line = COLORS.BASE(_color) + line
          if last_info and len(last_info.children) and indent == 0:
            line = '\n' + line
            shown_dots = False
          to_print.append(line)

        if indent == 0:
          last_key = param
          last_info = info


      extras = {'a': ('Add new parameter', COLORS.OKGREEN),
                'd': ('Delete parameter', COLORS.FAIL),
                'l': ('Toggle live tuning', COLORS.WARNING),
                'e': ('Exit opEdit', COLORS.PINK)}

      to_print += ['---'] + ['{}. {}'.format(ext_col + e, ext_txt + COLORS.ENDC) for e, (ext_txt, ext_col) in extras.items()]
      print('\n'.join(to_print))
      self.prompt('\nChoose a parameter to edit (by index or name):')

      choice = input('>> ').strip().lower()
      parsed, choice = self.parse_choice(choice, len(self.params) + len(extras))
      if parsed == 'continue':
        continue
      elif parsed == 'add':
        self.add_parameter()
      elif parsed == 'change':
        self.last_choice = choice
        self.change_parameter(choice)
      elif parsed == 'delete':
        self.delete_parameter()
      elif parsed == 'live':
        self.last_choice = None
        self.live_tuning = not self.live_tuning
        self.op_params.put('op_edit_live_mode', self.live_tuning)  # for next opEdit startup
      elif parsed == 'exit':
        return

  def parse_choice(self, choice, opt_len):
    if choice.isdigit():
      choice = int(choice)
      choice -= 1
      if choice not in range(opt_len):  # number of options to choose from
        self.error('Not in range!')
        return 'continue', choice
      return 'change', choice

    if choice in ['a', 'add']:  # add new parameter
      return 'add', choice
    elif choice in ['d', 'delete', 'del']:  # delete parameter
      return 'delete', choice
    elif choice in ['l', 'live']:  # live tuning mode
      return 'live', choice
    elif choice in ['exit', 'e', '']:
      self.error('Exiting opEdit!', sleep_time=0)
      return 'exit', choice
    else:  # find most similar param to user's input
      param_sims = [(idx, self.str_sim(choice, param.lower())) for idx, param in enumerate(self.params)]
      param_sims = [param for param in param_sims if param[1] > 0.5]
      if len(param_sims) > 0:
        chosen_param = sorted(param_sims, key=lambda param: param[1], reverse=True)[0]
        return 'change', chosen_param[0]  # return idx

    self.error('Invalid choice!')
    return 'continue', choice

  def str_sim(self, a, b):
    return difflib.SequenceMatcher(a=a, b=b).ratio()

  def change_parameter(self, choice):
    while True:
      chosen_key = list(self.params)[choice]
      param_info = self.op_params.param_info(chosen_key)

      old_value = self.params[chosen_key]
      self.info('Chosen parameter: {}'.format(chosen_key), sleep_time=0)

      to_print = []
      if param_info.has_description:
        to_print.append(COLORS.OKGREEN + '>>  Description: {}'.format(param_info.description.replace('\n', '\n  > ')) + COLORS.ENDC)
      if param_info.has_allowed_types:
        to_print.append(COLORS.RED + '>>  Allowed types: {}'.format(', '.join([at.__name__ if isinstance(at, type) else str(at) for at in param_info.allowed_types])) + COLORS.ENDC)
      if param_info.live:
        live_msg = '>>  This parameter supports live tuning!'
        if not self.live_tuning:
          live_msg += ' Updates should take effect within {} seconds'.format(self.op_params.read_frequency)
        to_print.append(COLORS.YELLOW + live_msg + COLORS.ENDC)

      if to_print:
        print('\n{}\n'.format('\n'.join(to_print)))

      if param_info.is_list:
        self.change_param_list(old_value, param_info, chosen_key)  # TODO: need to merge the code in this function with the below to reduce redundant code
        return

      self.info('Current value: {} (type: {})'.format(old_value, type(old_value).__name__), sleep_time=0)

      while True:
        if param_info.live:
          self.prompt('\nEnter your new value or [Enter] to exit:')
        else:
          self.prompt('\nEnter your new value:')
        new_value = input('>> ').strip()
        if new_value == '':
          self.info('Exiting this parameter...', 0.5)
          return

        new_value = self.str_eval(new_value)

        if param_info.is_bool and type(new_value) is int:
          new_value = bool(new_value)

        if not param_info.is_valid(new_value):
          self.error('The type of data you entered ({}) is not allowed with this parameter!'.format(type(new_value).__name__))
          continue

        if param_info.live:  # stay in live tuning interface
          self.op_params.put(chosen_key, new_value)
          self.success('Saved {} with value: {}! (type: {})'.format(chosen_key, new_value, type(new_value).__name__))
        else:  # else ask to save and break
          print('\nOld value: {} (type: {})'.format(old_value, type(old_value).__name__))
          print('New value: {} (type: {})'.format(new_value, type(new_value).__name__))
          self.prompt('\nDo you want to save this?')
          if self.input_with_options(['Y', 'n'], 'n')[0] == 0:
            self.op_params.put(chosen_key, new_value)
            self.success('Saved!')
          else:
            self.info('Not saved!', sleep_time=0)
          return

  def change_param_list(self, old_value, param_info, chosen_key, parent_list=None, parent_idx=None):
    while True:
      self.info('Current value: {} (type: {})'.format(old_value, type(old_value).__name__), sleep_time=0)
      self.prompt('\nEnter index to edit (0 to {}), or -i to remove index, or +value to append value:'.format(len(old_value) - 1 if old_value else 0))

      append_val = False
      remove_idx = False
      choice_idx = input('>> ')

      if choice_idx == '':
        self.info('Exiting this parameter...', 0.5)
        return

      if isinstance(choice_idx, str):
        if choice_idx[0] == '-':
          remove_idx = True
        if choice_idx[0] == '+':
          append_val = True

      if append_val or remove_idx:
        choice_idx = choice_idx[1::]

      choice_idx = self.str_eval(choice_idx)
      is_list = isinstance(choice_idx, list)

      if not append_val and not (is_list or (isinstance(choice_idx, int) and choice_idx in range(len(old_value)))):
        self.error('Must be an integar within list range!')
        continue

      while True:
        if append_val or remove_idx or is_list:
          new_value = choice_idx
        else:
          if isinstance(old_value[choice_idx], list):
            self.change_param_list(old_value[choice_idx], param_info, chosen_key, parent_list=old_value, parent_idx=choice_idx)
            break

          self.info('Chosen index: {}'.format(choice_idx), sleep_time=0)
          self.info('Value: {} (type: {})'.format(old_value[choice_idx], type(old_value[choice_idx]).__name__), sleep_time=0)
          self.prompt('\nEnter your new value:')
          new_value = input('>> ').strip()
          new_value = self.str_eval(new_value)

        if new_value == '':
          self.info('Exiting this list item...', 0.5)
          break

        if not param_info.is_valid(new_value):
          self.error('The type of data you entered ({}) is not allowed with this parameter!'.format(type(new_value).__name__))
          break

        if append_val:
          old_value.append(new_value)
        elif remove_idx:
          del old_value[choice_idx]
        elif is_list:
          old_value = new_value
        else:
          old_value[choice_idx] = new_value

        if parent_list and parent_idx is not None:
          parent_list[parent_idx] = old_value

        save_value = parent_list if parent_list else old_value
        self.op_params.put(chosen_key, save_value)
        self.success('Saved {} with value: {}! (type: {})'.format(chosen_key, save_value, type(save_value).__name__), end='\n')
        break

  def cyan(self, msg, end=''):
    msg = self.str_color(msg, style='cyan')
    # print(msg, flush=True, end='\n' + end)
    return msg

  def prompt(self, msg, end=''):
    msg = self.str_color(msg, style='prompt')
    print(msg, flush=True, end='\n' + end)

  def info(self, msg, sleep_time=None, end=''):
    if sleep_time is None:
      sleep_time = self.sleep_time
    msg = self.str_color(msg, style='info')

    print(msg, flush=True, end='\n' + end)
    time.sleep(sleep_time)

  def error(self, msg, sleep_time=None, end='', surround=True):
    if sleep_time is None:
      sleep_time = self.sleep_time
    msg = self.str_color(msg, style='fail', surround=surround)

    print(msg, flush=True, end='\n' + end)
    time.sleep(sleep_time)

  def success(self, msg, sleep_time=None, end=''):
    if sleep_time is None:
      sleep_time = self.sleep_time
    msg = self.str_color(msg, style='success')

    print(msg, flush=True, end='\n' + end)
    time.sleep(sleep_time)

  def str_color(self, msg, style, surround=False):
    if style == 'success':
      style = COLORS.SUCCESS
    elif style == 'fail':
      style = COLORS.FAIL
    elif style == 'prompt':
      style = COLORS.PROMPT
    elif style == 'info':
      style = COLORS.INFO
    elif style == 'cyan':
      style = COLORS.CYAN

    if surround:
      msg = '{}--------\n{}\n{}--------{}'.format(style, msg, COLORS.ENDC + style, COLORS.ENDC)
    else:
      msg = '{}{}{}'.format(style, msg, COLORS.ENDC)

    return msg

  def input_with_options(self, options, default=None):
    """
    Takes in a list of options and asks user to make a choice.
    The most similar option list index is returned along with the similarity percentage from 0 to 1
    """
    user_input = input('[{}]: '.format('/'.join(options))).lower().strip()
    if not user_input:
      return default, 0.0
    sims = [self.str_sim(i.lower().strip(), user_input) for i in options]
    argmax = sims.index(max(sims))
    return argmax, sims[argmax]

  def str_eval(self, dat):
    dat = dat.strip()
    try:
      dat = ast.literal_eval(dat)
    except: # noqa E722 pylint: disable=bare-except
      if dat.lower() == 'none':
        dat = None
      elif dat.lower() in ['false', 'f']:
        dat = False
      elif dat.lower() in ['true', 't']:  # else, assume string
        dat = True
    return dat

  def delete_parameter(self):
    while True:
      self.prompt('Enter the name of the parameter to delete:')
      key = self.str_eval(input('>> '))

      if key == '':
        return
      if not isinstance(key, str):
        self.error('Input must be a string!')
        continue
      if key not in self.params:
        self.error("Parameter doesn't exist!")
        continue

      value = self.params.get(key)
      print('Parameter name: {}'.format(key))
      print('Parameter value: {} (type: {})'.format(value, type(value).__name__))
      self.prompt('Do you want to delete this?')

      if self.input_with_options(['Y', 'n'], default='n')[0] == 0:
        self.op_params.delete(key)
        self.success('Deleted!')
      else:
        self.info('Not deleted!')
      return

  def add_parameter(self):
    while True:
      self.prompt('Type the name of your new parameter:')
      key = self.str_eval(input('>> '))

      if key == '':
        return
      if not isinstance(key, str):
        self.error('Input must be a string!')
        continue

      self.prompt("Enter the data you'd like to save with this parameter:")
      value = input('>> ').strip()
      value = self.str_eval(value)

      print('Parameter name: {}'.format(key))
      print('Parameter value: {} (type: {})'.format(value, type(value).__name__))
      self.prompt('Do you want to save this?')

      if self.input_with_options(['Y', 'n'], default='n')[0] == 0:
        self.op_params.put(key, value)
        self.success('Saved!')
      else:
        self.info('Not saved!')
      return

  def sort_params(self, kv):
    return self.get_sort_key(kv[0])

  def get_sort_key(self, k):
    p = self.op_params.param_info(k)

    if not p.depends_on:
      return f'{1 if not len(p.children) else 0}{k}'
    else:
      s = ''
      while p.depends_on:
        s = f'{p.depends_on},{s}'
        p = self.op_params.param_info(p.depends_on)
      return f'{0}{s}{k}'

opEdit()