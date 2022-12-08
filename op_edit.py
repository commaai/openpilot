#!/usr/bin/env python3
import time
from common.op_params import opParams
import ast
import difflib
from common.colors import COLORS


class opEdit:  # use by running `python /data/openpilot/op_edit.py`
  def __init__(self):
    self.op_params = opParams()
    self.params = None
    self.sleep_time = 0.5
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
      self.success('\nWelcome to the {}opParams{} command line editor!'.format(COLORS.CYAN, COLORS.SUCCESS), sleep_time=0)
      self.prompt('Would you like to add your Discord username for easier crash debugging for the fork owner?')
      self.prompt('Your username is only used for reaching out if a crash occurs.')

      username_choice = self.input_with_options(['Y', 'N', 'don\'t ask again'], default='n')[0]
      if username_choice == 0:
        self.prompt('Enter a unique identifer/Discord username:')
        username = ''
        while username == '':
          username = input('>> ').strip()
        self.op_params.put('username', username)
        self.username = username
        self.success('Thanks! Saved your username\n'
                     'Edit the \'username\' parameter at any time to update', sleep_time=1.5)
      elif username_choice == 2:
        self.op_params.put('username', False)
        self.info('Got it, bringing you into opEdit\n'
                  'Edit the \'username\' parameter at any time to update', sleep_time=1.0)
    else:
      self.success('\nWelcome to the {}opParams{} command line editor, {}!'.format(COLORS.CYAN, COLORS.SUCCESS, self.username), sleep_time=0)

    self.run_loop()

  def run_loop(self):
    while True:
      if not self.live_tuning:
        self.info('Here are all your parameters:', sleep_time=0)
        self.info('(non-static params update while driving)', end='\n', sleep_time=0)
      else:
        self.info('Here are your live parameters:', sleep_time=0)
        self.info('(changes take effect within a second)', end='\n', sleep_time=0)
      self.params = self.op_params.get(force_update=True)
      if self.live_tuning:  # only display live tunable params
        self.params = {k: v for k, v in self.params.items() if self.op_params.fork_params[k].live}

      values_list = []
      for k, v in self.params.items():
        if len(str(v)) < 20:
          v = self.color_from_type(v)
        else:
          v = '{} ... {}'.format(str(v)[:30], str(v)[-15:])
        values_list.append(v)

      static = [COLORS.INFO + '(static)' + COLORS.ENDC if self.op_params.fork_params[k].static else '' for k in self.params]

      to_print = []
      blue_gradient = [33, 39, 45, 51, 87]
      for idx, param in enumerate(self.params):
        line = '{}. {}: {}  {}'.format(idx + 1, param, values_list[idx], static[idx])
        if idx == self.last_choice and self.last_choice is not None:
          line = COLORS.OKGREEN + line
        else:
          _color = blue_gradient[min(round(idx / len(self.params) * len(blue_gradient)), len(blue_gradient) - 1)]
          line = COLORS.BASE(_color) + line
        to_print.append(line)

      extras = {'l': ('Toggle live params', COLORS.WARNING),
                'e': ('Exit opEdit', COLORS.PINK)}

      to_print += ['---'] + ['{}. {}'.format(ext_col + e, ext_txt + COLORS.ENDC) for e, (ext_txt, ext_col) in extras.items()]
      print('\n'.join(to_print))
      self.prompt('\nChoose a parameter to edit (by index or name):')

      choice = input('>> ').strip().lower()
      parsed, choice = self.parse_choice(choice, len(to_print) - len(extras))
      if parsed == 'continue':
        continue
      elif parsed == 'change':
        self.last_choice = choice
        self.change_parameter(choice)
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

    if choice in ['l', 'live']:  # live tuning mode
      return 'live', choice
    elif choice in ['exit', 'e', '']:
      self.error('Exiting opEdit!', sleep_time=0)
      return 'exit', choice
    else:  # find most similar param to user's input
      param_sims = [(idx, self.str_sim(choice, param.lower())) for idx, param in enumerate(self.params)]
      param_sims = [param for param in param_sims if param[1] > 0.33]
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
      param_info = self.op_params.fork_params[chosen_key]

      old_value = self.params[chosen_key]
      if not param_info.static:
        self.info2('Chosen parameter: {}{} (live!)'.format(chosen_key, COLORS.BASE(207)), sleep_time=0)
      else:
        self.info2('Chosen parameter: {}{} (static)'.format(chosen_key, COLORS.BASE(207)), sleep_time=0)

      to_print = []
      if param_info.has_description:
        to_print.append(COLORS.OKGREEN + '>>  Description: {}'.format(param_info.description.replace('\n', '\n  > ')) + COLORS.ENDC)
      if param_info.static:
        to_print.append(COLORS.WARNING + '>>  A reboot is required for changes to this parameter!' + COLORS.ENDC)
      if not param_info.static and not param_info.live:
        to_print.append(COLORS.WARNING + '>>  Changes take effect within 10 seconds for this parameter!' + COLORS.ENDC)
      if param_info.has_allowed_types:
        to_print.append(COLORS.RED + '>>  Allowed types: {}'.format(', '.join([at.__name__ for at in param_info.allowed_types])) + COLORS.ENDC)
      to_print.append(COLORS.WARNING + '>>  Default value: {}'.format(self.color_from_type(param_info.default_value)) + COLORS.ENDC)

      if to_print:
        print('\n{}\n'.format('\n'.join(to_print)))

      if param_info.is_list:
        self.change_param_list(old_value, param_info, chosen_key)  # TODO: need to merge the code in this function with the below to reduce redundant code
        return

      self.info('Current value: {}{} (type: {})'.format(self.color_from_type(old_value), COLORS.INFO, type(old_value).__name__), sleep_time=0)

      while True:
        self.prompt('\nEnter your new value (enter to exit):')
        new_value = input('>> ').strip()
        if new_value == '':
          self.info('Exiting this parameter...\n')
          return

        new_value = self.str_eval(new_value)
        if not param_info.is_valid(new_value):
          self.error('The type of data you entered ({}) is not allowed with this parameter!'.format(type(new_value).__name__))
          continue

        if not param_info.static:  # stay in live tuning interface
          self.op_params.put(chosen_key, new_value)
          self.success('Saved {} with value: {}{}! (type: {})'.format(chosen_key, self.color_from_type(new_value), COLORS.SUCCESS, type(new_value).__name__))
        else:  # else ask to save and break
          self.warning('\nOld value: {}{} (type: {})'.format(self.color_from_type(old_value), COLORS.WARNING, type(old_value).__name__))
          self.success('New value: {}{} (type: {})'.format(self.color_from_type(new_value), COLORS.OKGREEN, type(new_value).__name__), sleep_time=0)
          self.prompt('\nDo you want to save this?')
          if self.input_with_options(['Y', 'N'], 'N')[0] == 0:
            self.op_params.put(chosen_key, new_value)
            self.success('Saved!')
          else:
            self.info('Not saved!')
          return

  def change_param_list(self, old_value, param_info, chosen_key):
    while True:
      self.info('Current value: {} (type: {})'.format(old_value, type(old_value).__name__), sleep_time=0)
      self.prompt('\nEnter index to edit (0 to {}):'.format(len(old_value) - 1))
      choice_idx = self.str_eval(input('>> '))
      if choice_idx == '':
        self.info('Exiting this parameter...')
        return

      if not isinstance(choice_idx, int) or choice_idx not in range(len(old_value)):
        self.error('Must be an integar within list range!')
        continue

      while True:
        self.info('Chosen index: {}'.format(choice_idx), sleep_time=0)
        self.info('Value: {} (type: {})'.format(old_value[choice_idx], type(old_value[choice_idx]).__name__), sleep_time=0)
        self.prompt('\nEnter your new value:')
        new_value = input('>> ').strip()
        if new_value == '':
          self.info('Exiting this list item...')
          break

        new_value = self.str_eval(new_value)
        if not param_info.is_valid(new_value):
          self.error('The type of data you entered ({}) is not allowed with this parameter!'.format(type(new_value).__name__))
          continue

        old_value[choice_idx] = new_value

        self.op_params.put(chosen_key, old_value)
        self.success('Saved {} with value: {}{}! (type: {})'.format(chosen_key, self.color_from_type(new_value), COLORS.SUCCESS, type(new_value).__name__), end='\n')
        break

  def color_from_type(self, v):
    v_color = ''
    if type(v) in self.type_colors:
      v_color = self.type_colors[type(v)]
      if isinstance(v, bool):
        v_color = v_color[v]
    v = '{}{}{}'.format(v_color, v, COLORS.ENDC)
    return v

  def cyan(self, msg, end=''):
    msg = self.str_color(msg, style='cyan')
    # print(msg, flush=True, end='\n' + end)
    return msg

  def prompt(self, msg, end=''):
    msg = self.str_color(msg, style='prompt')
    print(msg, flush=True, end='\n' + end)

  def warning(self, msg, end=''):
    msg = self.str_color(msg, style='warning')
    print(msg, flush=True, end='\n' + end)

  def info(self, msg, sleep_time=None, end=''):
    if sleep_time is None:
      sleep_time = self.sleep_time
    msg = self.str_color(msg, style='info')

    print(msg, flush=True, end='\n' + end)
    time.sleep(sleep_time)

  def info2(self, msg, sleep_time=None, end=''):
    if sleep_time is None:
      sleep_time = self.sleep_time
    msg = self.str_color(msg, style=86)

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

  @staticmethod
  def str_color(msg, style, surround=False):
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
    elif style == 'warning':
      style = COLORS.WARNING
    elif isinstance(style, int):
      style = COLORS.BASE(style)

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
    except:
      if dat.lower() == 'none':
        dat = None
      elif dat.lower() == 'false':
        dat = False
      elif dat.lower() == 'true':  # else, assume string
        dat = True
    return dat


opEdit()
