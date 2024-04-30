import re
import inspect

def find_closing_bracket(string, opening_index, opening_bracket, closing_bracket):
  open_brackets = 1
  for i in range(opening_index + 1, len(string)):
      if(string[i] == opening_bracket):
          open_brackets += 1
      elif (string[i] == closing_bracket):
          open_brackets -= 1
          if(open_brackets == 0):
              return i
  return -1

def check_if_enum(obj):
  return obj.find(':') > 0

def left_value_to_str(lv):
  e_type, name = lv.split('.')
  name_list = name.split('|')
  str_value = e_type + "." + name_list[0]
  for n in name_list[1:]:
      temp = " | " + e_type + "." + n
      str_value += temp
  return str_value

def change_config(config):
  new_config = config
  i = 0
  while(i < len(config)):
      if(config[i] == '<'):
          start_index = i
          end_index = find_closing_bracket(config, i, '<', '>')
          obj = config[start_index:end_index+1]
          if(check_if_enum(obj)):
              left_value = obj.split(':')[0][1:]
              new_config = new_config.replace(obj, left_value_to_str(left_value))
          i += len(obj)
      i += 1

  return new_config

def indent2str(n):
  return " " * n

def format_config(config):
  formatted_config = ""
  opening_brackets = ['(', '[', '{']
  closing_brackets = [')', ']', '}']
  bracket_counter = -1
  for char in config:
      if char in opening_brackets:
          bracket_counter += 1
          if bracket_counter == 0:
              formatted_config += f'{char}\n{indent2str(4)}'
              continue
      elif char in closing_brackets:
          bracket_counter -= 1
          if bracket_counter == -1:
              formatted_config += f'\n{indent2str(2)}{char}'
              continue
      elif char == ',' and bracket_counter == 0:
          formatted_config += f'{char}\n{indent2str(3)}'
          continue
      formatted_config += char

  return formatted_config

def save_config(platform, config):
  file_path = inspect.getsourcefile(type(platform))
  with open(file_path, 'r') as f:
      tokens = f.read()

  regex = rf"{platform} = ([a-zA-Z0-9]*)PlatformConfig\("
  match = re.search(regex, tokens)
  start_index = match.start()
  opening_bracket_index = match.end()
  end_index = find_closing_bracket(tokens, opening_bracket_index, '(', ')')
  constructor_index = start_index + len(platform) + 3
  new_config = change_config(str(config))
  formatted_config = format_config(new_config)
  old_config = tokens[constructor_index:end_index+1]
  tokens = tokens.replace(old_config, formatted_config)

  with open(file_path, 'w+') as f:
      f.write(tokens)
