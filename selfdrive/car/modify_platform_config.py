import inspect
import re

def save_config(platform, config):
  file_path = inspect.getsourcefile(type(platform))
  with open(file_path, 'r') as f:
      tokens = f.read()

  regex = rf'{platform} = ([a-zA-Z0-9]*)PlatformConfig\((?:[^()]+|\((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*\))*\)'
  match = re.search(regex, tokens)
  old_config = tokens[match.start():match.end()+1]
  new_config = f'{platform} = {str(config)}\n'
  tokens = tokens.replace(old_config, new_config)

  with open(file_path, 'w+') as f:
      f.write(tokens)
