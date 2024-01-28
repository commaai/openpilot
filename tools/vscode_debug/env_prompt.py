import os
import readline

def get_existing_env_vars(env_file_path):
  existing_env_vars = {}
  if os.path.exists(env_file_path):
    with open(env_file_path, "r", encoding='utf_8') as file:
        print("Existing debug environment variables:")
        for line in file:
          line = line.strip()
          if line:
            key, value = line.split('=', 1) if "=" in line else [line, ""]
            existing_env_vars[key] = value
            print(line)
  return existing_env_vars

def main():
  existing_env_vars = get_existing_env_vars(".env")
  # Set up readline history with existing environment variables
  for var in existing_env_vars:
      readline.add_history(f"{var}={existing_env_vars[var]}")

  while True:
    env_name = input("Enter the environment variable name to add, modify or delete it. You can also assign it now with VAR=<val> or press enter to skip: ")
    if not env_name:
      break
    if "=" in env_name:
      key, value = env_name.split('=', 1)
      existing_env_vars[key] = value
      continue

    if env_name in existing_env_vars:
      action = input(f"Reuse(default), Modify, or Delete the value for {env_name} ({existing_env_vars[env_name]})? [r/m/d]: ")
      if action.lower() in ['', 'r', 'reuse']:
        continue
      elif action.lower() in ['d', 'delete']:
        del existing_env_vars[env_name]
        continue
      elif action.lower() in ['m', 'modify']:
        pass
      else:
        print("Invalid input")
        continue

    env_value = input(f"Enter the value for {env_name}: ")
    existing_env_vars[env_name] = env_value

  with open(".env", "w", encoding='utf_8') as file:
    for name, value in existing_env_vars.items():
      file.write(f"{name}={value}\n")

if __name__ == "__main__":
  main()
