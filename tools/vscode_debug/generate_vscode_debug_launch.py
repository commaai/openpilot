import json
import os

from openpilot.selfdrive.manager.process_config import procs, PythonProcess, NativeProcess, DaemonProcess

default_args = {
  "process": ("arg1 arg2 --arg3"),
}

def next_backup_name(base_path, extension=".json"):
  """Finds the next available backup file name."""
  counter = 0
  while True:
    counter += 1
    new_name = f"{base_path}_old_{counter}{extension}"
    if not os.path.exists(new_name):
      return new_name

def write_with_backup(file_path, content, extension=".json"):
  """
  Writes content to a file, backing up the old file if it exists.
  :param file_path: Path to the file to write
  :param content: Content to write to the file
  :param extension: Extension of the file (used for backup)
  """
  # Backs up the existing file by renaming it
  if os.path.exists(file_path):
    backup_path = next_backup_name(os.path.splitext(file_path)[0], extension)
    os.rename(file_path, backup_path)
    print(f"Renamed existing {os.path.basename(file_path)} to {os.path.basename(backup_path)}")
  with open(file_path, 'w', encoding='utf_8') as file:
    if extension == ".json":
      json.dump(content, file, indent=4)
    else:
      file.write(content)
    print(f"Wrote to {file_path}")

def create_config(proc):
  input_id = default_args.get(proc.name, "argInput")
  if type(proc) == NativeProcess:
    path = proc.cwd
    executable = proc.cmdline[0].strip('./')
    return {
      "name": f"(gdb) {proc.name} Launch",
      "type": "cppdbg",
      "request": "launch",
      "program": f"${{workspaceFolder}}/{path}/{executable}",
      "args": [f"${{input:{input_id}}}"],
      "stopAtEntry": False,
      "cwd": f"${{workspaceFolder}}/{path}",
      "environment": [],
      "preLaunchTask": "prompt-for-env",
      "envFile": "${workspaceFolder}/.env",
      "externalConsole": False,
      "MIMode": "gdb",
      "setupCommands": [
        {"description": "Enable pretty-printing for gdb", "text": "-enable-pretty-printing", "ignoreFailures": True},
        {"description": "Set Disassembly Flavor to Intel", "text": "-gdb-set disassembly-flavor intel", "ignoreFailures": True},
        {"description": "ignore SIGUSR2 signal", "text": "handle SIGUSR2 nostop noprint pass"}
      ]
    }
  elif isinstance(proc, (PythonProcess, DaemonProcess)):
    path: str = proc.module
    program_path = path.replace('.', '/')
    return {
      "name": f"Python: {proc.name}",
      "type": "python",
      "request": "launch",
      "program": f"${{workspaceFolder}}/{program_path}.py",
      "args": [f"${{input:{input_id}}}"],
      "env": {},
      "preLaunchTask": "prompt-for-env",
      "envFile": "${workspaceFolder}/.env",
      "justMyCode" : False
    }

def generate_tasks_json():
  tasks_content = {
    "version": "2.0.0",
    "tasks": [
      {
        "label": "prompt-for-env",
        "type": "shell",
        "command": "python",
        "args": [
          "${workspaceFolder}/tools/vscode_debug/env_prompt.py"
        ],
        "presentation": {
          "echo": True,
          "reveal": "always",
          "focus": True,
          "panel": "shared",
          "showReuseMessage": False,
          "clear": False
        }
      }
    ]
  }
  tasks_path = '.vscode/tasks.json'
  write_with_backup(tasks_path, tasks_content)

def generate_launch_json():
  inputs = [
    {
      "id": proc_name,
      "type": "promptString",
      "description": f"Enter the arguments for {proc_name}:",
      "default": args
    }
    for proc_name, args in default_args.items()
  ]

  # Add a generic input for processes without specific defaults
  inputs.append({
    "id": "argInput",
    "type": "promptString",
    "description": "Enter the arguments for the script:",
    "default": ""
  })

  launch_json = {
    "version": "0.2.0",
    "configurations": [create_config(proc) for proc in procs],
    "inputs": inputs
  }

  launch_json_path = ".vscode/launch.json"
  write_with_backup(launch_json_path, launch_json)

if __name__ == "__main__":
  generate_launch_json()
  generate_tasks_json()
