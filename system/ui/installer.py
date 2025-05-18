#!/usr/bin/env python3
import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import files
import pyray as rl

from cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.common.time_helpers import system_time_valid
from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.label import gui_label, gui_text_box

NETWORK_TYPE = log.DeviceState.NetworkType

# TODO: can we patch these strings
GIT_URL = "https://github.com/commaai/openpilot.git"
BRANCH = "master"

GIT_SSH_URL = "git@github.com:commaai/openpilot.git"
CONTINUE_PATH = "/data/continue.sh"

CACHE_PATH = "/data/openpilot.cache"

INSTALL_PATH = "/data/openpilot"
TMP_INSTALL_PATH = "/data/tmppilot"


class InstallStage(IntEnum):
  CLONING = 0
  RECEIVING_OBJECTS = 1
  RESOLVING_DELTAS = 2
  UPDATING_FILES = 3
  FINALIZING = 4
  COMPLETED = 5


# Stage weights for progress calculation
STAGE_WEIGHTS = [
  ("Receiving objects: ", 91),
  ("Resolving deltas: ", 2),
  ("Updating files: ", 7),
]


@dataclass
class StateInstalling:
  stage = InstallStage.CLONING
  progress = 0


@dataclass
class StateDone:
  pass


@dataclass
class StateError:
  error: str


class Installer:
  def __init__(self):
    self.state = StateInstalling()
    self.install_thread = threading.Thread(target=self.do_install, daemon=True)
    self.install_thread.start()

  def run_command(self, cmd):
    try:
      subprocess.run(cmd, check=True, shell=True)
      return True
    except subprocess.CalledProcessError as e:
      self.state = StateError(f"Command failed: {cmd}\nError: {e}")
      return False

  def run_git_command(self, args, cwd=None):
    proc = subprocess.Popen(["git"] + args, cwd=cwd, stderr=subprocess.PIPE, universal_newlines=True)
    self.read_progress(proc)
    ret = proc.wait()
    return ret == 0

  def read_progress(self, proc: subprocess.Popen):
    if proc.stderr is None:
      return
    base = 0
    for line in io.TextIOWrapper(proc.stderr, encoding="utf-8"):
      for prefix, weight in STAGE_WEIGHTS:
        if line.startswith(prefix):
          perc = line.split(prefix)[1].split("%")[0]
          p = base + int(float(perc) / 100.0 * weight)
          self.progress = p
          break
        base += weight

  def do_install(self):
    # wait for valid time
    while not system_time_valid():
      time.sleep(0.5)
      cloudlog.debug("Waiting for valid time")

    # cleanup previous install attempts
    if not self.run_command(f"rm -rf {TMP_INSTALL_PATH} {INSTALL_PATH}"):
      return

    # do the install
    if os.path.exists(CACHE_PATH):
      self.cached_fetch(CACHE_PATH)
    else:
      self.fresh_clone()

  def fresh_clone(self):
    cloudlog.debug("Doing fresh clone")
    success = self.run_git_command(["clone", "--progress", GIT_URL, "-b", BRANCH, "--depth=1", "--recurse-submodules", TMP_INSTALL_PATH])
    if not success:
      return

    self.finish_install()

  def cached_fetch(self, cache: str):
    cloudlog.debug(f"Fetching with cache: {cache}")
    if not self.run_command(f"cp -rp {cache} {TMP_INSTALL_PATH}"):
      return

    os.chdir(TMP_INSTALL_PATH)
    if not self.run_command(f"git remote set-branches --add origin {BRANCH}"):
      return

    self.progress = 10
    success = self.run_git_command(["fetch", "--progress", "origin", BRANCH], cwd=TMP_INSTALL_PATH)
    if not success:
      return

    self.finish_install()

  def finish_install(self):
    self.progress = 100
    self.stage = InstallStage.FINALIZING

    # Ensure correct branch
    os.chdir(TMP_INSTALL_PATH)
    if not self.run_command(f"git checkout {BRANCH}"):
      return
    if not self.run_command(f"git reset --hard origin/{BRANCH}"):
      return
    if not self.run_command("git submodule update --init"):
      return

    # move into place
    if not self.run_command(f"mv {TMP_INSTALL_PATH} {INSTALL_PATH}"):
      return

    # Set up SSH keys for internal builds if needed
    if os.environ.get('INTERNAL', '0') == '1':
      if not self.run_command("mkdir -p /data/params/d/"):
        return

      # https://github.com/commaci2.keys
      ssh_keys = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMX2kU8eBZyEWmbq0tjMPxksWWVuIV/5l64GabcYbdpI"
      params = {
        "SshEnabled": "1",
        "RecordFrontLock": "1",
        "GithubSshKeys": ssh_keys,
      }

      for key, value in params.items():
        with open(f"/data/params/d/{key}", "w") as param:
          param.write(value)

      if not self.run_command(
        f"cd {INSTALL_PATH} && "
        f"git remote set-url origin --push {GIT_SSH_URL} && "
        f"git config --replace-all remote.origin.fetch \"+refs/heads/*:refs/remotes/origin/*\""
      ):
        return

    # Copy continue.sh script
    with files("openpilot.selfdrive.ui").joinpath("installer/continue_openpilot.sh").open("rb") as rf:
      with open("/data/continue.sh.new", "wb") as wf:
        wf.write(rf.read())

    if not self.run_command("chmod +x /data/continue.sh.new"):
      return
    if not self.run_command(f"mv /data/continue.sh.new {CONTINUE_PATH}"):
      return

    self.stage = InstallStage.COMPLETED
    self.state = StateDone()

    # Exit after 60 seconds
    threading.Timer(60, sys.exit).start()

  def render(self, rect: rl.Rectangle):
    match self.state:
      case StateInstalling():
        self.render_installing(rect)
      case StateDone():
        self.render_done(rect)
      case StateError(error):
        self.render_error(rect, error)

  def render_installing(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x + 150, rect.y + 290, rect.width - 300, 90)
    gui_label(title_rect, "Installing...", 90, font_weight=FontWeight.SEMI_BOLD)

    bar_rect = rl.Rectangle(rect.x + 150, rect.y + 290 + 170, rect.width - 300, 72)
    rl.draw_rectangle_rounded(bar_rect, 0.05, 10, rl.Color(41, 41, 41, 255))

    if self.progress > 0:
      chunk_width = (rect.width - 300) * self.progress / 100
      chunk_rect = rl.Rectangle(rect.x + 150, rect.y + 290 + 170, chunk_width, 72)
      rl.draw_rectangle_rounded(chunk_rect, 0.05, 10, rl.Color(54, 77, 239, 255))

    percent_rect = rl.Rectangle(rect.x + 150, rect.y + 290 + 170 + 102, rect.width - 300, 70)
    gui_label(percent_rect, f"{self.progress}%", 70, font_weight=FontWeight.LIGHT)

  def render_done(self, rect: rl.Rectangle):
    title_rect = rl.Rectangle(rect.x + 150, rect.y + 290, rect.width - 300, 90)
    gui_label(title_rect, "Installation Complete", 90, font_weight=FontWeight.SEMI_BOLD)

    message_rect = rl.Rectangle(rect.x + 150, rect.y + 290 + 170, rect.width - 300, 72)
    gui_label(message_rect, "openpilot will start soon...", 70, font_weight=FontWeight.LIGHT)

  def render_error(self, rect: rl.Rectangle, error: str):
    title_rect = rl.Rectangle(rect.x + 150, rect.y + 200, rect.width - 300, 90)
    gui_label(title_rect, "Installation Error", 90, rl.Color(255, 89, 79, 255), font_weight=FontWeight.SEMI_BOLD)

    error_rect = rl.Rectangle(rect.x + 150, rect.y + 200 + 120, rect.width - 300, rect.height - 400)
    gui_text_box(error_rect, error, 50)

    button_rect = rl.Rectangle(rect.x + 150, rect.height - 150, rect.width - 300, 100)
    if gui_button(button_rect, "Reboot Device", button_style=ButtonStyle.PRIMARY):
      HARDWARE.reboot()


def main():
  try:
    gui_app.init_window("Installer")
    installer = Installer()
    for _ in gui_app.render():
      installer.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
  except Exception as e:
    print(f"Installer error: {e}")
  finally:
    gui_app.close()


if __name__ == "__main__":
  main()
