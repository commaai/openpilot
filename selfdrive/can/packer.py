# pylint: skip-file
import os
import subprocess

can_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(["make", "packer_impl.so"], cwd=can_dir)

from selfdrive.can.packer_impl import CANPacker
assert CANPacker
