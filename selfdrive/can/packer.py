# pylint: skip-file
import os
import subprocess

can_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(["make", "-j3", "packer_impl.so"], cwd=can_dir)  # don't use all the cores to avoid overheating

from selfdrive.can.packer_impl import CANPacker
assert CANPacker
