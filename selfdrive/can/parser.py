import os
import subprocess

can_dir = os.path.dirname(os.path.abspath(__file__))
libdbc_fn = os.path.join(can_dir, "libdbc.so")
subprocess.check_call(["make", "-j3"], cwd=can_dir)  # don't use all the cores to avoid overheating

from selfdrive.can.parser_pyx import CANParser # pylint: disable=no-name-in-module, import-error
assert CANParser
