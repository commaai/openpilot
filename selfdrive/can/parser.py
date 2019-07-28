import os
import subprocess

can_dir = os.path.dirname(os.path.abspath(__file__))
libdbc_fn = os.path.join(can_dir, "libdbc.so")
subprocess.check_call(["make"], cwd=can_dir)

from selfdrive.can.parser_pyx import CANParser # pylint: disable=no-name-in-module, import-error
assert CANParser
