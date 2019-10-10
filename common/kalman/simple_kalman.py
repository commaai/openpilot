# pylint: skip-file
import os
import subprocess

kalman_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.check_call(["make", "simple_kalman_impl.so"], cwd=kalman_dir)

from .simple_kalman_impl import KF1D as KF1D
# Silence pyflakes
assert KF1D
