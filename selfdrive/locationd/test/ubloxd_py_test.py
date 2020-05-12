import sys
import os

from selfdrive.locationd.test.ublox import UBloxMessage
from selfdrive.locationd.test.ubloxd import gen_solution, gen_raw, gen_nav_data
from common import realtime


def mkdirs_exists_ok(path):
  try:
    os.makedirs(path)
  except OSError:
    if not os.path.isdir(path):
      raise


def parser_test(fn, prefix):
  nav_frame_buffer = {}
  nav_frame_buffer[0] = {}
  for i in range(1, 33):
    nav_frame_buffer[0][i] = {}

  if not os.path.exists(prefix):
    print('Prefix invalid')
    sys.exit(-1)

  with open(fn, 'rb') as f:
    i = 0
    saved_i = 0
    msg = UBloxMessage()
    while True:
      n = msg.needed_bytes()
      b = f.read(n)
      if not b:
        break
      msg.add(b)
      if msg.valid():
        i += 1
        if msg.name() == 'NAV_PVT':
          sol = gen_solution(msg)
          sol.logMonoTime = int(realtime.sec_since_boot() * 1e9)
          with open(os.path.join(prefix, str(saved_i)), 'wb') as f1:
            f1.write(sol.to_bytes())
            saved_i += 1
        elif msg.name() == 'RXM_RAW':
          raw = gen_raw(msg)
          raw.logMonoTime = int(realtime.sec_since_boot() * 1e9)
          with open(os.path.join(prefix, str(saved_i)), 'wb') as f1:
            f1.write(raw.to_bytes())
            saved_i += 1
        elif msg.name() == 'RXM_SFRBX':
          nav = gen_nav_data(msg, nav_frame_buffer)
          if nav is not None:
            nav.logMonoTime = int(realtime.sec_since_boot() * 1e9)
            with open(os.path.join(prefix, str(saved_i)), 'wb') as f1:
              f1.write(nav.to_bytes())
              saved_i += 1

        msg = UBloxMessage()
        msg.debug_level = 0
    print('Parsed {} msgs'.format(i))
    print('Generated {} cereal events'.format(saved_i))


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print('Format: ubloxd_py_test.py file_path prefix')
    sys.exit(0)

  fn = sys.argv[1]
  if not os.path.isfile(fn):
    print('File path invalid')
    sys.exit(0)

  prefix = sys.argv[2]
  mkdirs_exists_ok(prefix)
  parser_test(fn, prefix)
