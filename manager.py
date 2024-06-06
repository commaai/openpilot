import time
import sys
import signal
import multiprocessing
from python_process import main
from setproctitle import setproctitle

from system.manager.process_config import managed_processes


def launcher():
  setproctitle('python_process')
  main()


if __name__ == '__main__':
  def signal_handler(sig, frame):
    print('(fake) manager got signal', sig)
    sys.exit()

  # SystemExit on sigterm
  signal.signal(signal.SIGTERM, signal_handler)

  p = multiprocessing.Process(target=launcher, name='python_process')
  try:
    p.start()
    # managed_processes['python_process'].start()
    input('Press enter to stop the process')
  finally:
    p.kill()
    # managed_processes['python_process'].stop()
