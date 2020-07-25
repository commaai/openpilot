#!/usr/bin/env python3
import paramiko  # pylint: disable=import-error
import os
import sys
import re
import time
import socket


SOURCE_DIR = "/data/openpilot_source/"
TEST_DIR = "/data/openpilot/"

def run_on_phone(test_cmd):

  eon_ip = os.environ.get('eon_ip', None)
  if eon_ip is None:
    raise Exception("'eon_ip' not set")

  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  key_file = open(os.path.join(os.path.dirname(__file__), "id_rsa"))
  key = paramiko.RSAKey.from_private_key(key_file)

  print("SSH to phone at {}".format(eon_ip))

  # try connecting for one minute
  t_start = time.time()
  while True:
    try:
      ssh.connect(hostname=eon_ip, port=8022, pkey=key, timeout=10)
    except (paramiko.ssh_exception.SSHException, socket.timeout, paramiko.ssh_exception.NoValidConnectionsError):
      print("Connection failed")
      if time.time() - t_start > 60:
        raise
    else:
      break
    time.sleep(1)

  branch = os.environ['GIT_BRANCH']
  commit = os.environ.get('GIT_COMMIT', branch)

  conn = ssh.invoke_shell()

  # pass in all environment variables prefixed with 'CI_'
  for k, v in os.environ.items():
    if k.startswith("CI_") or k in ["GIT_BRANCH", "GIT_COMMIT"]:
      conn.send(f"export {k}='{v}'\n")
  conn.send("export CI=1\n")

  # clear scons cache dirs that haven't been written to in one day
  conn.send("cd /tmp && find -name 'scons_cache_*' -type d -maxdepth 1 -mtime 1 -exec rm -rf '{}' \\;\n")

  # set up environment
  conn.send(f"cd {SOURCE_DIR}\n")
  conn.send("git reset --hard\n")
  conn.send("git fetch origin\n")
  conn.send("find . -maxdepth 1 -not -path './.git' -not -name '.' -not -name '..' -exec rm -rf '{}' \\;\n")
  conn.send(f"git reset --hard {commit}\n")
  conn.send(f"git checkout {commit}\n")
  conn.send("git clean -xdf\n")
  conn.send("git submodule update --init\n")
  conn.send("git submodule foreach --recursive git reset --hard\n")
  conn.send("git submodule foreach --recursive git clean -xdf\n")
  conn.send('echo "git took $SECONDS seconds"\n')

  conn.send(f"rsync -a --delete {SOURCE_DIR} {TEST_DIR}\n")

  # run the test
  conn.send(test_cmd + "\n")

  # get the result and print it back out
  conn.send('echo "RESULT:" $?\n')
  conn.send("exit\n")

  dat = b""
  conn.settimeout(240)

  while True:
    try:
      recvd = conn.recv(4096)
    except socket.timeout:
      print("connection to phone timed out")
      sys.exit(1)

    if len(recvd) == 0:
      break

    dat += recvd
    sys.stdout.buffer.write(recvd)
    sys.stdout.flush()

  return_code = int(re.findall(rb'^RESULT: (\d+)', dat[-1024:], flags=re.MULTILINE)[0])
  sys.exit(return_code)


if __name__ == "__main__":
  run_on_phone(sys.argv[1])
