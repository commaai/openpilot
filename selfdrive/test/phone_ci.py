#!/usr/bin/env python3
import paramiko  # pylint: disable=import-error
import os
import sys
import re
import time
import socket

TEST_DIR = "/data/openpilotci"

def run_test(name, test_func):
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  key_file = open(os.path.join(os.path.dirname(__file__), "../../tools/ssh/key/id_rsa"))
  key = paramiko.RSAKey.from_private_key(key_file)

  print("SSH to phone {}".format(name))

  # Try connecting for one minute
  t_start = time.time()
  while True:
    try:
      ssh.connect(hostname=name, port=8022, pkey=key, timeout=10)
    except (paramiko.ssh_exception.SSHException, socket.timeout, paramiko.ssh_exception.NoValidConnectionsError):
      print("Connection failed")
      if time.time() - t_start > 60:
        raise
    else:
      break
    time.sleep(1)

  conn = ssh.invoke_shell()
  branch = os.environ['GIT_BRANCH']
  commit = os.environ.get('GIT_COMMIT', branch)

  conn.send("uname -a\n")

  conn.send(f"cd {TEST_DIR}\n")
  conn.send("git reset --hard\n")
  conn.send("git fetch origin\n")
  conn.send("git checkout %s\n" % commit)
  conn.send("git clean -xdf\n")
  conn.send("git submodule update --init\n")
  conn.send("git submodule foreach --recursive git reset --hard\n")
  conn.send("git submodule foreach --recursive git clean -xdf\n")
  conn.send("echo \"git took $SECONDS seconds\"\n")

  test_func(conn)

  conn.send('echo "RESULT:" $?\n')
  conn.send("exit\n")
  return conn

def test_modeld(conn):
  conn.send(f"cd selfdrive/test/process_replay && PYTHONPATH={TEST_DIR} ./camera_replay.py\n")

if __name__ == "__main__":
  eon_name = os.environ.get('eon_name', None)

  conn = run_test(eon_name, test_modeld)

  dat = b""

  while True:
    recvd = conn.recv(4096)
    if len(recvd) == 0:
      break

    dat += recvd
    sys.stdout.buffer.write(recvd)
    sys.stdout.flush()

  returns = re.findall(rb'^RESULT: (\d+)', dat[-1024:], flags=re.MULTILINE)
  sys.exit(int(returns[0]))
