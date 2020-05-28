#!/usr/bin/env python2
import paramiko # pylint: disable=import-error
import os
import sys
import re
import time
import socket


def start_build(name):
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  key_file = open(os.path.join(os.path.dirname(__file__), "id_rsa_public"))
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

  conn.send('uname -a\n')

  conn.send('cd /data/openpilot_source\n')
  conn.send("git reset --hard\n")
  conn.send("git fetch origin\n")
  conn.send("git checkout %s\n" % commit)
  conn.send("git clean -xdf\n")
  conn.send("git submodule update --init\n")
  conn.send("git submodule foreach --recursive git reset --hard\n")
  conn.send("git submodule foreach --recursive git clean -xdf\n")
  conn.send("echo \"git took $SECONDS seconds\"\n")

  push = "PUSH=master-ci" if branch == "master" else ""

  conn.send("%s /data/openpilot_source/release/build_devel.sh\n" % push)
  conn.send('echo "RESULT:" $?\n')
  conn.send("exit\n")
  return conn


if __name__ == "__main__":
  eon_name = os.environ.get('eon_name', None)

  conn = start_build(eon_name)

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
