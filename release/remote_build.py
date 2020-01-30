#!/usr/bin/env python2
import paramiko
import os
import sys
import re
import collections
import StringIO
import time
import socket


SSH_KEY = StringIO.StringIO("""-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC+iXXq30Tq+J5N
Kat3KWHCzcmwZ55nGh6WggAqECa5CasBlM9VeROpVu3beA+5h0MibRgbD4DMtVXB
t6gEvZ8nd04E7eLA9LTZyFDZ7SkSOVj4oXOQsT0GnJmKrASW5KslTWqVzTfo2XCt
Z+004ikLxmyFeBO8NOcErW1pa8gFdQDToH9FrA7kgysic/XVESTOoe7XlzRoe/eZ
acEQ+jtnmFd21A4aEADkk00Ahjr0uKaJiLUAPatxs2icIXWpgYtfqqtaKF23wSt6
1OTu6cAwXbOWr3m+IUSRUO0IRzEIQS3z1jfd1svgzSgSSwZ1Lhj4AoKxIEAIc8qJ
rO4uymCJAgMBAAECggEBAISFevxHGdoL3Z5xkw6oO5SQKO2GxEeVhRzNgmu/HA+q
x8OryqD6O1CWY4037kft6iWxlwiLOdwna2P25ueVM3LxqdQH2KS4DmlCx+kq6FwC
gv063fQPMhC9LpWimvaQSPEC7VUPjQlo4tPY6sTTYBUOh0A1ihRm/x7juKuQCWix
Cq8C/DVnB1X4mGj+W3nJc5TwVJtgJbbiBrq6PWrhvB/3qmkxHRL7dU2SBb2iNRF1
LLY30dJx/cD73UDKNHrlrsjk3UJc29Mp4/MladKvUkRqNwlYxSuAtJV0nZ3+iFkL
s3adSTHdJpClQer45R51rFDlVsDz2ZBpb/hRNRoGDuECgYEA6A1EixLq7QYOh3cb
Xhyh3W4kpVvA/FPfKH1OMy3ONOD/Y9Oa+M/wthW1wSoRL2n+uuIW5OAhTIvIEivj
6bAZsTT3twrvOrvYu9rx9aln4p8BhyvdjeW4kS7T8FP5ol6LoOt2sTP3T1LOuJPO
uQvOjlKPKIMh3c3RFNWTnGzMPa0CgYEA0jNiPLxP3A2nrX0keKDI+VHuvOY88gdh
0W5BuLMLovOIDk9aQFIbBbMuW1OTjHKv9NK+Lrw+YbCFqOGf1dU/UN5gSyE8lX/Q
FsUGUqUZx574nJZnOIcy3ONOnQLcvHAQToLFAGUd7PWgP3CtHkt9hEv2koUwL4vo
ikTP1u9Gkc0CgYEA2apoWxPZrY963XLKBxNQecYxNbLFaWq67t3rFnKm9E8BAICi
4zUaE5J1tMVi7Vi9iks9Ml9SnNyZRQJKfQ+kaebHXbkyAaPmfv+26rqHKboA0uxA
nDOZVwXX45zBkp6g1sdHxJx8JLoGEnkC9eyvSi0C//tRLx86OhLErXwYcNkCf1it
VMRKrWYoXJTUNo6tRhvodM88UnnIo3u3CALjhgU4uC1RTMHV4ZCGBwiAOb8GozSl
s5YD1E1iKwEULloHnK6BIh6P5v8q7J6uf/xdqoKMjlWBHgq6/roxKvkSPA1DOZ3l
jTadcgKFnRUmc+JT9p/ZbCxkA/ALFg8++G+0ghECgYA8vG3M/utweLvq4RI7l7U7
b+i2BajfK2OmzNi/xugfeLjY6k2tfQGRuv6ppTjehtji2uvgDWkgjJUgPfZpir3I
RsVMUiFgloWGHETOy0Qvc5AwtqTJFLTD1Wza2uBilSVIEsg6Y83Gickh+ejOmEsY
6co17RFaAZHwGfCFFjO76Q==
-----END PRIVATE KEY-----
""")


def start_build(name):
  ssh = paramiko.SSHClient()
  ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  key = paramiko.RSAKey.from_private_key(SSH_KEY)

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

  # Send go.sh
  sftp = s.open_sftp()

  conn = ssh.invoke_shell()
  branch = os.environ['GIT_BRANCH']
  commit = os.environ.get('GIT_COMMIT', branch)

  conn.send('uname -a\n')
  return conn  # Exit early for test

  conn.send('cd /data/openpilot\n')
  conn.send("git reset --hard\n")
  conn.send("git fetch origin\n")
  conn.send("git checkout %s\n" % commit)
  conn.send("git clean -xdf\n")
  conn.send("git submodule update --init\n")
  conn.send("git submodule foreach --recursive git reset --hard\n")
  conn.send("git submodule foreach --recursive git clean -xdf\n")
  conn.send("echo \"git took $SECONDS seconds\"\n")

  push = "PUSH=one-master" if branch == "master" else ""

  conn.send("%s /data/openpilot/release/go.sh\n" % push)
  conn.send('echo "RESULT:" $?\n')
  conn.send("exit\n")
  return conn


if __name__ == "__main__":
  eon_name = os.environ.get('eon_name', None)

  conn = start_build(eon_name)

  dat = ""
  while True:
      recvd = conn.recv(4096)

      dat += recvd
      sys.stdout.write(recvd)
      sys.stdout.flush()

    if len(recvd) == 0
      break

  returns = int(re.findall(r'^RESULT: (\d+)', dat[-1024:], flags=re.MULTILINE)[0]
  sys.exit(returns)
