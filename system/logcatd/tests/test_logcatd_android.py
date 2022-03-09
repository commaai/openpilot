#!/usr/bin/env python3
import os
import random
import string
import time
import unittest
import uuid

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes

class TestLogcatdAndroid(unittest.TestCase):

  @with_processes(['logcatd'])
  def test_log(self):
    sock = messaging.sub_sock("androidLog", conflate=False)

    # make sure sockets are ready
    time.sleep(1)
    messaging.drain_sock(sock)

    sent_msgs = {}
    for _ in range(random.randint(2, 10)):
      # write some log messages
      for __ in range(random.randint(5, 50)):
        tag = uuid.uuid4().hex
        msg = ''.join(random.choice(string.ascii_letters) for _ in range(random.randrange(2, 50)))
        sent_msgs[tag] = {'recv_cnt': 0, 'msg': msg}
        os.system(f"log -t '{tag}' '{msg}'")

      time.sleep(1)
      msgs = messaging.drain_sock(sock)
      for m in msgs:
        self.assertTrue(m.valid)
        self.assertLess(time.monotonic() - (m.logMonoTime / 1e9), 30)
        tag = m.androidLog.tag
        if tag in sent_msgs:
          sent_msgs[tag]['recv_cnt'] += 1
          self.assertEqual(m.androidLog.message.strip(), sent_msgs[tag]['msg'])

    for v in sent_msgs.values():
      self.assertEqual(v['recv_cnt'], 1)

if __name__ == "__main__":
  unittest.main()
