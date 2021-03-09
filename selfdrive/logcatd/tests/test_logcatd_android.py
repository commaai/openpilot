#!/usr/bin/env python3
import os
import random
import string
import time
import unittest

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes

class TestLogcatdAndroid(unittest.TestCase):

  @with_processes(['logcatd'])
  def test_log(self):
    sock = messaging.sub_sock("androidLog", conflate=False)

    # make sure sockets are ready
    time.sleep(1)
    messaging.drain_sock(sock)

    for _ in range(random.randint(2, 10)):
      # write some log messages
      sent_msgs = {}
      for __ in range(random.randint(5, 50)):
        msg = ''.join([random.choice(string.ascii_letters) for _ in range(random.randrange(2, 50))])
        if msg in sent_msgs:
          continue
        sent_msgs[msg] = ''.join([random.choice(string.ascii_letters) for _ in range(random.randrange(2, 20))])
        os.system(f"log -t '{sent_msgs[msg]}' '{msg}'")

      time.sleep(1)
      msgs = messaging.drain_sock(sock)
      for m in msgs:
        self.assertTrue(m.valid)
        self.assertLess(time.monotonic() - (m.logMonoTime / 1e9), 30)

        recv_msg = m.androidLog.message.strip()
        if recv_msg not in sent_msgs:
          continue

        # see https://android.googlesource.com/platform/system/core/+/android-2.1_r1/liblog/logd_write.c#144
        radio_msg = m.androidLog.id == 1 and m.androidLog.tag.startswith("use-Rlog/RLOG-")
        if m.androidLog.tag == sent_msgs[recv_msg] or radio_msg:
          del sent_msgs[recv_msg]

      # ensure we received all the logs we sent
      self.assertEqual(len(sent_msgs), 0)


if __name__ == "__main__":
  unittest.main()
