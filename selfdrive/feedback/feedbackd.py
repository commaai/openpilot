#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog


def feedbackd_thread():
  pm = messaging.PubMaster(['userFlag'])
  sm = messaging.SubMaster(['feedbackState', 'bookmarkButton'])

  debounce_period = 2.0
  last_user_flag_time = 0
  consecutive_detections = 0

  while True:
    sm.update()
    if not (sm.updated['feedbackState'] or sm.updated['bookmarkButton']):
      continue

    current_time = time.monotonic()
    should_send_flag = False

    if sm.updated['feedbackState']:
      fs = sm['feedbackState']
      score = fs.wakewordProb
      if score > 0.2:
        consecutive_detections += 1
        cloudlog.debug(f"Wake word segment detected! Score: {score:.3f}, Consecutive: {consecutive_detections}")
        if (consecutive_detections >= 2 or score > 0.5) and (current_time - last_user_flag_time) > debounce_period:
          cloudlog.info("Wake word detected!")
          should_send_flag = True
      else:
        consecutive_detections = 0

    if sm.updated['bookmarkButton'] and (current_time - last_user_flag_time) > debounce_period:
      cloudlog.info("Bookmark button pressed!")
      should_send_flag = True

    if should_send_flag:
      last_user_flag_time = current_time
      msg = messaging.new_message('userFlag', valid=True)
      pm.send('userFlag', msg)


def main():
  feedbackd_thread()


if __name__ == '__main__':
  main()
