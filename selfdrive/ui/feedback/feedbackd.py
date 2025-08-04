#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from cereal import car

FEEDBACK_MAX_DURATION = 10.0

ButtonType = car.CarState.ButtonEvent.Type

def feedbackd_thread():
  pm = messaging.PubMaster(['userFlag', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton', 'carState'])
  should_record_audio = False
  segment_num = 0
  recording_start_time = None
  waiting_for_release = False
  early_send_triggered = False

  while True:
    sm.update()
    should_send_flag = False
    current_time = time.monotonic()

    if sm.updated['carState'] and sm['carState'].canValid:
      for be in sm['carState'].buttonEvents:
        if be.type == ButtonType.lkas:
          if be.pressed:
            if not should_record_audio:
              # Start recording on first press
              should_record_audio = True
              segment_num = 0
              recording_start_time = current_time
              waiting_for_release = False
              early_send_triggered = False
              should_send_flag = True
              cloudlog.info("LKAS button pressed - starting 10-second audio feedback")
            elif not waiting_for_release: # Wait for second press release to end early
              waiting_for_release = True
          elif waiting_for_release:  # Second press released
            early_send_triggered = True
            should_record_audio = False
            waiting_for_release = False
            recording_start_time = None
            cloudlog.info("LKAS button released - ending recording early")

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      should_send_flag = True

    # Check for timeout
    if should_record_audio and current_time - recording_start_time >= FEEDBACK_MAX_DURATION:
      should_record_audio = False
      recording_start_time = None
      waiting_for_release = False
      early_send_triggered = False
      cloudlog.info("10-second recording completed - stopping audio feedback")

    if (should_record_audio or early_send_triggered) and sm.updated['rawAudioData']:
      raw_audio = sm['rawAudioData']
      msg = messaging.new_message('audioFeedback', valid=True)
      msg.audioFeedback.audio.data = raw_audio.data
      msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
      msg.audioFeedback.segmentNum = segment_num
      if early_send_triggered:
        msg.audioFeedback.earlySend = True
        early_send_triggered = False
        cloudlog.info("Sent early send signal for audio feedback")
      pm.send('audioFeedback', msg)
      segment_num += 1

    if should_send_flag:
      msg = messaging.new_message('userFlag', valid=True)
      pm.send('userFlag', msg)

def main():
  feedbackd_thread()

if __name__ == '__main__':
  main()
