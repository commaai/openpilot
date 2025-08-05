#!/usr/bin/env python3
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from cereal import car
from openpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER

FEEDBACK_MAX_DURATION = 10.0
ButtonType = car.CarState.ButtonEvent.Type


def feedbackd_thread():
  params = Params()
  pm = messaging.PubMaster(['userBookmark', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton', 'carState'])
  should_record_audio = False
  segment_num = 0
  waiting_for_release = False
  early_stop_triggered = False

  while True:
    sm.update()
    should_send_bookmark = False

    if sm.updated['carState'] and sm['carState'].canValid:
      for be in sm['carState'].buttonEvents:
        if be.type == ButtonType.lkas:
          if be.pressed:
            if not should_record_audio and params.get_bool("RecordAudioFeedback"):  # Start recording on first press if toggle set
              should_record_audio = True
              segment_num = 0
              waiting_for_release = False
              early_stop_triggered = False
              should_send_bookmark = True
              cloudlog.info("LKAS button pressed - starting 10-second audio feedback")
            elif should_record_audio and not waiting_for_release:  # Wait for release of second press to stop recording early
              waiting_for_release = True
          elif waiting_for_release:  # Second press released
            waiting_for_release = False
            early_stop_triggered = True
            cloudlog.info("LKAS button released - ending recording early")

    if should_record_audio and (segment_num * SAMPLE_BUFFER / SAMPLE_RATE) >= FEEDBACK_MAX_DURATION:  # Check for timeout
      should_record_audio = False
      cloudlog.info("10-second recording completed or audio feedback disabled - stopping audio feedback")

    if should_record_audio and sm.updated['rawAudioData']:
      raw_audio = sm['rawAudioData']
      msg = messaging.new_message('audioFeedback', valid=True)
      msg.audioFeedback.audio.data = raw_audio.data
      msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
      msg.audioFeedback.segmentNum = segment_num
      if early_stop_triggered:
        msg.audioFeedback.earlyStop = True
        early_stop_triggered = False
        should_record_audio = False
        cloudlog.info("Sent early stop signal for audio feedback")
      pm.send('audioFeedback', msg)
      segment_num += 1

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      should_send_bookmark = True

    if should_send_bookmark:
      msg = messaging.new_message('userBookmark', valid=True)
      pm.send('userBookmark', msg)


def main():
  feedbackd_thread()


if __name__ == '__main__':
  main()
