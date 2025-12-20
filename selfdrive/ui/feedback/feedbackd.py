#!/usr/bin/env python3
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from cereal import car
from openpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER

FEEDBACK_MAX_DURATION = 10.0
ButtonType = car.CarState.ButtonEvent.Type


def handle_lkas_button(be, params, state):
  """
  Handle LKAS button press/release logic.
  Returns True if a bookmark should be sent.
  """
  if be.type != ButtonType.lkas:
    return False

  should_send_bookmark = False

  if be.pressed:
    if not state['should_record_audio']:
      if params.get_bool("RecordAudioFeedback"):
        state.update(should_record_audio=True, block_num=0, waiting_for_release=False, early_stop_triggered=False)
        cloudlog.info("LKAS button pressed - starting 10-second audio feedback")
      else:
        should_send_bookmark = True
        cloudlog.info("LKAS button pressed - bookmarking")
    elif state['should_record_audio'] and not state['waiting_for_release']:
      state['waiting_for_release'] = True
  elif state['waiting_for_release']:
    state['waiting_for_release'] = False
    state['early_stop_triggered'] = True
    cloudlog.info("LKAS button released - ending recording early")

  return should_send_bookmark

def record_audio_block(sm, pm, state):
  """
  Send one audio block. Returns True if a bookmark should be sent after this block.
  """
  raw_audio = sm['rawAudioData']
  msg = messaging.new_message('audioFeedback', valid=True)
  msg.audioFeedback.audio.data = raw_audio.data
  msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
  msg.audioFeedback.blockNum = state['block_num']
  state['block_num'] += 1
  pm.send('audioFeedback', msg)

  if (state['block_num'] * SAMPLE_BUFFER / SAMPLE_RATE) >= FEEDBACK_MAX_DURATION or state['early_stop_triggered']:  # Check for timeout or early stop
    state['should_record_audio'] = False
    state['early_stop_triggered'] = False
    cloudlog.info("10-second recording completed or second button press - stopping audio feedback")
    return True

  return False

def send_bookmark(pm):
  """
  Send a user bookmark message.
  """
  msg = messaging.new_message('userBookmark', valid=True)
  pm.send('userBookmark', msg)

def main():
  params = Params()
  pm = messaging.PubMaster(['userBookmark', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton', 'carState'])
  state = {
    'should_record_audio': False,
    'block_num': 0,
    'waiting_for_release': False,
    'early_stop_triggered': False
  }

  while True:
    sm.update()
    should_send_bookmark = False

    # TODO: https://github.com/commaai/openpilot/issues/36015
    if False and sm.updated['carState'] and sm['carState'].canValid:
      for be in sm['carState'].buttonEvents:
        if be.type == ButtonType.lkas:
          if handle_lkas_button(be, params, state):
            should_send_bookmark = True

    if state['should_record_audio'] and sm.updated['rawAudioData']:
      if record_audio_block(sm, pm, state):
        should_send_bookmark = True

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      should_send_bookmark = True

    if should_send_bookmark:
      send_bookmark(pm)

if __name__ == '__main__':
  main()
