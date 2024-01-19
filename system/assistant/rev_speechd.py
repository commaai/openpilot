import json
import time
import os
import re
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from websocket import _exceptions
from threading import Thread, Event
from queue import Queue
from cereal import messaging, log
from openpilot.common.params import Params
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE

STTState = log.SpeechToText.State

class SpeechToTextProcessor:
  TIMEOUT_DURATION = 10
  RATE = SAMPLE_RATE
  CHUNK = SAMPLE_BUFFER
  BUFFERS_PER_SECOND = SAMPLE_RATE/SAMPLE_BUFFER
  QUEUE_TIME = 10  # Save the first 10 seconds to the queue
  CONNECTION_TIMEOUT = 30
  INITIAL_TEXT = "Hello, I'm listening" # TODO : move to UI code to handle translations
  NO_RESP_TEXT = "Sorry, I didn't catch that"
  ERROR_TEXT = "Sorry, an error occorred"

  def __init__(self, access_token, queue_size=BUFFERS_PER_SECOND*QUEUE_TIME):
    self.reva_access_token = access_token
    self.audio_queue = Queue(maxsize=int(queue_size))
    self.stop_thread = Event()
    self.pm = messaging.PubMaster(['speechToText'])
    self.sm = messaging.SubMaster(['microphoneRaw'])
    media_config = MediaConfig('audio/x-raw', 'interleaved', 16000, 'S16LE', 1)
    self.streamclient = RevAiStreamingClient(self.reva_access_token, media_config)
    self.p = Params()
    self.error = False

  def microphone_data_collector(self):
    """Thread function for collecting microphone data."""
    while not self.stop_thread.is_set():
      self.sm.update(0)
      if self.sm.updated['microphoneRaw']:
        data = self.sm['microphoneRaw'].rawSample
        if not self.audio_queue.full():
          print("putting the audio")
          self.audio_queue.put(data)
        else:
          print("Queue is full, stopping")
          self.stop_thread.set()
          msg = messaging.new_message('speechToText', valid=False)
          self.pm.send('speechToText', msg)


  def microphone_stream(self):
    """Generator that yields audio chunks from the queue."""
    loop_count = 0
    start_time = time.time()
    while True:
      if loop_count >= self.audio_queue.maxsize or time.time() - start_time > self.CONNECTION_TIMEOUT:
        print(f'Timeout reached. {loop_count=}, {time.time()-start_time=}')
        break
      elif self.stop_thread.is_set():
        print(f'stop_thread.is_set()=')
        break
      elif not self.audio_queue.empty():
        data = self.audio_queue.get(block=True)
        loop_count += 1
        yield data
      else:
        time.sleep(.1)

  def listen_print_loop(self, response_gen, final_transcript):
    """Processes the streaming responses from Rev.ai."""
    try:
      for response in response_gen:
        data = json.loads(response)
        if data['type'] == 'final':
          # Extract and concatenate the final transcript then send it
          final_transcript = ' '.join([element['value'] for element in data['elements'] if element['type'] == 'text'])
        else:
          msg = messaging.new_message('speechToText', valid=True)
          # Handle partial transcripts (optional)
          partial_transcript = ' '.join([element['value'] for element in data['elements'] if element['type'] == 'text'])
          msg.speechToText.result = re.sub(r'<[^>]*>', '', partial_transcript)  # Remove atmospherics if they are present
          msg.speechToText.finalResultReady = False
          self.pm.send('speechToText', msg)

    except Exception as e:
      print(f"An error occurred: {e}")
      self.error=True

    return re.sub(r'<[^>]*>', '', final_transcript) # remove atmospherics. ex: <laugh>

  def run(self):
    self.audio_queue.queue.clear()
    collector_thread = Thread(target=self.microphone_data_collector)
    final_transcript = ""
    self.error = False
    while not self.p.get_bool("WakeWordDetected"):
      # Improve response time by combining wakewordd.py and this script. For now, keep it modular
      time.sleep(.5)
      print("waiting for wakeword")

    # Start the microphone data collector thread
    collector_thread.start()
    msg = messaging.new_message('speechToText', valid=True)
    msg.speechToText.state = STTState.begin # Show
    self.pm.send('speechToText', msg)

    try:
      # Start streaming to Rev.ai with a new generator instance
      response_gen = self.streamclient.start(self.microphone_stream(),
                                             remove_disfluencies=True, # remove umms
                                             filter_profanity=True, # brand integridity or something
                                             detailed_partials=False, # don't need time stamps
                                            )
      final_transcript = self.listen_print_loop(response_gen, final_transcript)

    except _exceptions.WebSocketAddressException as e:
      print(f"WebSocketAddressException: Address unreachable. {e}")
      self.error = True
    except Exception as e:
      # TODO: handle disconnection better? ssl send can hang forever until reconnected.
      # This tries to catch the error when it reconnects. Needs more testing
      print(f"An error occurred: {e}")
      self.error = True
    finally:
      print("Waiting for collector_thread to join...")
      self.stop_thread.set() # end the stream
      collector_thread.join()
      self.stop_thread.clear()
      print("collector_thread joined")

      msg = messaging.new_message('speechToText', valid=not self.error)
      msg.speechToText.result = final_transcript
      msg.speechToText.state = STTState.none if final_transcript else STTState.empty
      msg.speechToText.finalResultReady = True
      self.pm.send('speechToText', msg)


def main():
  try:
    reva_access_token = os.environ["REVAI_ACCESS_TOKEN"]
  except KeyError:
    print("your rev ai acccess token which can be obtained with a free account. https://www.rev.ai/access-token")
    print("Set your REVAI_ACCESS_TOKEN with the command:")
    print('export REVAI_ACCESS_TOKEN="your token string"')

  processor = SpeechToTextProcessor(access_token=reva_access_token)
  while True:
    processor.p.put_bool("WakeWordDetected", False)
    processor.run()

if __name__ == "__main__":
  main()
