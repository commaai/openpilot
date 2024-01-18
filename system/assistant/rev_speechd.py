import numpy as np
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
from websocket import _exceptions
from threading import Thread, Event
from queue import Queue
from cereal.messaging import SubMaster, PubMaster, new_message
from openpilot.common.params import Params
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE
import json
import time
import os
import re


try:
    REVAI_ACCESS_TOKEN = os.environ["REVAI_ACCESS_TOKEN"]
except:
    print("Set your google project authorization key location with the command:")
    print("export GOOGLE_APPLICATION_CREDENTIALS=<path/to/projectname.json>")
    exit()

TIMEOUT_DURATION = 10

RATE = SAMPLE_RATE
CHUNK = SAMPLE_BUFFER
BUFFERS_PER_SECOND = SAMPLE_RATE/SAMPLE_BUFFER
QUEUE_TIME = 10 # Save the first 10 seconds to the queue
CONNECTION_TIMEOUT = 30
audio_queue = Queue(maxsize=int(BUFFERS_PER_SECOND*QUEUE_TIME))

# Global variables
stop_thread = Event()
connection_timeout_reset = Event()

pm = PubMaster(['speechToText'])


def microphone_data_collector(stop_thread):
    """Thread function for collecting microphone data."""
    sm = SubMaster(['microphoneRaw'])
    audio_queue.queue.clear()
    while not stop_thread.is_set():
        sm.update(0)
        if sm.updated['microphoneRaw']:
            data = sm['microphoneRaw'].rawSample
            if not audio_queue.full():
                print("putting the audio")
                audio_queue.put(data)
            else:
                print("Queue is full, stopping")
                return

def microphone_stream():
    """Generator that yields audio chunks from the queue."""
    loop_count = 0
    start_time = time.time()
    while True:
        if loop_count >= audio_queue.maxsize or time.time() - start_time > CONNECTION_TIMEOUT:
            print(f'Timeout reached. {loop_count=}, {time.time()-start_time=}')
            break
        elif stop_thread.is_set():
            print(f'stop_thread.is_set()=')
            break
        elif not audio_queue.empty():
            data = audio_queue.get(block=True)
            loop_count+=1
            yield data
        else:
            time.sleep(.1)
        
            
def listen_print_loop(response_gen):
    """Processes the streaming responses from Rev.ai."""
    final_transcript = ""
    try:
        for response in response_gen:
            connection_timeout_reset.set() # Recieved response. Reset timeout
            data = json.loads(response)
            
            if data['type'] == 'final':
                # Extract and concatenate the final transcript then send it
                final_transcript = ' '.join([element['value'] for element in data['elements'] if element['type'] == 'text'])
                stop_thread.set()
            else:
                msg = new_message('speechToText', valid=True)
                # Handle partial transcripts (optional)
                partial_transcript = ' '.join([element['value'] for element in data['elements'] if element['type'] == 'text'])
                msg.speechToText.result = re.sub(r'<[^>]*>', '', partial_transcript) # Remove atmospherics if they are present
                msg.speechToText.finalResultReady = False
                pm.send('speechToText', msg)

    except _exceptions.WebSocketConnectionClosedException:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Perform any necessary cleanup here
        print("Exiting listen_print_loop.")
        return final_transcript
            
def run():
    global stop_thread, connection_timeout_reset, audio_queue

    example_mc = MediaConfig('audio/x-raw', 'interleaved', 16000, 'S16LE', 1)
    streamclient = RevAiStreamingClient(REVAI_ACCESS_TOKEN, example_mc)
    p = Params()
    while True:
        p.put_bool("WakeWordDetected", False)
        while not p.get_bool("WakeWordDetected"):
            time.sleep(.2)
            print("waiting for wakeword")
        msg = new_message('speechToText', valid=True)
        msg.speechToText.result = "Hello, I'm listening"
        pm.send('speechToText', msg)

            
        # Reset stop event and audio queue for a new session
        stop_thread.clear()
        connection_timeout_reset.set()

        # Start the microphone data collector thread
        collector_thread = Thread(target=microphone_data_collector, args=(stop_thread,))
        collector_thread.start()

        try:
            # Start streaming to Rev.ai with a new generator instance
            response_gen = streamclient.start(microphone_stream(),
                                              remove_disfluencies=True,
                                              filter_profanity=True,
                                              detailed_partials=False,
                                              )
            final_transcript = listen_print_loop(response_gen)

        except _exceptions.WebSocketAddressException:
             print(f"WebSocketAddressException: Address unreachable.")
        finally:
            # End streaming and cleanup
            print("Closing connection...")
            stop_thread.set()  # Signal threads to stop
            collector_thread.join()
            print("Connection closed.")
            msg.speechToText.result = re.sub(r'<[^>]*>', '', final_transcript) # Remove atmospherics if they are present
            msg.speechToText.finalResultReady = True
            pm.send('speechToText', msg)
        
            
def main():
    run()

if __name__ == "__main__":
   main()
