import os
import time
import numpy as np

import google.cloud.speech as speech
from openpilot.common.params import Params
from cereal.messaging import SubMaster, PubMaster, new_message
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE
import threading
from queue import Queue

# Google Cloud Speech Client
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
except:
    print("Set your google project authorization key location with the command:")
    print("export GOOGLE_APPLICATION_CREDENTIALS=<path/to/projectname.json>")
    exit()

client = speech.SpeechClient()
stop_thread = threading.Event()

pm = PubMaster(['speechToText'])

# Audio recording parameters
RATE = SAMPLE_RATE
CHUNK = SAMPLE_BUFFER
BUFFERS_PER_SECOND = SAMPLE_RATE/SAMPLE_BUFFER
TIMEOUT = 10
audio_queue = Queue(maxsize=BUFFERS_PER_SECOND*TIMEOUT)

# Configure the speech recognition
streaming_config = speech.StreamingRecognitionConfig(
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Use FLAC to save bandwidth
        sample_rate_hertz=RATE,
        language_code="en-US",
    ),
    single_utterance=True,
    interim_results=True,
    enable_voice_activity_events=False,
)

def microphone_data_collector():
    """Thread function for collecting microphone data."""
    sm = SubMaster(['microphoneRaw'])
    while not stop_thread.is_set():
        sm.update(0)
        if sm.updated['microphoneRaw']:
            data = np.frombuffer(sm['microphoneRaw'].rawSample, dtype=np.int16)
            if not audio_queue.full():
                audio_queue.put(data)
            else:
                print("Queue is full, dropping data")

def microphone_stream():
    """Generator that yields audio chunks from the queue."""
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            yield np.ndarray.tobytes(data)

def listen_print_loop(responses):
    """Processes the streaming responses from Google Speech API."""
    msg = new_message('speechToText', valid=True)
    for response in responses:
        for result in response.results:
            msg.speechToText.result = result.alternatives[0].transcript
            #print(f'Transcript: {result.alternatives[0].transcript}')
            if result.is_final:
                msg.speechToText.finalResultReady = True
                #print(f'Final transcript: {result.alternatives[0].transcript}')
                pm.send('speechToText', msg)
                return result.alternatives[0].transcript
            else:
                msg.speechToText.finalResultReady = False
                pm.send('speechToText', msg)

def process_request():
    audio_generator = microphone_stream()
    max_loops = BUFFERS_PER_SECOND * TIMEOUT
    loop_count = 0
    start_time = time.time()

    def timed_audio_generator():
        """Yields audio chunks with a loop-based and time-based timeout."""
        nonlocal loop_count  # Use the loop_count from the outer scope
        for chunk in audio_generator:
            current_time = time.time()
            if loop_count >= max_loops or current_time - start_time > TIMEOUT:
                print("Timeout reached")
                break
            yield chunk
            loop_count += 1

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in timed_audio_generator())
    responses = client.streaming_recognize(streaming_config, requests)

    return listen_print_loop(responses)

if __name__ == "__main__":
    p = Params()
    while True:  # Continuous outer loop
        while not p.get_bool("WakeWordDetected"):
            time.sleep(.1)
        collector_thread = threading.Thread(target=microphone_data_collector)
        collector_thread.start()
        final_text = process_request()
        stop_thread.set()
        collector_thread.join()
        stop_thread.clear()
