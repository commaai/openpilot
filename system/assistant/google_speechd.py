import os
import google.cloud.speech as speech
from openpilot.common.params import Params
from openpilot.common.retry import retry
import time
import numpy as np
import queue

p = Params()

# Google Cloud Speech Client
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
except:
    print("Set your google project authorization key location with the command:")
    print("export GOOGLE_APPLICATION_CREDENTIALS=<path/to/projectname.json>")
    exit()

client = speech.SpeechClient()

# Audio recording parameters
RATE = 16000
CHUNK = 1280  # 100ms

audio_queue = queue.Queue()

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

def audio_callback(indata, frames, time, status):
    """This is called for each audio block."""
    if status:
        print(f"Stream error: {status}", flush=True)
    audio_queue.put(indata.copy())

def microphone_stream():
    from cereal import messaging
    sm = messaging.SubMaster(['microphone'])
    """Generator that yields audio chunks from the queue."""
    while True:
        sm.update(0)
        if sm.updated['microphone']:
            data = np.frombuffer(sm['microphone'].rawSample[0], dtype=np.float32)
            idx = sm['microphone'].frameIndex
            print("streaming mic")
            yield np.ndarray.tobytes(data)         

def listen_print_loop(responses):
    """Processes the streaming responses from Google Speech API."""
    for response in responses:
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
            if result.is_final:
                print("Final transcript: {}".format(result.alternatives[0].transcript))
                return result.alternatives[0].transcript

def process_request():
    audio_generator = microphone_stream()
    #next(audio_generator)  # Prime the generator
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
    responses = client.streaming_recognize(streaming_config, requests)
    final_text = listen_print_loop(responses)
    return final_text

if __name__ == "__main__":
    while True:  # Continuous outer loop
        while not p.get_bool("WakeWordDetected"):
            time.sleep(.1)
        final_text = process_request()
        # Handle the final_text here
        print("Processed: ", final_text)

    
