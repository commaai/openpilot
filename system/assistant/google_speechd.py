import os
import time
import numpy as np

import google.cloud.speech as speech
from openpilot.common.params import Params
from cereal import messaging
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE

# Google Cloud Speech Client
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
except:
    print("Set your google project authorization key location with the command:")
    print("export GOOGLE_APPLICATION_CREDENTIALS=<path/to/projectname.json>")
    exit()

client = speech.SpeechClient()

# Audio recording parameters
RATE = SAMPLE_RATE
CHUNK = SAMPLE_BUFFER

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

def microphone_stream():
    """Generator that yields audio chunks from the queue."""
    sm = messaging.SubMaster(['microphoneRaw'])
    while True:
        sm.update(0)
        if sm.updated['microphoneRaw']:
            data = np.frombuffer(sm['microphoneRaw'].rawSample, dtype=np.int16)
            print(sm['microphoneRaw'].frameIndex)
            print("streaming mic")
            yield np.ndarray.tobytes(data)

def listen_print_loop(responses):
    """Processes the streaming responses from Google Speech API."""
    for response in responses:
        for result in response.results:
            print(f'Transcript: {result.alternatives[0].transcript}')
            if result.is_final:
                print(f'Final transcript: {result.alternatives[0].transcript}')
                return result.alternatives[0].transcript

def process_request(timeout=10):  # timeout in seconds
    start_time = time.time()
    audio_generator = microphone_stream()

    def timed_audio_generator():
        """Yields audio chunks with a timeout."""
        for chunk in audio_generator:
            current_time = time.time()
            if current_time - start_time > timeout:
                print("Timeout reached")
                break  # Stop yielding
            yield chunk

    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in timed_audio_generator())
    responses = client.streaming_recognize(streaming_config, requests)

    return listen_print_loop(responses)

if __name__ == "__main__":
    p = Params()
    while True:  # Continuous outer loop
        while not p.get_bool("WakeWordDetected"):
            time.sleep(.1)
        final_text = process_request()
        # Handle the final_text here
        print("Processed: ", final_text)

    
