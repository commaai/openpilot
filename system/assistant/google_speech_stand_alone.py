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
CHUNK = int(RATE / 10)  # 100ms

audio_queue = queue.Queue()

# Configure the speech recognition
streaming_config = speech.StreamingRecognitionConfig(
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    ),
    interim_results=True,
)

def audio_callback(indata, frames, time, status):
    """This is called for each audio block."""
    if status:
        print(f"Stream error: {status}", flush=True)
    audio_queue.put(indata.copy())

def microphone_stream():
    import sounddevice as sd
    """Generator that yields audio chunks from the queue."""
    with get_stream(sd):
        while True:
            data = audio_queue.get()
            yield np.ndarray.tobytes(data)

@retry(attempts=7, delay=3)
def get_stream(sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.InputStream(samplerate=RATE, channels=1, callback=audio_callback, blocksize=CHUNK, dtype='int16')

def listen_print_loop(responses):
    """Processes the streaming responses from Google Speech API."""
    for response in responses:
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
            if result.is_final:
                print("Final transcript: {}".format(result.alternatives[0].transcript))
                exit()

if __name__ == "__main__":
    audio_generator = microphone_stream()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
    #while not p.get_bool("WakeWordDetected"):
        #time.sleep(.1)
    responses = client.streaming_recognize(streaming_config, requests)
    listen_print_loop(responses)
