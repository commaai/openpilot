import asyncio
import pyaudio
import wave


SOUNDS = {
  'engage': '../../selfdrive/assets/sounds/engage.wav',
  'disengage': '../../selfdrive/assets/sounds/disengage.wav',
  'error': '../../selfdrive/assets/sounds/warning_immediate.wav',
}


async def play_sound(sound):
  chunk = 5120
  with wave.open(SOUNDS[sound], 'rb') as wf:
    def callback(in_data, frame_count, time_info, status):
      data = wf.readframes(frame_count)
      return data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    frames_per_buffer=chunk,
                    stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
      await asyncio.sleep(0)
    stream.stop_stream()
    stream.close()
    p.terminate()
