#!/usr/bin/env python3
import os
import sys
import wave
import argparse
import numpy as np

from openpilot.tools.lib.logreader import LogReader, ReadMode


def extract_audio(route_or_segment_name, output_file=None, play=False):
  lr = LogReader(route_or_segment_name, default_mode=ReadMode.AUTO_INTERACTIVE)
  audio_messages = list(lr.filter("rawAudioData"))
  if not audio_messages:
    print("No rawAudioData messages found in logs")
    return
  sample_rate = audio_messages[0].sampleRate

  audio_chunks = []
  total_frames = 0
  for msg in audio_messages:
    audio_array = np.frombuffer(msg.data, dtype=np.int16)
    audio_chunks.append(audio_array)
    total_frames += len(audio_array)
  full_audio = np.concatenate(audio_chunks)

  print(f"Found {total_frames} frames from {len(audio_messages)} audio messages at {sample_rate} Hz")

  if output_file:
    if write_wav_file(output_file, full_audio, sample_rate):
      print(f"Audio written to {output_file}")
    else:
      print("Audio extraction canceled.")
  if play:
    play_audio(full_audio, sample_rate)


def write_wav_file(filename, audio_data, sample_rate):
  if os.path.exists(filename):
    if input(f"File '{filename}' exists. Overwrite? (y/N): ").lower() not in ['y', 'yes']:
      return False

  with wave.open(filename, 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())
  return True


def play_audio(audio_data, sample_rate):
  try:
    import sounddevice as sd

    print("Playing audio... Press Ctrl+C to stop")
    sd.play(audio_data, sample_rate)
    sd.wait()
  except KeyboardInterrupt:
    print("\nPlayback stopped")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract audio data from openpilot logs")
  parser.add_argument("-o", "--output", help="Output WAV file path")
  parser.add_argument("--play", action="store_true", help="Play audio with sounddevice")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  output_file = args.output
  if not args.output and not args.play:
    output_file = "extracted_audio.wav"

  extract_audio(args.route_or_segment_name.strip(), output_file, args.play)
