# Sound preview

From the repository root, with the openpilot Python environment active:

```sh
python -m tools.sound_preview.preview
```

Open http://127.0.0.1:8769. This local-only workbench generates a WAV and a
20 Hz state trace using the real `Soundd` code with simulated state messages
and a simulated clock. It does not open an audio device or publish messages.

Choose a preset, or edit the alert phases and durations. Audio regenerates
automatically, and the previous audio is cleared immediately to avoid stale playback.
Scrub near the eight-second boundary to hear the switch. Initial app volume is
part of the simulation; listening volume only affects browser playback.
The downloaded WAV contains the actual app gain, including full-volume max.

Automatic checks cover both sound families, silent alerts, non-red alerts,
short warnings, interrupted timers, continuous changes, and timeout handling.
A reproduction of the old camera-preview metadata shows why it did not escalate.
These checks do not validate camera/event generation, the actual preview publisher,
hardware loudness, or microphone-driven ambient volume changes.

Change the `MaxAlert.critical` file in `soundd.py` when `critical_max.wav` is ready.
