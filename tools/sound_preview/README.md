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

## Escalation contract

`openpilot/selfdrive/ui/critical_alert.py` owns the timer and max-family routing.
It has no dependency on event definitions, original sound assignments or audio
files. `soundd` supplies eligibility: critical status and a visible alert.
Initially silent alerts also escalate; silence does not reset the timer. Ineligible ticks reset the timer. Continuous eligible
ticks preserve it even when the event or starting sound changes.

After eight seconds, playback uses the current event's max family at 100%.
DM event identities (driverDistracted, driverUnresponsive and the camera preview)
select the driver family; other, missing and new identities select the generic
family. Routing is isolated in `max_alert_for_type`; changing an event identity
can change its family but cannot disable escalation. Actual files remain in
`soundd.sound_list`. Both families currently use `dm_critical_max.wav`.

Publishers must provide `alertStatus`, `alertSize`, `alertSound` and, for DM
routing, `alertType`. Severity must not be inferred from the original sound.
The reassignment presets are synthetic tests of this contract, not current
openpilot event assignments.
