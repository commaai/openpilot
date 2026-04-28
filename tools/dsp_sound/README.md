# Comma Speaker

Turn a comma 4 (mici) into a network speaker for your laptop. Whatever you play on the host
(Spotify, browser, media player, DAW, …) is mirrored to the device's built-in speaker over
TCP. Output is bit-identical to what `selfdrive/ui/soundd.py` produces — no DSP, no EQ, no
compression, no resampling.

## Usage

```bash
./comma_speaker.py            # zero-arg foolproof path
./comma_speaker.py --test     # local end-to-end check, no device needed
./comma_speaker.py --ping     # play a 440 Hz beep on the device, prompt y/n
./comma_speaker.py --stop     # kill the server still running on the device
```

The launcher discovers the device (cached IP → LAN scan → comma-proxy lookup → prompt),
SCPs the simplified server to `/data/dsp_sound/play_server.py` and starts it, creates a
virtual sink **Comma Speaker**, sets it as the system default, and streams captured audio
over LAN. On `Ctrl+C` the previous default sink is restored and the device server is
stopped.

## First-time setup

### Linux (PipeWire / PulseAudio)

```bash
sudo apt install pipewire-pulse pipewire-utils    # Debian/Ubuntu
# or your distro's equivalent — needs `pactl`, `parec`, `pw-metadata`
```

Nothing else to install — `pactl` creates the virtual sink, `parec` captures from its
monitor source, `pw-metadata` switches the system default.

### macOS

```bash
brew install blackhole-2ch          # virtual audio driver (free, MIT)
brew install switchaudio-osx        # for auto-switching the system default output
```

After installing BlackHole you'll need to grant kernel-extension permission in
**System Settings → Privacy & Security**, then reboot once. After that, "BlackHole 2ch"
shows up in **Audio MIDI Setup** as both an input and an output device.

The first time the launcher runs, macOS asks Terminal (or whichever shell you're in) for
**microphone permission** — grant it once. Without this, the InputStream opens but
delivers silence.

If you skip `switchaudio-osx`, the launcher still works but won't change your default
output for you — open **System Settings → Sound → Output → BlackHole 2ch** manually.

### Windows

Not implemented in this iteration. Pattern would be VB-Cable + sounddevice; PRs welcome.

## Troubleshooting

- **"server not reachable at \<ip\>:7777"** — laptop and comma must be on the same
  WiFi/LAN. Pass `--ip 192.168.x.y` to override discovery.
- **"missing tools: pactl, parec, …"** (Linux) — install `pipewire-pulse` and
  `pipewire-utils`.
- **"BlackHole 2ch not found"** (macOS) — `brew install blackhole-2ch`, restart, grant
  kernel-extension permission.
- **macOS plays silence** — check microphone permission for your terminal in
  **System Settings → Privacy & Security → Microphone**.
- **"another comma_speaker is already running"** — another instance holds the lock.
  Find it with `ps aux | grep comma_speaker`. If it's a stale pidfile from a hard kill,
  `rm /tmp/comma_speaker.pid`.
- **No sound on device** — try `./comma_speaker.py --ping` to play a known beep, or
  check `/tmp/play_server.log` on the device.

## Local end-to-end test (no comma needed)

```bash
# bit-identical wire-format check
./comma_speaker.py --test

# real audio loop on this machine: play_server outputs to your laptop speakers
uv run python3 tools/dsp_sound/play_server.py &
./comma_speaker.py --ip 127.0.0.1 --no-default
# then play any audio routed to the "Comma Speaker" sink (paplay --device=comma_speaker …)
```

## Files

- `comma_speaker.py` — the foolproof launcher. Cross-platform: dispatches to
  `LinuxBackend` (pactl + parec) or `MacBackend` (BlackHole + sounddevice).
- `play_server.py` — device-side server. DSP-free passthrough that mirrors
  `selfdrive/ui/soundd.py`'s OutputStream exactly. Test hook: set
  `COMMA_SPEAKER_TEST=<path>` env to write to a WAV instead of opening sounddevice.
- `send_sink.py`, `send_audio.py` — older lower-level scripts kept for power users.
- `play_wav.py`, `mici.txt` — offline DSP experimentation tool (unrelated to the speaker).
