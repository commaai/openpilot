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
virtual PipeWire sink **Comma Speaker**, sets it as the system default, and streams
captured audio over LAN. On `Ctrl+C` the previous default sink is restored and the device
server is stopped.

## Troubleshooting

- **"server not reachable at \<ip\>:7777"** — the laptop and the comma must be on the
  same WiFi/LAN. Pass `--ip 192.168.x.y` to override discovery.
- **"missing required tools: pactl, parec, …"** — install `pipewire-pulse` and the
  PipeWire utility package (`pipewire-utils` on Debian/Ubuntu).
- **"another comma_speaker is already running"** — `rm /tmp/comma_speaker.pid` if you
  killed a previous run uncleanly.
- **No sound on device** — try `./comma_speaker.py --ping` to play a known beep, or check
  `/tmp/play_server.log` on the device.

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

- `comma_speaker.py` — the foolproof launcher (run this).
- `play_server.py` — the device-side server. DSP-free passthrough that mirrors
  `selfdrive/ui/soundd.py`'s OutputStream exactly.
- `send_sink.py`, `send_audio.py` — older lower-level scripts kept for power users.
- `play_wav.py`, `mici.txt` — offline DSP experimentation tool (unrelated to the speaker).
