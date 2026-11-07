#!/usr/bin/env python3
# workstation side of the comma four voice assistant
# device -> server frames: [u8 type][u32 len][payload]
#   1 json meta, 2 pcm s16le 16k mono, 3 end of utterance
#   4 ping, echoed back as 14
# stream mode sends 10 {'wake': true} as soon as the wake word is heard, before the query ends
# server -> device: 10 json transcript/verdict, 11 pcm s16le 22050 mono tts, 12 json reply text, 13 done, 14 pong
import asyncio, json, re, struct, time, os
import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
import aiohttp
import reverse_geocoder
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws import Kws

KIMI_URL = os.environ['KIMI_URL']
KIMI_MODEL = os.environ.get('KIMI_MODEL', '/models/kimi-k3')
# comma dongle ids allowed to talk to this server, any registered comma device when unset
ALLOWED = {d for d in os.environ.get('VOICE_DONGLES', '').split(',') if d}
COMMA_API = 'https://api.commadotai.com'
VOICE = os.environ.get('PIPER_VOICE', os.path.expanduser('~/.local/share/piper/en_US-lessac-medium.onnx'))
PORT = int(os.environ.get('PORT', '8765'))
WAKE = 'clanker'
KWS_MODEL = os.environ.get('KWS_MODEL', os.path.join(os.path.dirname(__file__), 'kws_v2.npz'))
# stream mode: low detector threshold, whisper drops the false candidates
KWS_TH = float(os.environ.get('KWS_TH', '0.3'))
HOP = 1280
PRE_S, MAX_S, END_SIL_S, MIN_S = 4.0, 8.0, 0.7, 1.2
EARLY_S = 0.6


def _edits(a: str, b: str) -> int:
  d = list(range(len(b) + 1))
  for i, ca in enumerate(a, 1):
    p, d[0] = d[0], i
    for j, cb in enumerate(b, 1):
      p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
  return d[-1]


def find_wake(text: str):
  # whisper often hears the wake word as blanker, planker or clinker
  words = list(re.finditer(r"[a-zA-Z']+", text))
  for k, m in enumerate(words):
    w = m.group().lower()
    # v1 favours always answering, looser at the start, chatter before the wake word stays in the stream window
    if len(w) >= 4 and _edits(w, WAKE) <= (3 if k < 3 else 2):
      return m
  return None
SYSTEM = "You are clanker, the voice assistant in a car running openpilot. The driver is driving: answer in one or two short spoken sentences, no lists, no markdown, no reasoning. Never tell the driver to look at a phone or screen. If you lack live data, say so in a few words."

whisper = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
whisper.transcribe(np.zeros(16000, np.float32))
tts = PiperVoice.load(VOICE)
reverse_geocoder.search((0.0, 0.0), mode=1)
TTS_SR = tts.config.sample_rate
kws = Kws(KWS_MODEL)


def describe(ctx: dict) -> str:
  # turn the device snapshot into a few facts the model can use, units the driver expects
  out = []
  if 'unix_time' in ctx:
    # the car's clock offset, not the workstation's
    tz = ctx.get('tz', '+0000')
    off = (1 if tz[0] == '+' else -1) * (int(tz[1:3]) * 3600 + int(tz[3:5]) * 60)
    out.append('local time: ' + time.strftime('%A %H:%M', time.gmtime(ctx['unix_time'] + off)))
  g = ctx.get('gps')
  if g:
    p = reverse_geocoder.search((g['lat'], g['lon']), mode=1)[0]
    out.append(f"position: {g['lat']}, {g['lon']} near {p['name']}, {p['admin1']}, {p['cc']}, heading {g['bearing_deg']} deg")
  c = ctx.get('car', {})
  if 'speed_mps' in c:
    out.append(f"car speed: {c['speed_mps'] * 3.6:.0f} km/h ({c['speed_mps'] * 2.237:.0f} mph), gear {c.get('gear', '?')}")
  if c.get('cruise_set_mps'):
    out.append(f"cruise set: {c['cruise_set_mps'] * 3.6:.0f} km/h")
  if 'model' in c:
    out.append(f"car: {c['model']}")
  op = ctx.get('openpilot')
  if op:
    out.append(f"openpilot engaged: {op['engaged']}, personality: {op['personality']}")
  n = ctx.get('nav')
  if n and n.get('next'):
    out.append(f"navigation: next '{n['next']}' in {n['next_in_m']} m, {n['remaining_m'] / 1000:.1f} km and {n['remaining_s'] // 60} min to go")
  d = ctx.get('device')
  if d:
    out.append(f"device temperature {d['max_temp_c']} C, network {d['network']}")
  return '\n'.join(out)


_authed: dict[tuple[str, str], float] = {}


async def authorized(meta: dict) -> bool:
  # the port is public through tailscale funnel, the device proves who it is with its own comma jwt
  dongle, token = meta.get('dongle_id', ''), meta.get('token', '')
  if not re.fullmatch(r'[0-9a-f]{16}', dongle) or not token or (ALLOWED and dongle not in ALLOWED):
    return False
  if _authed.get((dongle, token), 0) > time.monotonic():
    return True
  async with aiohttp.ClientSession() as s:
    async with s.get(f'{COMMA_API}/v1.1/devices/{dongle}/', headers={'Authorization': f'JWT {token}'}, timeout=aiohttp.ClientTimeout(total=10)) as r:
      ok = r.status == 200
  if ok:
    _authed[(dongle, token)] = time.monotonic() + 600
  return ok


def frame(t: int, payload: bytes) -> bytes:
  return struct.pack('<BI', t, len(payload)) + payload


def transcribe(pcm: np.ndarray) -> str:
  # the comma four mic sits around -55 dBFS, whisper drops quiet starts
  pcm = pcm / (np.abs(pcm).max() + 1e-6) * 0.9
  segs, _ = whisper.transcribe(pcm, language='en', beam_size=1, vad_filter=False)
  return ' '.join(s.text for s in segs).strip()


def strip_wake(text: str) -> str:
  m = find_wake(text)
  return text[m.end():].lstrip(' ,.!?') if m else text


async def kimi_stream(session, query: str, ctx: str = ''):
  # kimi thinks by default, which costs ~1 s before the first spoken word
  body = {'model': KIMI_MODEL, 'stream': True, 'max_tokens': 200, 'temperature': 0.6, 'chat_template_kwargs': {'thinking': False},
          'messages': [{'role': 'system', 'content': SYSTEM + ('\n\nLive data from the car, use it when relevant:\n' + ctx if ctx else '')}, {'role': 'user', 'content': query}]}
  async with session.post(KIMI_URL, json=body, timeout=aiohttp.ClientTimeout(total=30)) as r:
    async for line in r.content:
      line = line.decode().strip()
      if not line.startswith('data:') or line.endswith('[DONE]'):
        continue
      c = json.loads(line[5:])['choices'][0]['delta'].get('content')
      if c:
        yield c


def synth(text: str) -> bytes:
  return b''.join(c.audio_int16_bytes for c in tts.synthesize(text, syn_config=SynthesisConfig(length_scale=0.95)))


async def respond(writer, pcm, meta, t) -> str | None:
  text = await asyncio.to_thread(transcribe, pcm)
  t['stt'] = time.monotonic()
  verified = find_wake(text) is not None or not meta.get('verify', True)
  query = strip_wake(text)
  writer.write(frame(10, json.dumps({'transcript': text, 'verified': verified, 'query': query}).encode()))
  await writer.drain()
  if not verified:
    writer.write(frame(13, json.dumps({'reason': 'not verified'}).encode()))
    await writer.drain()
    return None
  if not query:
    # bare wake word, ask and let the device listen once more without the wake word
    writer.write(frame(12, json.dumps({'sample_rate': TTS_SR, 'say': 'Yes?'}).encode()))
    writer.write(frame(11, await asyncio.to_thread(synth, 'Yes?')))
    writer.write(frame(13, json.dumps({'followup': True}).encode()))
    await writer.drain()
    return 'followup'
  writer.write(frame(12, json.dumps({'sample_rate': TTS_SR}).encode()))
  reply, buf = '', ''
  async with aiohttp.ClientSession() as s:
    async for tok in kimi_stream(s, query, describe(meta.get('context', {}))):
      if 'llm_first' not in t:
        t['llm_first'] = time.monotonic()
      reply += tok
      buf += tok
      # speak each finished sentence while the rest is still streaming
      m = re.search(r'[.!?](\s|$)', buf)
      if m:
        sent, buf = buf[:m.end()], buf[m.end():]
        pcm_out = await asyncio.to_thread(synth, sent)
        t.setdefault('tts_first', time.monotonic())
        # the text goes right before its audio so the caption follows the voice
        writer.write(frame(12, json.dumps({'say': sent}).encode()))
        writer.write(frame(11, pcm_out))
        await writer.drain()
  if buf.strip():
    writer.write(frame(12, json.dumps({'say': buf}).encode()))
    writer.write(frame(11, await asyncio.to_thread(synth, buf)))
    t.setdefault('tts_first', time.monotonic())
  lat = {k: round(v - t['eou'], 3) for k, v in t.items() if k not in ('start', 'eou')}
  writer.write(frame(12, json.dumps({'reply': reply.strip(), 'latency': lat}).encode()))
  writer.write(frame(13, b'{}'))
  await writer.drain()
  print(json.dumps({'transcript': text, 'reply': reply.strip(), 'latency': lat, 'audio_s': round(len(pcm) / 16000, 2), 'meta': {k: v for k, v in meta.items() if k != 'token'}}), flush=True)
  return 'done'


async def early_wake(writer, pcm: np.ndarray) -> None:
  # check just the wake word right away so the device can chime while the driver still talks
  if find_wake(await asyncio.to_thread(transcribe, pcm)):
    writer.write(frame(10, json.dumps({'wake': True}).encode()))
    await writer.drain()


async def stream(reader, writer, meta):
  # v1: the device streams the mic nonstop, wake detection, verify and reply all run here
  ring = np.zeros(0, np.float32)
  cand, last, floor, followup = None, -1e9, -60.0, False
  pos = 0
  while True:
    typ, n = struct.unpack('<BI', await reader.readexactly(5))
    payload = await reader.readexactly(n)
    if typ == 1:
      meta = json.loads(payload)
      continue
    if typ == 4:
      # latency ping, echoed as is
      writer.write(frame(14, payload))
      await writer.drain()
      continue
    if typ != 2:
      continue
    x = np.frombuffer(payload, np.int16).astype(np.float32) / 32768
    ring = np.concatenate([ring, x])[-int(16000 * (PRE_S + MAX_S + 2)):]
    pos += len(x)
    now = pos / 16000
    if not len(x):
      continue
    db = 10 * np.log10(np.mean(x ** 2) + 1e-12)
    if cand is None:
      floor = min(db, floor + 0.05) if db > floor else 0.9 * floor + 0.1 * db
    if followup and now - last > 6.0:
      followup = False
    if cand is None and followup and db > floor + 10:
      # answer to our 'Yes?', no wake word needed
      cand = {'t0': now - 0.3, 'quiet': 0.0, 'verify': False}
    if cand is None:
      if len(ring) >= 20480 and now - last > 2.0 and kws(ring[-20480:]) > KWS_TH:
        # the detector fires mid word, check once the whole word is in
        cand = {'t0': now, 'quiet': 0.0, 'verify': True, 'early_at': now + EARLY_S}
      continue
    if cand.get('early_at') and now >= cand['early_at']:
      cand['early_at'] = None
      asyncio.create_task(early_wake(writer, ring[-int(16000 * PRE_S):].copy()))
    # endpoint on energy relative to the running noise floor
    cand['quiet'] = cand['quiet'] + len(x) / 16000 if db < floor + 6 else 0.0
    dur = now - cand['t0']
    if dur < MIN_S or (cand['quiet'] < END_SIL_S and dur < MAX_S):
      continue
    pcm = ring[-int(16000 * (PRE_S + dur)):]
    t = {'start': time.monotonic(), 'eou': time.monotonic()}
    m = dict(meta, verify=cand['verify'])
    cand, last = None, now
    if os.environ.get('DUMP_DIR'):
      np.save(os.path.join(os.environ['DUMP_DIR'], f'{time.time():.3f}.npy'), pcm)
    # the device stops streaming while it plays the reply, so it never hears itself
    res = await respond(writer, pcm, m, t)
    followup = res == 'followup'
    if res:
      ring = np.zeros(0, np.float32)


async def handle(reader, writer):
  t = {'start': time.monotonic()}
  meta, chunks = {}, []
  try:
    while True:
      typ, n = struct.unpack('<BI', await reader.readexactly(5))
      payload = await reader.readexactly(n)
      if typ == 1:
        meta = json.loads(payload)
        if not await authorized(meta):
          print(f"rejected {meta.get('dongle_id')!r}", flush=True)
          return
        if meta.get('stream'):
          return await stream(reader, writer, meta)
      elif typ == 2:
        chunks.append(payload)
      elif typ == 3:
        break
    t['eou'] = time.monotonic()
    pcm = np.frombuffer(b''.join(chunks), np.int16).astype(np.float32) / 32768
    if os.environ.get('DUMP_DIR'):
      np.save(os.path.join(os.environ['DUMP_DIR'], f'{time.time():.3f}.npy'), pcm)
    await respond(writer, pcm, meta, t)
  except asyncio.IncompleteReadError:
    pass
  finally:
    writer.close()


async def main():
  srv = await asyncio.start_server(handle, '0.0.0.0', PORT)
  print(f'voiceserver ready on {PORT}', flush=True)
  async with srv:
    await srv.serve_forever()


if __name__ == '__main__':
  asyncio.run(main())
