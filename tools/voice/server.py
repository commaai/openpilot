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

KIMI_URL = os.environ['KIMI_URL']
KIMI_MODEL = os.environ.get('KIMI_MODEL', '/models/kimi-k3')
# comma dongle ids allowed to talk to this server, any registered comma device when unset
ALLOWED = {d for d in os.environ.get('VOICE_DONGLES', '').split(',') if d}
COMMA_API = 'https://api.commadotai.com'
VOICE = os.environ.get('PIPER_VOICE', os.path.expanduser('~/.local/share/piper/glados/glados.onnx'))
PORT = int(os.environ.get('PORT', '8765'))
WAKE = 'clanker'
# whisper reads the last WIN samples every ASR_HOP, a query ends after END_GAP without new words, END_GAP_OPEN without an end of sentence
ASR_HOP, WIN, RING = int(0.2 * 16000), 6 * 16000, 12 * 16000
END_GAP, END_GAP_OPEN = int(0.3 * 16000), int(0.6 * 16000)
# quiet time after which the llm is asked ahead of the end of the query
SPEC_GAP = int(0.2 * 16000)
# how long a bare wake word waits for the question, longest query
BARE_WAIT, MAX_S, FOLLOWUP_S = 1.5, 10.0, 4.0
# forced wake word decode log prob that counts as a wake word
FORCED_MIN, FORCED_GAIN = -0.25, 0.1


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
  for m in words:
    w = m.group().lower()
    # 3 edits matched chance, corner and later in fleet chatter
    if len(w) >= 5 and _edits(w, WAKE) <= 2:
      return m
  return None
SYSTEM = "You are clanker, the voice assistant in a car running openpilot. The driver is driving: answer in one or two short spoken sentences, no lists, no markdown, no reasoning. Never tell the driver to look at a phone or screen. If you lack live data, say so in a few words."

whisper = WhisperModel(os.environ.get('WHISPER_MODEL', 'large-v3-turbo'), device='cuda', compute_type='float16', cpu_threads=2)
whisper.transcribe(np.zeros(16000, np.float32))
tts = PiperVoice.load(VOICE)
reverse_geocoder.search((0.0, 0.0), mode=1)
TTS_SR = tts.config.sample_rate
HEDGE_S = 0.9
# live streams skip audio whisper fell behind on, offline evals feed everything at once and need every hop
SKIP_BACKLOG = True
PCM_CHUNK = int(0.25 * TTS_SR) * 2


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


def log_event(kind: str, **kw) -> None:
  # one json line per event, the time matches the REC_DIR audio file names
  entry = json.dumps({'time': round(time.time(), 3), 'kind': kind, **kw})
  print(entry, flush=True)
  if os.environ.get('WAKE_LOG'):
    with open(os.environ['WAKE_LOG'], 'a') as f:
      f.write(entry + '\n')


def frame(t: int, payload: bytes) -> bytes:
  return struct.pack('<BI', t, len(payload)) + payload


def transcribe(pcm: np.ndarray) -> str:
  # the comma four mic sits around -55 dBFS, whisper drops quiet starts
  pcm = pcm / (np.abs(pcm).max() + 1e-6) * 0.9
  segs, _ = whisper.transcribe(pcm, language='en', beam_size=1, temperature=0.0, vad_filter=False)
  return ' '.join(s.text for s in segs).strip()


def strip_wake(text: str) -> str:
  m = find_wake(text)
  return text[m.end():].lstrip(' ,.!?') if m else text


async def _kimi(session, body, timeout: float) -> str:
  async with session.post(KIMI_URL, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
    return (await r.json())['choices'][0]['message']['content'] or ''


_session = None


def ask(query: str, meta: dict) -> asyncio.Task:
  global _session
  if _session is None:
    _session = aiohttp.ClientSession()
  return asyncio.create_task(kimi(_session, query, describe(meta.get('context', {}))))


async def kimi(session, query: str, ctx: str = '') -> str:
  # whole replies take ~0.4 s so they are not streamed: kimi sometimes leaks its reasoning before a <|...|> marker,
  # and a request that hangs gets a second one raced against it
  body = {'model': KIMI_MODEL, 'max_tokens': 400, 'temperature': 0.6, 'chat_template_kwargs': {'thinking': False},
          'messages': [{'role': 'system', 'content': SYSTEM + ('\n\nLive data from the car, use it when relevant:\n' + ctx if ctx else '')}, {'role': 'user', 'content': query}]}

  async def one() -> str:
    text = await _kimi(session, body, 4)
    text = re.split(r'<\|[^|]*\|>', text)[-1].strip()
    if not text:
      raise ValueError('empty reply')
    return text

  tasks = [asyncio.create_task(one())]
  for hedge in (HEDGE_S, HEDGE_S, 2.0):
    done, _ = await asyncio.wait(tasks, timeout=hedge, return_when=asyncio.FIRST_COMPLETED)
    for d in done:
      tasks.remove(d)
      if d.cancelled() or d.exception() is not None:
        log_event('llm_error', error='cancelled' if d.cancelled() else repr(d.exception()))
      else:
        for o in tasks:
          o.cancel()
        return d.result()
    tasks.append(asyncio.create_task(one()))
  for o in tasks:
    o.cancel()
  log_event('llm_error', error='timeout', query=query)
  return "Sorry, I can't reach my brain right now."


def synth(text: str) -> bytes:
  return b''.join(c.audio_int16_bytes for c in tts.synthesize(text, syn_config=SynthesisConfig(length_scale=0.95)))


YES_PCM = synth('Yes?')


async def respond(writer, text, meta, t, query='', bare=False, spec=None) -> str:
  # the device reports its playback timing back with this id
  rid = f'{time.time():.3f}'
  writer.write(frame(10, json.dumps({'transcript': text, 'verified': True, 'query': query, 'id': rid}).encode()))
  await writer.drain()
  t['verified'] = time.monotonic()
  meta['last_reply'] = {'id': rid, 'gap': t.get('gap', 0.), 'verified': round(t['verified'] - t['eou'], 3)}
  if bare:
    log_event('bare', id=rid, dongle=meta.get('dongle_id'), transcript=text)
    # bare wake word, ask and let the device listen once more without the wake word
    writer.write(frame(12, json.dumps({'sample_rate': TTS_SR, 'say': 'Yes?'}).encode()))
    writer.write(frame(11, YES_PCM))
    writer.write(frame(13, json.dumps({'followup': True}).encode()))
    await writer.drain()
    return 'followup'
  writer.write(frame(12, json.dumps({'sample_rate': TTS_SR}).encode()))
  reply = await (spec if spec is not None else ask(query, meta))
  t['llm'] = time.monotonic()
  # speak sentence by sentence so the first one goes out while the rest is synthesized
  for sent in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', reply):
    if not sent.strip():
      continue
    pcm_out = await asyncio.to_thread(synth, sent)
    t.setdefault('tts_first', time.monotonic())
    # the text goes right before its audio so the caption follows the voice
    writer.write(frame(12, json.dumps({'say': sent.strip()}).encode()))
    # small chunks so the device starts playing before the whole sentence is through the network
    for i in range(0, len(pcm_out), PCM_CHUNK):
      writer.write(frame(11, pcm_out[i:i + PCM_CHUNK]))
      await writer.drain()
  lat = {k: round(v - t['eou'], 3) if k != 'gap' else v for k, v in t.items() if k != 'eou'}
  writer.write(frame(12, json.dumps({'reply': reply, 'latency': lat}).encode()))
  writer.write(frame(13, b'{}'))
  await writer.drain()
  log_event('reply', id=rid, dongle=meta.get('dongle_id'), transcript=text, query=query, reply=reply, latency=lat,
            context=meta.get('context', {}))
  return 'done'


def words(pcm: np.ndarray, t0: int) -> list[tuple[str, int, int]]:
  # whisper words with absolute sample positions
  if len(pcm) < 8000:
    return []
  pcm = pcm / (np.abs(pcm).max() + 1e-6) * 0.9
  segs, _ = whisper.transcribe(pcm, language='en', beam_size=1, temperature=0.0, vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
  # whisper's own silence rule, what it writes over noise is mostly made up
  return [(w.word, t0 + int(w.start * 16000), t0 + int(w.end * 16000)) for g in segs for w in (g.words or [])
          if not (g.no_speech_prob > 0.6 and g.avg_logprob < -1.0)]


# what whisper makes up from road noise
NOISE_TEXT = {'thanks for watching', 'thank you for watching', "i'll see you next time", "we'll see you next time", "we'll be right back", 'take care'}
NOISE_WORDS = {'you', 'thank', 'thanks', 'oh', 'uh', 'um', 'hmm', 'bye', 'okay', 'ok', 'yes', 'yeah', 'so', 'no'}


def is_noise(text: str) -> bool:
  t = re.sub(r"[^a-z' ]", '', text.lower()).strip()
  return t in NOISE_TEXT or set(t.split()) <= NOISE_WORDS


def wake_at(ws: list[tuple[str, int, int]], after: int) -> int | None:
  # index of the last wake word that starts after the last handled request
  hits = [i for i, (w, s0, _) in enumerate(ws) if s0 >= after and find_wake(w)]
  return hits[-1] if hits else None


def sounds_like_wake(ws: list[tuple[str, int, int]], k: int) -> bool:
  w = re.sub(r"[^a-z]", '', ws[k][0].lower())
  nxt = re.sub(r"[^a-z]", '', ws[k + 1][0].lower()) if k + 1 < len(ws) else ''
  return len(w) >= 3 and (_edits(w, WAKE) <= 4 or _edits(w + nxt, WAKE) <= 3)


def forced_wake(pcm: np.ndarray) -> bool:
  # made to start with the wake word, whisper is much surer of a real one (-0.3 -> -0.02) and not of 'thank you' or clear radio speech
  pcm = pcm / (np.abs(pcm).max() + 1e-6) * 0.9
  lp = []
  for prefix in ('Clanker,', None):
    segs, _ = whisper.transcribe(pcm, language='en', beam_size=1, temperature=0.0, vad_filter=False, condition_on_previous_text=False, prefix=prefix, without_timestamps=True)
    segs = list(segs)
    lp.append(float(np.mean([g.avg_logprob for g in segs])) if segs else -9.0)
  return lp[0] > FORCED_MIN and lp[0] - lp[1] > FORCED_GAIN


async def rescue_wake(ws, ring, base, after, checked) -> int | None:
  # whisper hears a quick 'clanker' as 'thank you' or 'clint' depending on what follows, check once per word with the wake word forced
  for k, (w, a, _) in enumerate(ws):
    # word times wobble a little between windows, key them to 0.3 s
    key = a // 4800
    if a < after or not sounds_like_wake(ws, k):
      continue
    seen = [checked[j] for j in (key - 1, key, key + 1) if j in checked]
    if any(ok for ok, _ in seen):
      return k
    # a no only holds for the audio it saw, check again once more words came in
    if seen and max(end for _, end in seen) >= ws[-1][2] - 4800:
      continue
    ok = await asyncio.to_thread(forced_wake, ring[max(0, a - base - 4800):].copy())
    if not ok and key not in checked and not all(is_noise(x) for x, *_ in ws):
      # sounded like the wake word but whisper was not sure, where to look for misses
      log_event('near_miss', word=w.strip(), text=''.join(x for x, *_ in ws).strip())
    checked[key] = (ok, ws[-1][2])
    if ok:
      return k
  return None


async def stream(reader, writer, meta):
  # the device streams the mic nonstop and whisper reads the last few seconds every ASR_HOP: a wake word that starts after
  # the last handled request opens a query, and the query is done once whisper hears no new words for END_GAP
  ring = np.zeros(0, np.float32)
  base = pos = handled = last_asr = 0
  followup_until, woke = -1, -1
  checked: dict[int, tuple[bool, int]] = {}
  spec = None
  rec, rec_t = None, 0.0
  # how far the audio lags real time on the network, and how stale it is when whisper reads it
  t_conn, offset, last_rx = time.monotonic(), None, 0.
  link = {'net_max': 0., 'stale_max': 0., 'asr': [], 'next': 10 * 16000}
  q: asyncio.Queue = asyncio.Queue()

  async def read():
    # pings are answered right away, everything else queues so whisper can skip audio it fell behind on
    try:
      while True:
        typ, n = struct.unpack('<BI', await reader.readexactly(5))
        payload = await reader.readexactly(n)
        if typ == 4:
          writer.write(frame(14, payload))
          await writer.drain()
        else:
          q.put_nowait((typ, payload, time.monotonic()))
    except Exception as e:
      q.put_nowait(e)

  reader_task = asyncio.create_task(read())
  try:
    while True:
      item = await q.get()
      if isinstance(item, Exception):
        raise item
      typ, payload, rx = item
      n = len(payload)
      if typ == 1:
        # keep the auth fields and the last reply id, the device only refreshes its context
        meta = {**meta, **json.loads(payload)}
        continue
      if typ == 5:
        # device playback timing, seconds since it got the verified transcript
        d, last = json.loads(payload), meta.get('last_reply', {})
        if d.get('id') == last.get('id') and 'play_start' in d:
          # one way is half the measured round trip
          d['last_word_to_speaker'] = round(last['gap'] + last['verified'] + d.get('rtt_ms', 0) / 2000 + d['play_start'], 3)
        log_event('device', dongle=meta.get('dongle_id'), **d)
        continue
      if typ != 2 or not n:
        continue
      if os.environ.get('REC_DIR'):
        if rec is None or time.monotonic() - rec_t > 60:
          rec_t = time.monotonic()
          rec = open(os.path.join(os.environ['REC_DIR'], f'{time.time():.0f}.s16'), 'ab')
        rec.write(payload)
      ring = np.concatenate([ring, np.frombuffer(payload, np.int16).astype(np.float32) / 32768])
      pos += n // 2
      behind = rx - t_conn - pos / 16000
      # the device pauses the mic while it speaks, start over from there
      if offset is None or rx - last_rx > 1.5:
        offset = behind
      offset, last_rx = min(offset, behind), rx
      link['net_max'] = max(link['net_max'], behind - offset)
      if pos >= link['next']:
        asr = link['asr'] or [0.]
        log_event('link', dongle=meta.get('dongle_id'), rtt_ms=meta.get('rtt_ms'), net_lag=round(behind - offset, 3), net_lag_max=round(link['net_max'], 3),
                  stale_max=round(link['stale_max'], 3), asr_mean=round(sum(asr) / len(asr), 3), asr_max=round(max(asr), 3), hops=len(asr))
        link = {'net_max': 0., 'stale_max': 0., 'asr': [], 'next': pos + 10 * 16000}
      if len(ring) > RING:
        base += len(ring) - RING
        ring = ring[-RING:]
      # behind on audio, read the rest first so whisper always looks at the newest
      if pos - last_asr < ASR_HOP or (SKIP_BACKLOG and not q.empty()):
        continue
      last_asr = pos
      now = time.monotonic()
      link['stale_max'] = max(link['stale_max'], now - rx)
      w0 = max(base, handled, pos - WIN)
      ws = await asyncio.to_thread(words, ring[w0 - base:].copy(), w0)
      t = {'eou': now, 'stt': time.monotonic()}
      link['asr'].append(t['stt'] - now)
      follow = pos < followup_until
      i = -1 if follow else wake_at(ws, handled)
      k_rescued = i is None
      if i is None:
        k = await rescue_wake(ws, ring, base, handled, checked)
        if k is None:
          continue
        # the misheard word and anything glued to it become the wake word
        nxt = ws[k + 1:]
        if nxt and is_noise(nxt[0][0]):
          nxt = nxt[1:]
        ws = ws[:k] + [(' Clanker,', ws[k][1], ws[k][2])] + nxt
        i = k
      if not follow and woke < handled:
        # chime and duck the fan while the driver is still talking
        woke = pos
        writer.write(frame(10, json.dumps({'wake': True}).encode()))
        await writer.drain()
        log_event('wake', dongle=meta.get('dongle_id'), word=ws[i][0].strip() if i >= 0 else '', rescued=k_rescued,
                  detect_delay=round((pos - ws[i][2]) / 16000, 2) if i >= 0 else None)
      rest = ws[i + 1:]
      query = ''.join(w for w, *_ in rest).strip()
      # whisper pads road noise with 'you' and 'thank you', those words don't keep the query open
      real = [w for w in ws[i + 1:] if not is_noise(w[0])] or ws[max(i, 0):max(i, 0) + 1]
      last_end = real[-1][2] if real else pos
      start = ws[i][1] if i >= 0 and ws else handled
      if is_noise(query) or (follow and len(query.split()) < 2):
        if follow:
          # radio chatter in the follow-up window is mostly a word or two
          continue
        # whisper filling the pause after the wake word with noise words, same as a bare wake word
        query = ''
      if not query:
        if follow or pos - last_end < BARE_WAIT * 16000:
          continue
        # a bare wake word and nothing after it, ask
        await respond(writer, ''.join(w for w, *_ in ws[max(i, 0):]).strip(), meta, t, bare=True)
        followup_until, handled = pos + int(FOLLOWUP_S * 16000), pos
        ring, base = np.zeros(0, np.float32), pos
        continue
      if spec is not None and spec.query != query:
        spec.cancel()
        spec = None
      if spec is None and pos - last_end >= SPEC_GAP:
        # ask the llm already, the query is most likely done and the answer is ready by the time it is confirmed
        spec = ask(query, meta)
        spec.query = query
      # whisper only ends a sentence with . ? or ! after a real pause, a comma or nothing means the driver may go on
      done = re.search(r'(?<!\.)[.?!]$', real[-1][0].strip()) if real else None
      if pos - last_end < (END_GAP if done else END_GAP_OPEN) and pos - start < MAX_S * 16000:
        continue
      text = ''.join(w for w, *_ in ws[max(i, 0):]).strip()
      t['gap'] = round((pos - last_end) / 16000, 2)
      followup_until = -1
      # the device stops streaming while it plays the reply, so it never hears itself
      await respond(writer, text, meta, t, query=query, spec=spec)
      spec = None
      handled = pos
      ring, base = np.zeros(0, np.float32), pos
  finally:
    reader_task.cancel()


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
  except (asyncio.IncompleteReadError, ConnectionResetError):
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
