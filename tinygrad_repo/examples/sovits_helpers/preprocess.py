import math
from typing import Optional, Tuple
from tinygrad import Tensor, dtypes
import librosa
import soundfile
import numpy as np
import parselmouth

class PMF0Predictor:  # from https://github.com/svc-develop-team/so-vits-svc/
  def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
    self.hop_length, self.f0_min, self.f0_max, self.sampling_rate, self.name = hop_length, f0_min, f0_max, sampling_rate, "pm"
  def interpolate_f0(self,f0):
    vuv_vector = np.zeros_like(f0, dtype=np.float32)
    vuv_vector[f0 > 0.0] = 1.0
    vuv_vector[f0 <= 0.0] = 0.0
    nzindex = np.nonzero(f0)[0]
    data = f0[nzindex]
    nzindex = nzindex.astype(np.float32)
    time_org = self.hop_length / self.sampling_rate * nzindex
    time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate
    if data.shape[0] <= 0: return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector
    if data.shape[0] == 1: return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector
    f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
    return f0,vuv_vector
  def compute_f0(self,wav,p_len=None):
    x = wav
    if p_len is None: p_len = x.shape[0]//self.hop_length
    else: assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
    time_step = self.hop_length / self.sampling_rate * 1000
    f0 = parselmouth.Sound(x, self.sampling_rate) \
                    .to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6,pitch_floor=self.f0_min, pitch_ceiling=self.f0_max) \
                    .selected_array['frequency']
    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
      f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
    f0,uv = self.interpolate_f0(f0)
    return f0
  def compute_f0_uv(self,wav,p_len=None):
    x = wav
    if p_len is None: p_len = x.shape[0]//self.hop_length
    else: assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
    time_step = self.hop_length / self.sampling_rate * 1000
    f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
      time_step=time_step / 1000, voicing_threshold=0.6,
      pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']
    pad_size=(p_len - len(f0) + 1) // 2
    if(pad_size>0 or p_len - len(f0) - pad_size>0):
      f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
    f0,uv = self.interpolate_f0(f0)
    return f0,uv

class Slicer:  # from https://github.com/svc-develop-team/so-vits-svc/
  def __init__(self, sr: int, threshold: float = -40., min_length: int = 5000, min_interval: int = 300, hop_size: int = 20, max_sil_kept: int = 5000):
    if not min_length >= min_interval >= hop_size:
      raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
    if not max_sil_kept >= hop_size:
      raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
    min_interval = sr * min_interval / 1000
    self.threshold = 10 ** (threshold / 20.)
    self.hop_size = round(sr * hop_size / 1000)
    self.win_size = min(round(min_interval), 4 * self.hop_size)
    self.min_length = round(sr * min_length / 1000 / self.hop_size)
    self.min_interval = round(min_interval / self.hop_size)
    self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)
  def _apply_slice(self, waveform, begin, end):
    if len(waveform.shape) > 1: return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
    else: return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]
  def slice(self, waveform):
    samples = librosa.to_mono(waveform) if len(waveform.shape) > 1 else waveform
    if samples.shape[0] <= self.min_length: return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
    rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
    sil_tags, silence_start, clip_start = [], None, 0
    for i, rms in enumerate(rms_list):
      if rms < self.threshold:  # Keep looping while frame is silent.
        if silence_start is None:  # Record start of silent frames.
          silence_start = i
        continue
      if silence_start is None: continue  # Keep looping while frame is not silent and silence start has not been recorded.
      # Clear recorded silence start if interval is not enough or clip is too short
      is_leading_silence = silence_start == 0 and i > self.max_sil_kept
      need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
      if not is_leading_silence and not need_slice_middle:
        silence_start = None
        continue
      if i - silence_start <= self.max_sil_kept:  # Need slicing. Record the range of silent frames to be removed.
        pos = rms_list[silence_start: i + 1].argmin() + silence_start
        sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))
        clip_start = pos
      elif i - silence_start <= self.max_sil_kept * 2:
        pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
        pos += i - self.max_sil_kept
        pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
        pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
        if silence_start == 0:
          sil_tags.append((0, pos_r))
          clip_start = pos_r
        else:
          sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
          clip_start = max(pos_r, pos)
      else:
        pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
        pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
        sil_tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
        clip_start = pos_r
      silence_start = None
    total_frames = rms_list.shape[0]
    if silence_start is not None and total_frames - silence_start >= self.min_interval:  # Deal with trailing silence.
      silence_end = min(total_frames, silence_start + self.max_sil_kept)
      pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
      sil_tags.append((pos, total_frames + 1))
    if len(sil_tags) == 0: return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}  # Apply and return slices.
    chunks = []
    if sil_tags[0][0]:
      chunks.append({"slice": False, "split_time": f"0,{min(waveform.shape[0], sil_tags[0][0] * self.hop_size)}"})
    for i in range(0, len(sil_tags)):
      if i: chunks.append({"slice": False, "split_time": f"{sil_tags[i - 1][1] * self.hop_size},{min(waveform.shape[0], sil_tags[i][0] * self.hop_size)}"})
      chunks.append({"slice": True, "split_time": f"{sil_tags[i][0] * self.hop_size},{min(waveform.shape[0], sil_tags[i][1] * self.hop_size)}"})
    if sil_tags[-1][1] * self.hop_size < len(waveform):
      chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1] * self.hop_size},{len(waveform)}"})
    chunk_dict = {}
    for i in range(len(chunks)): chunk_dict[str(i)] = chunks[i]
    return chunk_dict

# sinc_interp_hann audio resampling
class Resample:
  def __init__(self, orig_freq:int=16000, new_freq:int=16000, lowpass_filter_width:int=6, rolloff:float=0.99, beta:Optional[float]=None, dtype:Optional[dtypes]=None):
    self.orig_freq, self.new_freq, self.lowpass_filter_width, self.rolloff, self.beta = orig_freq, new_freq, lowpass_filter_width, rolloff, beta
    self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
    self.kernel, self.width = self._get_sinc_resample_kernel(dtype) if self.orig_freq != self.new_freq else (None, None)
  def __call__(self, waveform:Tensor) -> Tensor:
    if self.orig_freq == self.new_freq: return waveform
    return self._apply_sinc_resample_kernel(waveform)
  def _apply_sinc_resample_kernel(self, waveform:Tensor):
    if not waveform.is_floating_point(): raise TypeError(f"Waveform tensor expected to be of type float, but received {waveform.dtype}.")
    orig_freq, new_freq = (int(self.orig_freq) // self.gcd), (int(self.new_freq) // self.gcd)
    shape = waveform.shape
    waveform = waveform.reshape(-1, shape[-1])  # pack batch
    num_wavs, length = waveform.shape
    target_length = int(math.ceil(new_freq * length / orig_freq))
    waveform = waveform.pad((self.width, self.width + orig_freq))
    resampled = waveform[:, None].conv2d(self.kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    resampled = resampled[..., :target_length]
    resampled = resampled.reshape(shape[:-1] + resampled.shape[-1:])  # unpack batch
    return resampled
  def _get_sinc_resample_kernel(self, dtype=None):
    orig_freq, new_freq = (int(self.orig_freq) // self.gcd), (int(self.new_freq) // self.gcd)
    if self.lowpass_filter_width <= 0: raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    base_freq *= self.rolloff
    width = math.ceil(self.lowpass_filter_width * orig_freq / base_freq)
    idx = Tensor.arange(-width, width + orig_freq, dtype=(dtype if dtype is not None else dtypes.float32))[None, None] / orig_freq
    t = Tensor.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clip(-self.lowpass_filter_width, self.lowpass_filter_width)
    window = (t * math.pi / self.lowpass_filter_width / 2).cos() ** 2
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = Tensor.where(t == 0, Tensor(1.0, dtype=t.dtype).to(t.device), t.sin() / t)
    kernels *= window * scale
    if dtype is None: kernels = kernels.cast(dtype=dtypes.float32)
    return kernels, width

def sinc_interp_resample(x:Tensor, orig_freq:int=16000, new_freq:int=1600, lowpass_filter_width:int=6, rolloff:float=0.99, beta:Optional[float]=None):
  resamp = Resample(orig_freq, new_freq, lowpass_filter_width, rolloff, beta, x.dtype)
  return resamp(x)

def cut(audio_path, db_thresh=-30, min_len=5000):
  audio, sr = librosa.load(audio_path, sr=None)
  slicer = Slicer(sr=sr, threshold=db_thresh, min_length=min_len)
  chunks = slicer.slice(audio)
  return chunks

def chunks2audio(audio_path, chunks):
  chunks = dict(chunks)
  audio, sr = load_audiofile(audio_path)
  if len(audio.shape) == 2 and audio.shape[1] >= 2:
    audio = audio.mean(0).unsqueeze(0)
  audio = audio.numpy()[0]
  result = []
  for k, v in chunks.items():
    tag = v["split_time"].split(",")
    if tag[0] != tag[1]:
      result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
  return result, sr

def load_audiofile(filepath:str, frame_offset:int=0, num_frames:int=-1, channels_first:bool=True):
  with soundfile.SoundFile(filepath, "r") as file_:
    frames = file_._prepare_read(frame_offset, None, num_frames)
    waveform = file_.read(frames, "float32", always_2d=True)
    sample_rate = file_.samplerate
  waveform = Tensor(waveform)
  if channels_first: waveform = waveform.transpose(0, 1)
  return waveform, sample_rate

def get_unit_f0(wav:Tensor, tran, hop_length, target_sample, f0_filter=False) -> Tuple[Tensor,Tensor,Tensor]:
  f0_predictor = PMF0Predictor(hop_length, sampling_rate=target_sample)
  f0, uv = f0_predictor.compute_f0_uv(wav.numpy())
  if f0_filter and sum(f0) == 0: raise RuntimeError("No voice detected")
  f0 = Tensor(f0.astype(np.float32)).float()
  f0 = (f0 * 2 ** (tran / 12)).unsqueeze(0)
  uv = Tensor(uv.astype(np.float32)).float().unsqueeze(0)
  wav16k = sinc_interp_resample(wav[None,:], target_sample, 16000)[0]
  return wav16k.realize(), f0.realize(), uv.realize()
