import argparse, os, hashlib, functools
from typing import Iterator, Callable
from tinygrad.helpers import getenv, DEBUG, round_up, Timing, tqdm, fetch, ceildiv
from extra.hevc.hevc import parse_hevc_file_headers, untile_nv12, to_bgr, nv_gpu
from tinygrad import Tensor, dtypes, Device, Variable, TinyJit

# rounds up hevc input data to 32 bytes, so more optimal kernels can be generated
HEVC_ROUNDUP = getenv("DATA_ROUNDUP", 32)

@functools.cache
def _hevc_jitted_decoder(out_image_size:tuple[int, int], max_hist:int, inplace:bool):
  def hevc_decode_frame(pos:Variable, hevc_tensor:Tensor, offset:Variable, sz:Variable, opaque:Tensor, i:Variable, *hist:Tensor, outbuf:Tensor|None=None):
    x = hevc_tensor[offset:offset+sz*HEVC_ROUNDUP].decode_hevc_frame(pos, out_image_size, opaque[i], hist)
    if outbuf is not None: outbuf.assign(x).realize()
    return x.realize()
  return TinyJit(hevc_decode_frame)

def hevc_decode(hevc_tensor:Tensor, opaque:Tensor, frame_info:list, luma_h:int, luma_w:int,
                history:list[Tensor]|None=None, preallocated_outputs:list[Tensor]|None=None, warmup=False) -> Iterator[Tensor]:
  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)
  max_hist = max((hs for _, _, _, hs, _ in frame_info), default=0)

  v_pos = Variable("pos", 0, max_hist + 1)
  v_offset = Variable("offset", 0, hevc_tensor.numel()-1)
  v_sz = Variable("sz", 1, ceildiv(hevc_tensor.numel(), HEVC_ROUNDUP))
  v_i = Variable("i", 0, len(frame_info)-1)

  decode_jit = _hevc_jitted_decoder(out_image_size, max_hist, preallocated_outputs is not None)
  history = history or [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(max_hist)]
  assert len(history) == max_hist, f"history length {len(history)} does not match max_hist {max_hist}"

  for i, (offset, sz, frame_pos, _, is_hist) in enumerate(frame_info):
    history = history[-max_hist:] if max_hist > 0 else []
    img = decode_jit(v_pos.bind(frame_pos), hevc_tensor, v_offset.bind(offset), v_sz.bind(ceildiv(sz, HEVC_ROUNDUP)),
                     opaque, v_i.bind(i), *history, outbuf=preallocated_outputs[i] if preallocated_outputs else None)
    res = preallocated_outputs[i] if preallocated_outputs else img.clone().realize()
    if is_hist: history.append(res)
    yield res

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str, default="")
  parser.add_argument("--output_dir", type=str, default="extra/hevc/out")
  args = parser.parse_args()

  if args.input_file == "":
    url = "https://github.com/haraschax/filedump/raw/09a497959f7fa6fd8dba501a25f2cdb3a41ecb12/comma_video.hevc"
    hevc_tensor = Tensor.from_url(url, device="CPU")
  else:
    hevc_tensor = Tensor.empty(os.stat(args.input_file).st_size, dtype=dtypes.uint8, device=f"disk:{args.input_file}").to("CPU")

  dat = bytes(hevc_tensor.data())
  dat_hash = hashlib.md5(dat).hexdigest()

  with Timing("prep infos: "):
    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat)

  frame_info = frame_info[:getenv("MAX_FRAMES", len(frame_info))]

  # move all needed data to gpu
  with Timing("copy to gpu: "):
    opaque_nv = opaque.to("NV").contiguous().realize()
    hevc_tensor = hevc_tensor.to("NV")

  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)

  # preallocate output/hist buffers
  max_hist = max((hs for _, _, _, hs, _ in frame_info), default=0)
  hist = [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(max_hist)]
  out_images = [Tensor.zeros(*out_image_size, dtype=dtypes.uint8, device="NV").contiguous().realize() for _ in range(len(frame_info))]

  # warmup decode
  _ = list(hevc_decode(hevc_tensor, opaque_nv, frame_info[:3], luma_h, luma_w, history=hist, preallocated_outputs=out_images))
  Device.default.synchronize()

  # decode all frames using the iterator
  with Timing("decoding whole file: ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
    images = list(hevc_decode(hevc_tensor, opaque_nv, frame_info, luma_h, luma_w, history=hist, preallocated_outputs=out_images))
    Device.default.synchronize()

  # validation
  if getenv("VALIDATE", 0):
    import pickle
    if dat_hash == "b813bfdbec194fd17fdf0e3ceb8cea1c":
      url = "https://github.com/nimlgen/hevc_validate_set/raw/refs/heads/main/decoded_frames_b813bfdbec194fd17fdf0e3ceb8cea1c.pkl"
      decoded_frames = pickle.load(fetch(url).open("rb"))
    else: decoded_frames = pickle.load(open(f"extra/hevc/decoded_frames_{dat_hash}.pkl", "rb"))
  else: import cv2

  for i, img in tqdm(enumerate(images)):
    if getenv("VALIDATE", 0):
      if i < len(decoded_frames) and len(decoded_frames[i]) > 0:
        img = untile_nv12(img, h, w, luma_w, chroma_off).realize()
        assert img.data() == decoded_frames[i], f"Frame {i} does not match reference decoder!"
        print(f"Frame {i} matches reference decoder!")
    else:
      if len(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        img = to_bgr(img, h, w, luma_w, chroma_off).realize()
        cv2.imwrite(f"{args.output_dir}/out_frame_{i:04d}.png", img.numpy())
