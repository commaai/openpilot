import argparse, os, hashlib
from tinygrad.helpers import getenv, DEBUG, round_up, Timing, tqdm, fetch
from extra.hevc.hevc import parse_hevc_file_headers, untile_nv12, to_bgr, nv_gpu
from tinygrad import Tensor, dtypes, Device, Variable, TinyJit

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
    dat_nv = hevc_tensor.to("NV")
    opaque, frame_info, w, h, luma_w, luma_h, chroma_off = parse_hevc_file_headers(dat)

  frame_info = frame_info[:getenv("MAX_FRAMES", len(frame_info))]

  # move all needed data to gpu
  #all_slices = []
  with Timing("copy to gpu: "):
    opaque_nv = opaque.to("NV").contiguous().realize()
    hevc_tensor = hevc_tensor.to("NV")

  out_image_size = luma_h + (luma_h + 1) // 2, round_up(luma_w, 64)
  max_hist = max(history_sz for _, _, _, history_sz, _ in frame_info)

  # define variables
  v_pos = Variable("pos", 0, max_hist + 1)
  v_offset = Variable("offset", 0, hevc_tensor.numel()-1)
  v_sz = Variable("sz", 0, hevc_tensor.numel())
  v_i = Variable("i", 0, len(frame_info)-1)

  @TinyJit
  def decode_jit(pos:Variable, src:Tensor, data:Tensor, *hist:Tensor):
    return src.decode_hevc_frame(pos, out_image_size, data, hist).realize()

  # warm up
  history = [Tensor.empty(*out_image_size, dtype=dtypes.uint8, device="NV") for _ in range(max_hist)]
  for i in range(3):
    hevc_frame = hevc_tensor.shrink((((bound_offset:=v_offset.bind(frame_info[0][0])), bound_offset+v_sz.bind(frame_info[0][1])),))
    decode_jit(v_pos.bind(0), hevc_frame, opaque_nv[v_i.bind(0)], *history)

  out_images = []
  with Timing("decoding whole file: ", on_exit=(lambda et: f", {len(frame_info)} frames, {len(frame_info)/(et/1e9):.2f} fps")):
    for i, (offset, sz, frame_pos, history_sz, is_hist) in enumerate(frame_info):
      history = history[-max_hist:] if max_hist > 0 else []
      # TODO: this shrink should work as a slice
      hevc_frame = hevc_tensor.shrink((((bound_offset:=v_offset.bind(offset)), bound_offset+v_sz.bind(sz)),))

      outimg = decode_jit(v_pos.bind(frame_pos), hevc_frame, opaque_nv[v_i.bind(i)], *history).clone()
      out_images.append(outimg)
      if is_hist: history.append(outimg)

    Device.default.synchronize()

  if getenv("VALIDATE", 0):
    import pickle
    if dat_hash == "b813bfdbec194fd17fdf0e3ceb8cea1c":
      url = "https://github.com/nimlgen/hevc_validate_set/raw/refs/heads/main/decoded_frames_b813bfdbec194fd17fdf0e3ceb8cea1c.pkl"
      decoded_frames = pickle.load(fetch(url).open("rb"))
    else: decoded_frames = pickle.load(open(f"extra/hevc/decoded_frames_{dat_hash}.pkl", "rb"))
  else: import cv2

  for i, img in tqdm(enumerate(out_images)):
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
