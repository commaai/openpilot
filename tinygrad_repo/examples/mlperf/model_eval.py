import time, math, os
start = time.perf_counter()
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, dtypes, GlobalCounters, TinyJit
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load
from tinygrad.helpers import getenv, Context, prod
from extra.bench_log import BenchEvent, WallTimeEvent
def tlog(x): print(f"{x:25s}  @ {time.perf_counter()-start:5.2f}s")

def eval_resnet():
  with WallTimeEvent(BenchEvent.FULL):
    # Resnet50-v1.5
    from extra.models.resnet import ResNet50
    tlog("imports")
    GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 6))]
    for x in GPUS: Device[x]
    tlog("got devices")    # NOTE: this is faster with rocm-smi running

    class ResnetRunner:
      def __init__(self, device=None):
        self.mdl = ResNet50()
        for x in get_parameters(self.mdl) if device else []: x.to_(device)
        if (fn:=getenv("RESNET_MODEL", "")): load_state_dict(self.mdl, safe_load(fn))
        else: self.mdl.load_from_pretrained()
        self.input_mean = Tensor([0.485, 0.456, 0.406], device=device).reshape(1, -1, 1, 1)
        self.input_std = Tensor([0.229, 0.224, 0.225], device=device).reshape(1, -1, 1, 1)
      def __call__(self, x:Tensor) -> Tensor:
        x = x.permute([0,3,1,2]).cast(dtypes.float32) / 255.0
        x -= self.input_mean
        x /= self.input_std
        return self.mdl(x).log_softmax().argmax(axis=1).realize()

    mdl = TinyJit(ResnetRunner(GPUS))
    tlog("loaded models")

    # evaluation on the mlperf classes of the validation set from imagenet
    from examples.mlperf.dataloader import batch_load_resnet
    iterator = batch_load_resnet(getenv("BS", 128*6), val=getenv("VAL", 1), shuffle=False, pad_first_batch=True)
    def data_get():
      x,y,cookie = next(iterator)
      return x.shard(GPUS, axis=0).realize(), y, cookie
    n,d = 0,0
    proc = data_get()
    tlog("loaded initial data")
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      proc = (mdl(proc[0]), proc[1], proc[2])  # this frees the images
      run = time.perf_counter()
      # load the next data here
      try: next_proc = data_get()
      except StopIteration: next_proc = None
      nd = time.perf_counter()
      y = np.array(proc[1])
      proc = (proc[0].numpy() == y) & (y != -1)  # this realizes the models and frees the cookies
      n += proc.sum()
      d += (y != -1).sum()
      et = time.perf_counter()
      tlog(f"****** {n:5d}/{d:5d}  {n*100.0/d:.2f}% -- {(run-st)*1000:7.2f} ms to enqueue, {(et-run)*1000:7.2f} ms to realize ({(nd-run)*1000:7.2f} ms fetching). {(len(proc))/(et-st):8.2f} examples/sec. {GlobalCounters.global_ops*1e-12/(et-st):5.2f} TFLOPS")
      st = et
      proc, next_proc = next_proc, None
    tlog("done")

def eval_unet3d():
  # UNet3D
  from extra.models.unet3d import UNet3D
  from extra.datasets.kits19 import iterate, sliding_window_inference, get_val_files
  from examples.mlperf.metrics import dice_score
  mdl = UNet3D()
  mdl.load_from_pretrained()
  s = 0
  st = time.perf_counter()
  for i, (image, label) in enumerate(iterate(get_val_files()), start=1):
    mt = time.perf_counter()
    pred, label = sliding_window_inference(mdl, image, label)
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    s += dice_score(Tensor(pred), Tensor(label)).mean().item()
    print(f"****** {s:.2f}/{i}  {s/i:.5f} Mean DICE score")
    st = time.perf_counter()

def eval_retinanet():
  # RetinaNet with ResNeXt50_32X4D
  from examples.mlperf.dataloader import batch_load_retinanet
  from extra.datasets.openimages import normalize, download_dataset, BASEDIR
  from extra.models.resnet import ResNeXt50_32X4D
  from extra.models.retinanet import RetinaNet
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  from contextlib import redirect_stdout
  tlog("imports")

  mdl = RetinaNet(ResNeXt50_32X4D())
  mdl.load_from_pretrained()
  tlog("loaded models")

  coco = COCO(download_dataset(base_dir:=getenv("BASEDIR", BASEDIR), 'validation'))
  coco_eval = COCOeval(coco, iouType="bbox")
  coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)
  tlog("loaded dataset")

  iterator = batch_load_retinanet(coco, True, Path(base_dir), getenv("BS", 8), shuffle=False)
  def data_get():
    x, img_ids, img_sizes, cookie = next(iterator)
    return x.to(Device.DEFAULT).realize(), img_ids, img_sizes, cookie
  n = 0
  proc = data_get()
  tlog("loaded initial data")
  st = time.perf_counter()
  while proc is not None:
    GlobalCounters.reset()
    proc = (mdl(normalize(proc[0])), proc[1], proc[2], proc[3])
    run = time.perf_counter()
    # load the next data here
    try: next_proc = data_get()
    except StopIteration: next_proc = None
    nd = time.perf_counter()
    predictions, img_ids = mdl.postprocess_detections(proc[0].numpy(), orig_image_sizes=proc[2]), proc[1]
    pd = time.perf_counter()
    coco_results  = [{"image_id": img_ids[i], "category_id": label, "bbox": box.tolist(), "score": score}
      for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())]
    with redirect_stdout(None):
      coco_eval.cocoDt = coco.loadRes(coco_results)
      coco_eval.params.imgIds = img_ids
      coco_eval.evaluate()
    evaluated_imgs.extend(img_ids)
    coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
    n += len(proc[0])
    et = time.perf_counter()
    tlog(f"****** {(run-st)*1000:7.2f} ms to enqueue, {(et-run)*1000:7.2f} ms to realize ({(nd-run)*1000:7.2f} ms fetching, {(pd-run)*1000:4.2f} ms postprocess_detections). {(len(proc))/(et-st):8.2f} examples/sec. {GlobalCounters.global_ops*1e-12/(et-st):5.2f} TFLOPS")
    st = et
    proc, next_proc = next_proc, None

  coco_eval.params.imgIds = evaluated_imgs
  coco_eval._paramsEval.imgIds = evaluated_imgs
  coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
  coco_eval.accumulate()
  coco_eval.summarize()
  tlog("done")

def eval_rnnt():
  # RNN-T
  from extra.models.rnnt import RNNT
  mdl = RNNT()
  mdl.load_from_pretrained()

  from extra.datasets.librispeech import iterate
  from examples.mlperf.metrics import word_error_rate

  LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

  c = 0
  scores = 0
  words = 0
  st = time.perf_counter()
  for X, Y in iterate():
    mt = time.perf_counter()
    tt = mdl.decode(Tensor(X[0]), Tensor([X[1]]))
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model")
    for n, t in enumerate(tt):
      tnp = np.array(t)
      _, scores_, words_ = word_error_rate(["".join([LABELS[int(tnp[i])] for i in range(tnp.shape[0])])], [Y[n]])
      scores += scores_
      words += words_
    c += len(tt)
    print(f"WER: {scores/words}, {words} words, raw scores: {scores}, c: {c}")
    st = time.perf_counter()

def eval_bert():
  # Bert-QA
  from extra.models.bert import BertForQuestionAnswering
  mdl = BertForQuestionAnswering()
  mdl.load_from_pretrained()

  @TinyJit
  def run(input_ids, input_mask, segment_ids):
    return mdl(input_ids, input_mask, segment_ids).realize()

  from extra.datasets.squad import iterate
  from examples.mlperf.helpers import get_bert_qa_prediction
  from examples.mlperf.metrics import f1_score
  from transformers import BertTokenizer

  tokenizer = BertTokenizer(str(Path(__file__).parents[2] / "extra/weights/bert_vocab.txt"))

  c = 0
  f1 = 0.0
  st = time.perf_counter()
  for X, Y in iterate(tokenizer):
    mt = time.perf_counter()
    outs = []
    for x in X:
      outs.append(run(Tensor(x["input_ids"]), Tensor(x["input_mask"]), Tensor(x["segment_ids"])).numpy())
    et = time.perf_counter()
    print(f"{(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model over {len(X)} features")

    pred = get_bert_qa_prediction(X, Y, outs)
    print(f"pred: {pred}\nans: {Y['answers']}")
    f1 += max([f1_score(pred, ans) for ans in Y["answers"]])
    c += 1
    print(f"f1: {f1/c}, raw: {f1}, c: {c}\n")

    st = time.perf_counter()

def eval_llama3():
  from extra.models.llama import Transformer
  from examples.llama3 import MODEL_PARAMS, load, convert_from_huggingface
  from tinygrad.helpers import tqdm

  BASEDIR = Path(getenv("BASEDIR", "/raid/datasets/c4/"))
  BS = getenv("BS", 4)
  SMALL = getenv("SMALL", 0)
  SEQLEN = getenv("SEQLEN", 8192)
  MODEL_PATH = Path(getenv("MODEL_PATH", "/raid/weights/llama31_8b/"))

  params = MODEL_PARAMS[getenv("LLAMA3_SIZE", "8B")]["args"]
  params = params | {"vocab_size": 32000} if not SMALL else params
  if (llama_layers:=getenv("LLAMA_LAYERS")) != 0: params['n_layers'] = llama_layers
  model = Transformer(**params, max_context=SEQLEN, jit=False, disable_kv_cache=True)

  # load weights
  weights = load(str(MODEL_PATH / "model.safetensors.index.json"))
  if "model.embed_tokens.weight" in weights:
    print("converting from huggingface format")
    weights = convert_from_huggingface(weights, params["n_layers"], params["n_heads"], params["n_kv_heads"])

  load_state_dict(model, weights, strict=False, consume=True)

  @TinyJit
  def eval_step(model, tokens):
    logits:Tensor = model(tokens[:, :-1], start_pos=0, temperature=math.nan)
    loss = logits.sparse_categorical_crossentropy(tokens[:, 1:])
    return loss.flatten().float()

  from examples.mlperf.dataloader import get_llama3_dataset, iterate_llama3_dataset
  eval_dataset = get_llama3_dataset(5760, SEQLEN, BASEDIR, val=True, small=bool(SMALL))
  iter = iterate_llama3_dataset(eval_dataset, BS)

  losses = []
  for tokens in tqdm(iter, total=5760//BS):
    GlobalCounters.reset()
    losses += eval_step(model, tokens).tolist()
    tqdm.write(f"loss: {np.mean(losses)}")

  log_perplexity = np.mean(losses)
  print(f"Log Perplexity: {log_perplexity}")

# NOTE: BEAM hangs on 8xmi300x with DECODE_BS=384 in final realize below; function is declared here for external testing
@TinyJit
def vae_decode(x:Tensor, vae, disable_beam=False) -> Tensor:
  from examples.stable_diffusion import AutoencoderKL
  assert isinstance(vae, AutoencoderKL)
  x = vae.post_quant_conv(1./0.18215 * x)

  x = vae.decoder.conv_in(x)
  x = vae.decoder.mid(x)
  for i, l in enumerate(vae.decoder.up[::-1]):
    print("decode", x.shape)
    for b in l['block']: x = b(x)
    if 'upsample' in l:
      bs,c,py,px = x.shape
      x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
      x = l['upsample']['conv'](x)
    if i == len(vae.decoder.up) - 1 and disable_beam:
      with Context(BEAM=0): x.realize()
    else: x.realize()
  x = vae.decoder.conv_out(vae.decoder.norm_out(x).swish())

  x = ((x + 1.0) / 2.0).clip(0.0, 1.0)
  return x

def eval_stable_diffusion():
  import csv, PIL, sys
  from tqdm import tqdm
  from examples.mlperf.initializers import init_stable_diffusion, gelu_erf
  from examples.stable_diffusion import AutoencoderKL
  from extra.models.unet import UNetModel
  from tinygrad.nn.state import load_state_dict, torch_load
  from tinygrad.helpers import BEAM
  from extra.models import clip
  from extra.models.clip import FrozenOpenClipEmbedder
  from extra.models.clip import OpenClipEncoder
  from extra.models.inception import FidInceptionV3

  config = {}
  GPUS               = config["GPUS"]                   = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  for x in GPUS: Device[x]
  print(f"running eval on {GPUS}")
  seed               = config["seed"]                   = getenv("SEED", 12345)
  CKPTDIR            = config["CKPTDIR"]                = Path(getenv("CKPTDIR", "./checkpoints"))
  DATADIR            = config["DATADIR"]                = Path(getenv("DATADIR", "./datasets"))
  CONTEXT_BS         = config["CONTEXT_BS"]             = getenv("CONTEXT_BS", 1 * len(GPUS))
  DENOISE_BS         = config["DENOISE_BS"]             = getenv("DENOISE_BS", 1 * len(GPUS))
  DECODE_BS          = config["DECODE_BS"]              = getenv("DECODE_BS", 1 * len(GPUS))
  INCEPTION_BS       = config["INCEPTION_BS"]           = getenv("INCEPTION_BS", 1 * len(GPUS))
  CLIP_BS            = config["CLIP_BS"]                = getenv("CLIP_BS", 1 * len(GPUS))
  EVAL_CKPT_DIR      = config["EVAL_CKPT_DIR"]          = getenv("EVAL_CKPT_DIR", "")
  STOP_IF_CONVERGED  = config["STOP_IF_CONVERGED"]      = getenv("STOP_IF_CONVERGED", 0)

  if (WANDB := getenv("WANDB", "")):
    import wandb
    wandb.init(config=config, project="MLPerf-Stable-Diffusion")

  assert EVAL_CKPT_DIR != "", "provide a directory with checkpoints to be evaluated"
  print(f"running eval on checkpoints in {EVAL_CKPT_DIR}\nSEED={seed}")
  eval_queue:list[tuple[int, Path]] = []
  for p in Path(EVAL_CKPT_DIR).iterdir():
    if p.name.endswith(".safetensors"):
      ckpt_iteration = p.name.split(".safetensors")[0]
      assert ckpt_iteration.isdigit(), f"invalid checkpoint name: {p.name}, expected <digits>.safetensors"
      eval_queue.append((int(ckpt_iteration), p))
  assert len(eval_queue), f'no files ending with ".safetensors" were found in {EVAL_CKPT_DIR}'
  print(sorted(eval_queue, reverse=True))

  Tensor.manual_seed(seed)  # seed for weight initialization
  model, unet, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = init_stable_diffusion("v2-mlperf-eval", CKPTDIR / "sd" / "512-base-ema.ckpt", GPUS)

  # load prompts for generating images for validation; 2 MB of data total
  with open(DATADIR / "coco2014" / "val2014_30k.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    eval_inputs:list[dict] = [{"image_id": int(row["image_id"]), "id": int(row["id"]), "caption": row["caption"]} for row in reader]
  assert len(eval_inputs) == 30_000
  # NOTE: the clip weights are the same between model.cond_stage_model and clip_encoder
  eval_timesteps = list(reversed(range(1, 1000, 20)))

  original_device, Device.DEFAULT = Device.DEFAULT, "CPU"
  # The choice of alphas_prev[0] = alphas_cumprod[0] seems arbitrary, but it's how the mlperf ref does it:
  #   alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
  eval_alphas_prev = model.alphas_cumprod[0:1].cat(model.alphas_cumprod[list(range(1, 1000, 20))[:-1]]).to(GPUS).realize()
  inception = FidInceptionV3().load_from_pretrained(CKPTDIR / "inception" / "pt_inception-2015-12-05-6726825d.pth")
  vision_cfg = {'width': 1280, 'layers': 32, 'd_head': 80, 'image_size': 224, 'patch_size': 14}
  text_cfg = {'width': 1024, 'n_heads': 16, 'layers': 24, 'vocab_size': 49408, 'ctx_length': 77}
  clip.gelu = gelu_erf
  clip_encoder = OpenClipEncoder(1024, text_cfg, vision_cfg)
  loaded = torch_load(CKPTDIR / "clip" / "open_clip_pytorch_model.bin")
  loaded.update({"attn_mask": clip_encoder.attn_mask, "mean": clip_encoder.mean, "std": clip_encoder.std})
  load_state_dict(clip_encoder, loaded)
  Device.DEFAULT=original_device

  @TinyJit
  def denoise_step(x:Tensor, x_x:Tensor, t_t:Tensor, uc_c:Tensor, sqrt_alphas_cumprod_t:Tensor, sqrt_one_minus_alphas_cumprod_t:Tensor,
                    alpha_prev:Tensor, unet:UNetModel, GPUS) -> Tensor:
    out_uncond, out = unet(x_x, t_t, uc_c).to("CPU").reshape(-1, 2, 4, 64, 64).chunk(2, dim=1)
    out_uncond = out_uncond.squeeze(1).shard(GPUS,axis=0)
    out = out.squeeze(1).shard(GPUS,axis=0)
    v_t = out_uncond + 8.0 * (out - out_uncond)
    e_t = sqrt_alphas_cumprod_t * v_t + sqrt_one_minus_alphas_cumprod_t * x
    pred_x0 = sqrt_alphas_cumprod_t * x - sqrt_one_minus_alphas_cumprod_t * v_t
    dir_xt = (1. - alpha_prev).sqrt() * e_t
    x_prev = alpha_prev.sqrt() * pred_x0 + dir_xt
    return x_prev.realize()

  def shard_tensor(t:Tensor) -> Tensor: return t.shard(GPUS, axis=0) if len(GPUS) > 1 else t.to(GPUS[0])
  def get_batch(whole:Tensor, i:int, bs:int) -> tuple[Tensor, int]:
    batch = whole[i: i + bs].to("CPU")
    if (unpadded_bs:=batch.shape[0]) < bs:
      batch = batch.cat(batch[-1:].expand(bs - unpadded_bs, *batch[-1].shape))
    return batch, unpadded_bs 

  @Tensor.train(mode=False)
  def eval_unet(eval_inputs:list[dict], unet:UNetModel, cond_stage:FrozenOpenClipEmbedder, first_stage:AutoencoderKL,
                inception:FidInceptionV3, clip:OpenClipEncoder) -> tuple[float, float]:
    # Eval is divided into 5 jits, one per model
    # It doesn't make sense to merge these jits, e.g. unet repeats 50 times in isolation; images fork to separate inception/clip
    # We're generating and scoring 30,000 images per eval, and all the data can flow through one jit at a time
    # To maximize throughput for each jit, we have only one model/jit on the GPU at a time, and pool outputs from each jit off-GPU
    for model in (unet, first_stage, inception, clip):
      Tensor.realize(*[p.to_("CPU") for p in get_parameters(model)])

    uc_written = False
    models = (cond_stage, unet, first_stage, inception, clip)
    jits = (jit_context:=TinyJit(cond_stage.embed_tokens), denoise_step, vae_decode, jit_inception:=TinyJit(inception),
            jit_clip:=TinyJit(clip.get_clip_score))
    all_bs = (CONTEXT_BS, DENOISE_BS, DECODE_BS, INCEPTION_BS, CLIP_BS)
    if (EVAL_SAMPLES:=getenv("EVAL_SAMPLES", 0)) and EVAL_SAMPLES > 0:
      eval_inputs = eval_inputs[0:EVAL_SAMPLES]
    output_shapes = [(ns:=len(eval_inputs),77), (ns,77,1024), (ns,4,64,64), (ns,3,512,512), (ns,2048), (ns,)]
    # Writing progress to disk lets us resume eval if we crash
    stages = ["tokens", "embeds", "latents", "imgs", "inception", "clip"]
    disk_tensor_names, disk_tensor_shapes = stages + ["end", "uc"], output_shapes + [(6,), (1,77,1024)]
    if not all(os.path.exists(f"{EVAL_CKPT_DIR}/{name}.bytes") for name in disk_tensor_names):
      for name, shape in zip(disk_tensor_names, disk_tensor_shapes):
        file = Path(f"{EVAL_CKPT_DIR}/{name}.bytes")
        file.unlink(missing_ok=True)
        with file.open("wb") as f: f.truncate(prod(shape) * 4)
    progress = {name: Tensor.empty(*shape, device=f"disk:{EVAL_CKPT_DIR}/{name}.bytes", dtype=dtypes.int if name in {"tokens", "end"} else dtypes.float)
                for name, shape in zip(disk_tensor_names, disk_tensor_shapes)}

    def embed_tokens(tokens:Tensor) -> Tensor:
      nonlocal uc_written
      if not uc_written:
        with Context(BEAM=0): progress["uc"].assign(cond_stage.embed_tokens(cond_stage.tokenize("").to(GPUS)).to("CPU").realize()).realize()
        uc_written = True
      return jit_context(shard_tensor(tokens))

    def generate_latents(embeds:Tensor) -> Tensor:
      uc_c = Tensor.stack(progress["uc"].to("CPU").expand(bs, 77, 1024), embeds, dim=1).reshape(-1, 77, 1024)
      uc_c = shard_tensor(uc_c)
      x = shard_tensor(Tensor.randn(bs,4,64,64))
      for step_idx, timestep in enumerate(tqdm(eval_timesteps)):
        reversed_idx = Tensor([50 - step_idx - 1], device=GPUS)
        alpha_prev = eval_alphas_prev[reversed_idx]
        ts = Tensor.full(bs, fill_value=timestep, dtype=dtypes.int, device="CPU")
        ts_ts = shard_tensor(ts.cat(ts))
        ts = shard_tensor(ts)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[ts].reshape(bs, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[ts].reshape(bs, 1, 1, 1)
        x_x = shard_tensor(Tensor.stack(x.to("CPU"), x.to("CPU"), dim=1).reshape(-1, 4, 64, 64))
        x.assign(denoise_step(x, x_x, ts_ts, uc_c, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t, alpha_prev, unet, GPUS)).realize()
      return x

    def decode_latents(latents:Tensor) -> Tensor: return vae_decode(shard_tensor(latents), first_stage, disable_beam=True)
    def generate_inception(imgs:Tensor) -> Tensor: return jit_inception(shard_tensor(imgs))[:,:,0,0]

    def calc_clip_scores(batch:Tensor, batch_tokens:Tensor) -> Tensor:
      # Tensor.interpolate does not yet support bicubic, so we use PIL
      batch = (batch.to(GPUS[0]).permute(0,2,3,1) * 255).clip(0, 255).cast(dtypes.uint8).numpy()
      batch = [np.array(PIL.Image.fromarray(batch[i]).resize((224,224), PIL.Image.BICUBIC)) for i in range(bs)]
      batch = shard_tensor(Tensor(np.stack(batch, axis=0).transpose(0,3,1,2), device="CPU").realize())
      batch = batch.cast(dtypes.float) / 255
      batch = (batch - model.mean) / model.std
      batch = jit_clip(shard_tensor(batch_tokens), batch)
      return batch

    callbacks = (embed_tokens, generate_latents, decode_latents, generate_inception, calc_clip_scores)

    # save every forward pass output to disk; NOTE: this needs ~100 GB disk space because 30k images are large
    def stage_progress(stage_idx:int) -> int: return progress["end"].to("CPU")[stage_idx].item()
    if stage_progress(0) < len(eval_inputs):
      tokens = []
      for i in tqdm(range(0, len(eval_inputs), CONTEXT_BS)):
        subset = [cond_stage.tokenize(row["caption"], device="CPU") for row in eval_inputs[i: i+CONTEXT_BS]]
        tokens.append(Tensor.cat(*subset, dim=0).realize())
      progress["tokens"].assign(Tensor.cat(*tokens, dim=0).realize()).realize()
      progress["end"][0:1].assign(Tensor([len(eval_inputs)], dtype=dtypes.int)).realize()
    prev_stage = "tokens"
    tokens = progress["tokens"]

    # wrapper code for every model
    for stage_idx, model, jit, bs, callback in zip(range(1,6), models, jits, all_bs, callbacks):
      stage = stages[stage_idx]
      if stage_progress(stage_idx) >= len(eval_inputs):
        prev_stage = stage
        continue # use cache
      t0 = time.perf_counter()
      print(f"starting eval with model: {model}")
      if stage_idx == 1: inputs = tokens
      elif stage_idx == 5: inputs = progress["imgs"]
      else: inputs = progress[prev_stage]

      Tensor.realize(*[p.to_(GPUS) for p in get_parameters(model)])
      for batch_idx in tqdm(range(stage_progress(stage_idx), inputs.shape[0], bs)):
        t1 = time.perf_counter()
        batch, unpadded_bs = get_batch(inputs, batch_idx, bs)
        if isinstance(model, OpenClipEncoder): batch = callback(batch, get_batch(tokens, batch_idx, bs)[0].realize())
        else: batch = callback(batch)
        # to(GPUS[0]) is necessary for this to work, without that the result is still on GPUS, probably due to a bug
        batch = batch.to(GPUS[0]).to("CPU")[0:unpadded_bs].realize()
        progress[stage][batch_idx: batch_idx + bs].assign(batch).realize()
        # keep track of what our last output was, so we can resume from there if we crash in this loop
        progress["end"][stage_idx: stage_idx + 1].assign(Tensor([batch_idx + bs], dtype=dtypes.int)).realize()
        print(f"model: {model}, batch_idx: {batch_idx}, elapsed: {(time.perf_counter() - t1):.2f}")
      del batch
        
      jit.reset()
      Tensor.realize(*[p.to_("CPU") for p in get_parameters(model)])
      print(f"done with model: {model}, elapsed: {(time.perf_counter() - t0):.2f}")
      prev_stage = stage

    inception_stats_fn = str(DATADIR / "coco2014" / "val2014_30k_stats.npz")
    fid_score = inception.compute_score(progress["inception"].to("CPU"), inception_stats_fn)
    clip_score = progress["clip"].to(GPUS[0]).mean().item()
    for name in disk_tensor_names:
      Path(f"{EVAL_CKPT_DIR}/{name}.bytes").unlink(missing_ok=True)
    
    if EVAL_SAMPLES and BEAM:
      print("BEAM COMPLETE", flush=True) # allows wrapper script to detect BEAM search completion and retry if it failed
      sys.exit() # Don't eval additional models; we don't care about clip/fid scores when running BEAM on eval sample subset

    return clip_score, fid_score

  # evaluate checkpoints in reverse chronological order
  for ckpt_iteration, p in sorted(eval_queue, reverse=True):
    unet_ckpt = safe_load(p)
    load_state_dict(unet, unet_ckpt)
    clip_score, fid_score = eval_unet(eval_inputs, unet, model.cond_stage_model, model.first_stage_model, inception, clip_encoder)
    converged = True if clip_score >= 0.15 and fid_score <= 90 else False
    print(f"eval results for {EVAL_CKPT_DIR}/{p.name}: clip={clip_score}, fid={fid_score}, converged={converged}")
    if WANDB:
      wandb.log({"eval/ckpt_iteration": ckpt_iteration, "eval/clip_score": clip_score, "eval/fid_score": fid_score})
    if converged and STOP_IF_CONVERGED:
      print(f"Convergence detected, exiting early before evaluating other checkpoints due to STOP_IF_CONVERGED={STOP_IF_CONVERGED}")
      sys.exit()

  # for testing
  return clip_score, fid_score, ckpt_iteration

if __name__ == "__main__":
  # inference only
  Tensor.training = False

  models = getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert").split(",")
  for m in models:
    nm = f"eval_{m}"
    if nm in globals():
      print(f"eval {m}")
      globals()[nm]()
