import os, time, math, functools, random
from pathlib import Path
import multiprocessing

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv, BEAM, WINO, round_up, diskcache_clear, FUSE_CONV_BW, Profiling
from tinygrad.nn.state import get_parameters, get_state_dict, safe_load, safe_save
from tinygrad.nn.optim import LAMB, LARS, SGD, OptimizerGroup, Adam

from extra.lr_scheduler import LRSchedulerGroup
from examples.mlperf.helpers import get_training_state, load_training_state
# TODO: fix benchmark logging and use tinygrad tqdm
from tqdm import tqdm

def train_resnet():
  from extra.models import resnet
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup
  from examples.mlperf.initializers import Conv2dHeNormal, Linear
  from examples.hlb_cifar10 import UnsyncedBatchNorm

  config = {}
  seed = config["seed"] = getenv("SEED", 42)
  Tensor.manual_seed(seed)  # seed for weight initialization

  INITMLPERF = getenv("INITMLPERF")
  RUNMLPERF = getenv("RUNMLPERF")
  if getenv("LOGMLPERF"):
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants
    mllog.config(filename=f"result_resnet_{seed}.txt")
    mllog.config(root_dir=Path(__file__).parents[3].as_posix())  # truncate to log this. "file": "tinygrad/examples/mlperf/model_train.py"
    MLLOGGER = mllog.get_mllogger()
    if INITMLPERF:
      # common.yaml
      MLLOGGER.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
      MLLOGGER.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      MLLOGGER.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
      MLLOGGER.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)
      # closed_common.yaml
      MLLOGGER.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.RESNET)
      diskcache_clear()
      MLLOGGER.event(key=mllog_constants.CACHE_CLEAR, value=True)
      MLLOGGER.start(key=mllog_constants.INIT_START)
    if RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.RUN_START)
      MLLOGGER.event(key=mllog_constants.SEED, value=seed)
  else:
    MLLOGGER = None

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]

  TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)

  # ** model definition and initializers **
  num_classes = 1000
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"): resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=len(GPUS))
  model = resnet.ResNet50(num_classes)

  # shard weights and initialize in order
  for k, x in get_state_dict(model).items():
    if not getenv("SYNCBN") and ("running_mean" in k or "running_var" in k):
      x.realize().shard_(GPUS, axis=0)
    else:
      x.realize().to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  epochs            = config["epochs"]            = getenv("EPOCHS", 37)
  BS                = config["BS"]                = getenv("BS", 104 * len(GPUS))  # fp32 GPUS<=6 7900xtx can fit BS=112
  EVAL_BS           = config["EVAL_BS"]           = getenv("EVAL_BS", BS)
  base_lr           = config["base_lr"]           = getenv("LR", 7.2 * (BS/1536))
  lr_warmup_epochs  = config["lr_warmup_epochs"]  = getenv("WARMUP_EPOCHS", 2)
  decay             = config["decay"]             = getenv("DECAY", 2e-4)

  loss_scaler       = config["LOSS_SCALER"]       = getenv("LOSS_SCALER", 256.0 if dtypes.default_float == dtypes.float16 else 1.0)

  target, achieved  = getenv("TARGET", 0.759), False
  eval_start_epoch  = getenv("EVAL_START_EPOCH", 0)
  eval_freq         = getenv("EVAL_FREQ", 1)

  steps_in_train_epoch  = config["steps_in_train_epoch"]  = (round_up(len(get_train_files()), BS) // BS)
  steps_in_val_epoch    = config["steps_in_val_epoch"]    = (round_up(len(get_val_files()), EVAL_BS) // EVAL_BS)

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["BEAM"]          = BEAM.value
  config["TRAIN_BEAM"]    = TRAIN_BEAM
  config["EVAL_BEAM"]     = EVAL_BEAM
  config["WINO"]          = WINO.value
  config["SYNCBN"]        = getenv("SYNCBN")

  # ** Optimizer **
  skip_list = [v for k, v in get_state_dict(model).items() if "bn" in k or "bias" in k or "downsample.1" in k]
  parameters = [x for x in parameters if x not in set(skip_list)]
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay)
  optimizer_skip = SGD(skip_list, base_lr, momentum=.9, weight_decay=0.0, classic=True)
  optimizer_group = OptimizerGroup(optimizer, optimizer_skip)

  # ** LR scheduler **
  scheduler = PolynomialDecayWithWarmup(optimizer, initial_lr=base_lr, end_lr=1e-4,
                                        train_steps=epochs * steps_in_train_epoch,
                                        warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_skip = PolynomialDecayWithWarmup(optimizer_skip, initial_lr=base_lr, end_lr=1e-4,
                                             train_steps=epochs * steps_in_train_epoch,
                                             warmup=lr_warmup_epochs * steps_in_train_epoch)
  scheduler_group = LRSchedulerGroup(scheduler, scheduler_skip)
  print(f"training with batch size {BS} for {epochs} epochs")

  # log mlperf hparams
  if MLLOGGER:
    if RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=BS)
      from extra.datasets.imagenet import get_train_files, get_val_files
      MLLOGGER.event(key=mllog_constants.TRAIN_SAMPLES, value=len(get_train_files()))
      MLLOGGER.event(key=mllog_constants.EVAL_SAMPLES, value=len(get_val_files()))

      MLLOGGER.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
      MLLOGGER.event(key=mllog_constants.OPT_NAME, value="lars")
      assert scheduler.initial_lr == scheduler_skip.initial_lr
      assert scheduler.end_lr == scheduler_skip.end_lr
      assert scheduler.power == scheduler_skip.power
      MLLOGGER.event(key=mllog_constants.LARS_OPT_BASE_LEARNING_RATE, value=scheduler.initial_lr)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_END_LR, value=scheduler.end_lr)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LR_DECAY_POLY_POWER, value=scheduler.power)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LR_DECAY_STEPS, value=epochs)
      MLLOGGER.event(key=mllog_constants.LARS_EPSILON, value=0)  # does not support epsilon != 0
      MLLOGGER.event(key=mllog_constants.LARS_OPT_LEARNING_RATE_WARMUP_EPOCHS, value=lr_warmup_epochs)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_MOMENTUM, value=optimizer.momentum)
      MLLOGGER.event(key=mllog_constants.LARS_OPT_WEIGHT_DECAY, value=optimizer.wd)

  # ** resume from checkpointing **
  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming from {ckpt} at epoch {start_epoch}")

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args)

  BENCHMARK = getenv("BENCHMARK")

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return (x.permute([0, 3, 1, 2]) - input_mean).cast(dtypes.default_float)
  @TinyJit
  def train_step(X, Y):
    optimizer_group.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    (loss * loss_scaler).backward()
    for t in optimizer_group.params: t.grad = t.grad.contiguous() / loss_scaler
    optimizer_group.step()
    scheduler_group.step()
    return loss.realize(), top_1.realize()

  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), top_1.realize()

  def fake_data_get(batch_size):
    x = Tensor.zeros(batch_size, 224, 224, 3, dtype=dtypes.uchar).contiguous()
    y = [0] * batch_size
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), y, None

  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), Tensor(y, requires_grad=False).shard(GPUS, axis=0), y, cookie

  # ** epoch loop **
  step_times = []
  for e in range(start_epoch, epochs):
    # ** train loop **
    if MLLOGGER and RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.EPOCH_START, value=e+1, metadata=dict(epoch_num=e+1))
    Tensor.training = True
    BEAM.value = TRAIN_BEAM

    if INITMLPERF:
      i, proc = 0, fake_data_get(BS)
    else:
      batch_loader = batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e, pad_first_batch=True)
      it = iter(tqdm(batch_loader, total=steps_in_train_epoch, desc=f"epoch {e}", disable=BENCHMARK))
      i, proc = 0, data_get(it)

    prev_cookies = []
    st = time.perf_counter()
    while proc is not None:
      GlobalCounters.reset()
      (loss, top_1), y, proc = train_step(proc[0], proc[1]), proc[2], proc[3]

      pt = time.perf_counter()

      if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
      try:
        if INITMLPERF:
          next_proc = fake_data_get(BS)
        else:
          next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss, top_1 = loss.numpy().item(), top_1.numpy().item()
      top_1_acc = top_1 / sum(yi != -1 for yi in y)

      cl = time.perf_counter()
      if BENCHMARK:
        step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(), "train/loss": loss, "train/top_1_acc": top_1_acc, "train/step_time": cl - st,
                   "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      prev_cookies.append(proc)
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        assert not math.isnan(loss)
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * epochs / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        print(f"epoch global_ops: {steps_in_train_epoch * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {steps_in_train_epoch * GlobalCounters.global_mem:_}")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return
    if MLLOGGER and RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.EPOCH_STOP, value=e+1, metadata=dict(epoch_num=e+1))

    # ** eval loop **
    # always eval for epoch >= 33 to stop the clock as soon as eval target hits, it can converge in epoch in [33, 37]
    if steps_in_val_epoch > 0 and ((e + 1 - eval_start_epoch) % eval_freq == 0 or e + 1 >= 33):
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.start(key=mllog_constants.EVAL_START, value=e+1, metadata=dict(epoch_num=e+1))
      if getenv("RESET_STEP", 1): train_step.reset()  # free the train step memory :(
      eval_times = []
      eval_loss = 0.0
      eval_top_1 = 0
      eval_num_samples = 0
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      if INITMLPERF:
        i, proc = 0, fake_data_get(EVAL_BS)
      else:
        it = iter(tqdm(batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False, pad_first_batch=True), total=steps_in_val_epoch))
        i, proc = 0, data_get(it)

      prev_cookies = []
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        (loss, top_1), y, proc = eval_step(proc[0], proc[1]), proc[2], proc[3]  # drop inputs, keep cookie

        if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
        try:
          if INITMLPERF:
            next_proc = fake_data_get(EVAL_BS)
          else:
            next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        loss, top_1 = loss.numpy().item(), top_1.numpy().item()
        num_samples = sum(yi != -1 for yi in y)
        eval_loss += loss * num_samples
        eval_top_1 += top_1
        eval_num_samples += num_samples
        prev_cookies.append(proc)
        proc, next_proc = next_proc, None
        i += 1
        if i == BENCHMARK:
          # assume INITMLPERF has BENCHMARK set
          if MLLOGGER and INITMLPERF:
            MLLOGGER.event(key=mllog_constants.INIT_STOP)
          return

        et = time.time()
        eval_times.append(et - st)

      if getenv("RESET_STEP", 1): eval_step.reset()
      if not BENCHMARK:
        assert eval_num_samples == len(get_val_files()), f"eval sample count mismatched. {eval_num_samples=} != {len(get_val_files())}"
      total_loss = eval_loss / eval_num_samples
      total_top_1 = eval_top_1 / eval_num_samples
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss, "eval/top_1_acc": total_top_1, "eval/forward_time": total_fw_time, "epoch": e + 1})
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.event(key=mllog_constants.EVAL_ACCURACY, value=total_top_1, metadata=dict(epoch_num=e+1))
        MLLOGGER.event(key=mllog_constants.EVAL_STOP, value=e+1, metadata=dict(epoch_num=e+1))

      # save model if achieved target
      if not achieved and total_top_1 >= target:
        # stop once achieve the target
        if MLLOGGER and RUNMLPERF:
          MLLOGGER.event(key=mllog_constants.RUN_STOP, metadata=dict(status=mllog_constants.SUCCESS))
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        fn = f"./ckpts/resnet50_{seed}.safe"
        safe_save(get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True
        break

      # checkpoint every time we eval
      if getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)

def train_retinanet():
  from contextlib import redirect_stdout
  from examples.mlperf.dataloader import batch_load_retinanet
  from examples.mlperf.initializers import FrozenBatchNorm2dRetinaNet, Conv2dNormalRetinaNet, Conv2dKaimingUniformRetinaNet, Linear, Conv2dRetinaNet
  from extra.datasets.openimages import MLPERF_CLASSES, BASEDIR, download_dataset, normalize, get_dataset_count
  from extra.models import resnet, retinanet
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  from tinygrad.helpers import colored
  from typing import Iterator

  import numpy as np

  config, target_metric = {}, 0.34

  config["SEED"] = SEED = getenv("SEED", random.SystemRandom().randint(0, 2**32 - 1))
  Tensor.manual_seed(SEED)

  NUM_CLASSES = len(MLPERF_CLASSES)
  BASEDIR = getenv("BASEDIR", BASEDIR)
  BENCHMARK = getenv("BENCHMARK")
  INITMLPERF = getenv("INITMLPERF")
  RUNMLPERF = getenv("RUNMLPERF")

  if getenv("LOGMLPERF"):
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants

    mllog.config(filename=f"result_retinanet_{SEED}.log")
    mllog.config(root_dir=Path(__file__).parents[3].as_posix())
    MLLOGGER = mllog.get_mllogger()
    MLLOGGER.logger.propagate = False

    if INITMLPERF:
      assert BENCHMARK, "BENCHMARK must be set for INITMLPERF"
      MLLOGGER.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
      MLLOGGER.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      MLLOGGER.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
      MLLOGGER.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)

      MLLOGGER.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.RETINANET)

      diskcache_clear()
      MLLOGGER.event(key=mllog_constants.CACHE_CLEAR, value=True)
      MLLOGGER.start(key=mllog_constants.INIT_START)

    if RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.RUN_START)
      MLLOGGER.event(key=mllog_constants.SEED, value=SEED)
  else:
    MLLOGGER = None

  config["gpus"] = GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 6))]

  for x in GPUS: Device[x]
  print(f"training on {GPUS}")

  def _freeze_backbone_layers(backbone:resnet.ResNet, trainable_layers:int):
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    for k, v in get_state_dict(backbone).items():
      if all([not k.startswith(layer) for layer in layers_to_train]):
        v.requires_grad = False

  def _data_get(it:Iterator[tuple[Tensor, ...]], val:bool=False):
    if val:
      x, img_ids, img_sizes, cookie = next(it)
      return x.shard(GPUS, axis=0), img_ids, img_sizes, cookie

    x, y_boxes, y_labels, matches, anchors, cookie = next(it)
    return x.shard(GPUS, axis=0), y_boxes.shard(GPUS, axis=0), y_labels.shard(GPUS, axis=0), matches.shard(GPUS, axis=0), anchors.shard(GPUS, axis=0), cookie

  def _fake_data_get(bs:int, val:bool=False):
    x = Tensor.empty(bs, 800, 800, 3, dtype=dtypes.uint8)
    if val:
      img_ids, img_sizes = [0] * bs, [(800, 800)] * bs
      return x.shard(GPUS, axis=0), img_ids, img_sizes, None

    y_boxes = Tensor.empty(bs, 120087, 4, dtype=dtypes.float32)
    y_labels = Tensor.empty(bs, 120087, dtype=dtypes.int64)
    matches = Tensor.empty(bs, 120087, dtype=dtypes.int64)
    anchors = Tensor.empty(bs, 120087, 4, dtype=dtypes.float64)
    return x.shard(GPUS, axis=0), y_boxes.shard(GPUS, axis=0), y_labels.shard(GPUS, axis=0), matches.shard(GPUS, axis=0), anchors.shard(GPUS, axis=0), None

  @TinyJit
  def _train_step(model, optim, loss_scaler, x, **kwargs):
    optim.zero_grad()

    losses = model(normalize(x, GPUS), **kwargs)
    loss = sum(losses.values())

    (loss * loss_scaler).backward()
    for t in optim.params: t.grad = t.grad / loss_scaler

    optim.step()

    return loss.realize(), losses

  @TinyJit
  def _eval_step(model, x, **kwargs):
    out = model(normalize(x, GPUS), **kwargs)
    # reassemble on GPUS[0] before sending back to CPU for speed
    return out.to(GPUS[0]).realize()

  # ** hyperparameters **
  config["BS"] = BS = getenv("BS", 16 * len(GPUS) if dtypes.default_float == dtypes.float16 else 12 * len(GPUS))
  config["EVAL_BS"] = EVAL_BS = getenv("EVAL_BS", BS)
  config["EPOCHS"] = EPOCHS = getenv("EPOCHS", 4)
  config["TRAIN_BEAM"] = TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  config["EVAL_BEAM"] = EVAL_BEAM = getenv("EVAL_BEAM", BEAM.value)
  config["LR"] = lr = getenv("LR", 9.5e-5 * (BS / 96))
  config["LOSS_SCALER"] = loss_scaler = getenv("LOSS_SCALER", 2**11 if dtypes.default_float == dtypes.float16 else 1.0)
  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["EVAL_FREQ"] = eval_freq = getenv("EVAL_FREQ", 1)

  # ** initialize wandb **
  if (WANDB:=getenv("WANDB")):
    import wandb
    wandb.init(config=config, project="MLPerf-RetinaNet")

  # ** model initializers **
  resnet.BatchNorm = FrozenBatchNorm2dRetinaNet
  resnet.Linear = Linear
  resnet.Conv2d = Conv2dRetinaNet

  retinanet.ConvHead = Conv2dNormalRetinaNet
  retinanet.ConvClassificationHeadLogits = functools.partial(Conv2dNormalRetinaNet, prior_prob=0.01)
  retinanet.ConvFPN = Conv2dKaimingUniformRetinaNet

  # ** model setup **
  backbone = resnet.ResNeXt50_32X4D(num_classes=None)
  if RUNMLPERF:
    backbone.load_from_pretrained()
  _freeze_backbone_layers(backbone, 3)

  model = retinanet.RetinaNet(backbone, num_classes=NUM_CLASSES)
  params = get_parameters(model)

  if not RUNMLPERF:
    # for init, zero out all weights
    for p in params:
      p = p.assign(Tensor.zeros_like(p).contiguous()).realize()

  if len(GPUS) > 1:
    for p in params: p.to_(GPUS)

  step_times, start_epoch = [], 0

  # ** optimizer **
  optim = Adam(params, lr=lr)

  # ** dataset **
  config["STEPS_IN_TRAIN_EPOCH"] = steps_in_train_epoch = round_up(get_dataset_count((base_dir_path:=Path(BASEDIR)), False), BS) // BS
  config["STEPS_IN_VAL_EPOCH"] = steps_in_val_epoch = (round_up(get_dataset_count(base_dir_path, True), EVAL_BS) // EVAL_BS)

  # log mlperf hparams
  if MLLOGGER:
    if RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=config["BS"])
      MLLOGGER.event(key=mllog_constants.TRAIN_SAMPLES, value=config["STEPS_IN_TRAIN_EPOCH"])
      MLLOGGER.event(key=mllog_constants.EVAL_SAMPLES, value=config["STEPS_IN_VAL_EPOCH"])
      MLLOGGER.event(key=mllog_constants.EPOCH_COUNT, value=config["EPOCHS"])
      MLLOGGER.event(key=mllog_constants.FIRST_EPOCH_NUM, value=start_epoch)

      MLLOGGER.event(key=mllog_constants.OPT_NAME, value=mllog_constants.ADAM)
      MLLOGGER.event(key=mllog_constants.OPT_BASE_LR, value=config["LR"])
      MLLOGGER.event(key=mllog_constants.OPT_WEIGHT_DECAY, value=0)
      MLLOGGER.event(key=mllog_constants.OPT_LR_WARMUP_EPOCHS, value=0)
      MLLOGGER.event(key=mllog_constants.OPT_LR_WARMUP_FACTOR, value=0)
      MLLOGGER.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)

  if RUNMLPERF:
    train_dataset = COCO(download_dataset(BASEDIR, "train"))
    val_dataset = COCO(download_dataset(BASEDIR, "validation"))
    coco_val = COCOeval(cocoGt=val_dataset, iouType="bbox")

  print(f"training with batch size {BS} for {EPOCHS} epochs")

  for e in range(start_epoch, EPOCHS):
    # ** training loop **
    if MLLOGGER and RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.EPOCH_START, value=e + 1, metadata={"epoch_num": e + 1})

    BEAM.value = TRAIN_BEAM

    if not RUNMLPERF:
      i, proc = 0, _fake_data_get(BS)
    else:
      train_dataloader = batch_load_retinanet(train_dataset, False, base_dir_path, batch_size=BS, seed=SEED)
      it = iter(tqdm(train_dataloader, total=steps_in_train_epoch, desc=f"epoch {e + 1}", disable=BENCHMARK))
      i, proc = 0, _data_get(it)

    prev_cookies = []
    st = time.perf_counter()

    while proc is not None:
      GlobalCounters.reset()

      x, y_bboxes, y_labels, matches, anchors, proc = proc
      loss, losses = _train_step(model, optim, loss_scaler, x, labels=y_labels, matches=matches, anchors=anchors, bboxes=y_bboxes)

      pt = time.perf_counter()

      if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
      try:
        if not RUNMLPERF:
          next_proc = _fake_data_get(BS)
        else:
          next_proc = _data_get(it)
      except StopIteration:
        next_proc = None

      dt = time.perf_counter()

      device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
      loss = loss.item()

      cl = time.perf_counter()
      if BENCHMARK: step_times.append(cl - st)

      if not math.isfinite(loss):
        print("loss is nan")
        return

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {losses['classification_loss'].item():5.4f} classification loss, {losses['regression_loss'].item():5.4f} regression loss, "
        f"{optim.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS"
      )

      if WANDB:
        wandb.log({"lr": optim.lr.numpy(), "train/loss": loss, "train/classification_loss": losses["classification_loss"].item(), "train/regression_loss": losses["regression_loss"].item(),
                  "train/step_time": cl - st, "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                  "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": e + (i + 1) / steps_in_train_epoch})

      st = cl
      prev_cookies.append(proc)
      proc, next_proc = next_proc, None  # return old cookie
      i += 1

      if i == BENCHMARK:
        assert not math.isnan(loss)
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * steps_in_train_epoch * EPOCHS / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        print(f"epoch global_ops: {steps_in_train_epoch * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {steps_in_train_epoch * GlobalCounters.global_mem:_}")
        # if we are doing beam search, run the first eval too
        if (TRAIN_BEAM or EVAL_BEAM) and e == start_epoch: break
        return

    if MLLOGGER and RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.EPOCH_STOP, value=e + 1, metadata={"epoch_num": e + 1})

    # ** eval loop **
    if (e + 1) % eval_freq == 0:
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.start(key=mllog_constants.EVAL_START, value=e + 1, metadata={"epoch_num": e + 1})

      BEAM.value = EVAL_BEAM

      if getenv("RESET_STEP", 1): _train_step.reset()

      with Tensor.train(mode=False), Tensor.test():
        if not RUNMLPERF:
          i, proc = 0, _fake_data_get(EVAL_BS, val=(val:=True))
        else:
          val_dataloader = batch_load_retinanet(val_dataset, (val:=True), Path(BASEDIR), batch_size=EVAL_BS, shuffle=False, seed=SEED)
          it = iter(tqdm(val_dataloader, total=steps_in_val_epoch))
          i, proc = 0, _data_get(it, val=val)
          val_img_ids, val_imgs, ncats, narea = [], [], len(coco_val.params.catIds), len(coco_val.params.areaRng)

        eval_times, prev_cookies = [], []

        while proc is not None:
          GlobalCounters.reset()
          st = time.time()

          out, img_ids, img_sizes, proc = _eval_step(model, (x:=proc[0])).numpy(), proc[1], proc[2], proc[3]

          if RUNMLPERF:
            out = model.postprocess_detections(out, input_size=x.shape[1:3], orig_image_sizes=img_sizes)
            coco_results  = [{"image_id": img_ids[i], "category_id": label, "bbox": box.tolist(), "score": score}
              for i, prediction in enumerate(out) for box, score, label in zip(*prediction.values())]

            with redirect_stdout(None):
              coco_val.cocoDt = val_dataset.loadRes(coco_results)
              coco_val.params.imgIds = img_ids
              coco_val.evaluate()

            val_img_ids.extend(img_ids)
            val_imgs.append(np.array(coco_val.evalImgs).reshape(ncats, narea, len(img_ids)))

          if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
          try:
            if not RUNMLPERF:
              next_proc = _fake_data_get(EVAL_BS, val=val)
            else:
              next_proc = _data_get(it, val=val)
          except StopIteration:
            next_proc = None

          prev_cookies.append(proc)
          proc, next_proc = next_proc, None
          i += 1

          et = time.time()
          eval_times.append(et - st)

          if i == BENCHMARK:
            # assume INITMLPERF has BENCHMARK set
            if MLLOGGER and INITMLPERF:
              MLLOGGER.event(key=mllog_constants.INIT_STOP)
            return

        if getenv("RESET_STEP", 1): _eval_step.reset()
        total_fw_time = sum(eval_times) / len(eval_times)

        if RUNMLPERF:
          coco_val.params.imgIds = val_img_ids
          coco_val._paramsEval.imgIds = val_img_ids
          coco_val.evalImgs = list(np.concatenate(val_imgs, -1).flatten())
          coco_val.accumulate()
          coco_val.summarize()

          val_metric = coco_val.stats[0]

          tqdm.write(f"eval time: {total_fw_time:.2f}, eval metric: {val_metric:.4f}")

          if WANDB:
            wandb.log({"eval/forward_time": total_fw_time, "eval/metric": val_metric, "epoch": e + 1})

          if MLLOGGER:
            MLLOGGER.event(key=mllog_constants.EVAL_ACCURACY, value=val_metric, metadata={"epoch_num": e + 1}, clear_line=True)
            MLLOGGER.end(key=mllog_constants.EVAL_STOP, value=e + 1, metadata={"epoch_num": e + 1})

          if val_metric >= target_metric:
            print(colored(f"target metric reached: {val_metric:.2f}/{target_metric:.2f}", color="green"))

            if MLLOGGER:
              MLLOGGER.end(key=mllog_constants.RUN_STOP, metadata={"status": mllog_constants.SUCCESS})

            break

def train_unet3d():
  """
  Trains the UNet3D model.

  Instructions:
  1) Run the following script from the root folder of `tinygrad`:
  ```./examples/mlperf/scripts/setup_kits19_dataset.sh```

  Optionally, `BASEDIR` can be set to download and process the dataset at a specific location:
  ```BASEDIR=<folder_path> ./examples/mlperf/scripts/setup_kits19_dataset.sh```

  2) To start training the model, run the following:
  ```time PYTHONPATH=. WANDB=1 TRAIN_BEAM=3 FUSE_CONV_BW=1 GPUS=6 BS=6 MODEL=unet3d python3 examples/mlperf/model_train.py```
  """
  from examples.mlperf.losses import dice_ce_loss
  from examples.mlperf.metrics import dice_score
  from examples.mlperf.dataloader import batch_load_unet3d
  from extra.models.unet3d import UNet3D
  from extra.datasets.kits19 import iterate, get_train_files, get_val_files, sliding_window_inference, preprocess_dataset, TRAIN_PREPROCESSED_DIR, VAL_PREPROCESSED_DIR
  from tinygrad import Context
  from tinygrad.nn.optim import SGD
  from math import ceil

  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  for x in GPUS: Device[x]

  TARGET_METRIC = 0.908
  NUM_EPOCHS = getenv("NUM_EPOCHS", 4000)
  BS = getenv("BS", 1 * len(GPUS))
  LR = getenv("LR", 2.0 * (BS / 28))
  LR_WARMUP_EPOCHS = getenv("LR_WARMUP_EPOCHS", 1000)
  LR_WARMUP_INIT_LR = getenv("LR_WARMUP_INIT_LR", 0.0001)
  WANDB = getenv("WANDB")
  PROJ_NAME = getenv("PROJ_NAME", "tinygrad_unet3d_mlperf")
  SEED = getenv("SEED", -1) if getenv("SEED", -1) >= 0 else None
  TRAIN_DATASET_SIZE, VAL_DATASET_SIZE = len(get_train_files()), len(get_val_files())
  SAMPLES_PER_EPOCH = TRAIN_DATASET_SIZE // BS
  START_EVAL_AT = getenv("START_EVAL_AT", ceil(1000 * TRAIN_DATASET_SIZE / (SAMPLES_PER_EPOCH * BS)))
  EVALUATE_EVERY = getenv("EVALUATE_EVERY", ceil(20 * TRAIN_DATASET_SIZE / (SAMPLES_PER_EPOCH * BS)))
  TRAIN_BEAM, EVAL_BEAM = getenv("TRAIN_BEAM", BEAM.value), getenv("EVAL_BEAM", BEAM.value)
  BENCHMARK = getenv("BENCHMARK")
  CKPT = getenv("CKPT")

  config = {
    "num_epochs": NUM_EPOCHS,
    "batch_size": BS,
    "learning_rate": LR,
    "learning_rate_warmup_epochs": LR_WARMUP_EPOCHS,
    "learning_rate_warmup_init": LR_WARMUP_INIT_LR,
    "start_eval_at": START_EVAL_AT,
    "evaluate_every": EVALUATE_EVERY,
    "train_beam": TRAIN_BEAM,
    "eval_beam": EVAL_BEAM,
    "wino": WINO.value,
    "fuse_conv_bw": FUSE_CONV_BW.value,
    "gpus": GPUS,
    "default_float": dtypes.default_float.name
  }

  if WANDB:
    try:
      import wandb
    except ImportError:
      raise "Need to install wandb to use it"

  if SEED is not None:
    config["seed"] = SEED
    Tensor.manual_seed(SEED)

  model = UNet3D()
  params = get_parameters(model)

  for p in params: p.realize().to_(GPUS)

  optim = SGD(params, lr=LR, momentum=0.9, nesterov=True)

  def lr_warm_up(optim, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    optim.lr.assign(Tensor([init_lr + (lr - init_lr) * scale], device=GPUS)).realize()

  def save_checkpoint(state_dict, fn):
    if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
    print(f"saving checkpoint to {fn}")
    safe_save(state_dict, fn)

  def data_get(it):
    x, y, cookie = next(it)
    return x.shard(GPUS, axis=0).realize(), y.shard(GPUS, axis=0), cookie

  @TinyJit
  @Tensor.train()
  def train_step(model, x, y):
    optim.zero_grad()

    y_hat = model(x)
    loss = dice_ce_loss(y_hat, y)

    loss.backward()
    optim.step()
    return loss.realize()

  @Tensor.train(mode=False)
  @Tensor.test()
  def eval_step(model, x, y):
    y_hat, y = sliding_window_inference(model, x, y, gpus=GPUS)
    y_hat, y = Tensor(y_hat), Tensor(y, requires_grad=False)
    loss = dice_ce_loss(y_hat, y)
    score = dice_score(y_hat, y)
    return loss.realize(), score.realize()

  if WANDB: wandb.init(config=config, project=PROJ_NAME)

  step_times, start_epoch = [], 1
  is_successful, diverged = False, False
  start_eval_at, evaluate_every = 1 if BENCHMARK else START_EVAL_AT, 1 if BENCHMARK else EVALUATE_EVERY
  next_eval_at = start_eval_at

  print(f"Training on {GPUS}")

  if BENCHMARK: print("Benchmarking UNet3D")
  else: print(f"Start evaluation at epoch {start_eval_at} and every {evaluate_every} epoch(s) afterwards")

  if not TRAIN_PREPROCESSED_DIR.exists(): preprocess_dataset(get_train_files(), TRAIN_PREPROCESSED_DIR, False)
  if not VAL_PREPROCESSED_DIR.exists(): preprocess_dataset(get_val_files(), VAL_PREPROCESSED_DIR, True)

  for epoch in range(1, NUM_EPOCHS + 1):
    with Context(BEAM=TRAIN_BEAM):
      if epoch <= LR_WARMUP_EPOCHS and LR_WARMUP_EPOCHS > 0:
        lr_warm_up(optim, LR_WARMUP_INIT_LR, LR, epoch, LR_WARMUP_EPOCHS)

      train_dataloader = batch_load_unet3d(TRAIN_PREPROCESSED_DIR, batch_size=BS, val=False, shuffle=True, seed=SEED)
      it = iter(tqdm(train_dataloader, total=SAMPLES_PER_EPOCH, desc=f"epoch {epoch}", disable=BENCHMARK))
      i, proc = 0, data_get(it)

      prev_cookies = []
      st = time.perf_counter()

      while proc is not None:
        GlobalCounters.reset()

        loss, proc = train_step(model, proc[0], proc[1]), proc[2]

        pt = time.perf_counter()

        if len(prev_cookies) == getenv("STORE_COOKIES", 1): prev_cookies = []  # free previous cookies after gpu work has been enqueued
        try:
          next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        dt = time.perf_counter()

        device_str = loss.device if isinstance(loss.device, str) else f"{loss.device[0]} * {len(loss.device)}"
        loss = loss.numpy().item()

        cl = time.perf_counter()

        if BENCHMARK: step_times.append(cl - st)

        tqdm.write(
          f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
          f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {optim.lr.numpy()[0]:.6f} LR, "
          f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS"
        )

        if WANDB:
          wandb.log({"lr": optim.lr.numpy(), "train/loss": loss, "train/step_time": cl - st, "train/python_time": pt - st, "train/data_time": dt - pt,
                     "train/cl_time": cl - dt, "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": epoch + (i + 1) / SAMPLES_PER_EPOCH})

        st = cl
        prev_cookies.append(proc)
        proc, next_proc = next_proc, None  # return old cookie
        i += 1

        if i == BENCHMARK:
          median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
          estimated_total_minutes = int(median_step_time * SAMPLES_PER_EPOCH * NUM_EPOCHS / 60)
          print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
          if (TRAIN_BEAM or EVAL_BEAM) and epoch == start_epoch: break
          return

    with Context(BEAM=EVAL_BEAM):
      if epoch == next_eval_at:
        next_eval_at += evaluate_every
        eval_loss = []
        scores = []

        for x, y in tqdm(iterate(get_val_files(), preprocessed_dir=VAL_PREPROCESSED_DIR), total=VAL_DATASET_SIZE):
          eval_loss_value, score = eval_step(model, x, y)
          eval_loss.append(eval_loss_value)
          scores.append(score)

        scores = Tensor.mean(Tensor.stack(*scores, dim=0), axis=0).numpy()
        eval_loss = Tensor.mean(Tensor.stack(*eval_loss, dim=0), axis=0).numpy()

        l1_dice, l2_dice = scores[0][-2], scores[0][-1]
        mean_dice = (l2_dice + l1_dice) / 2

        tqdm.write(f"{l1_dice} L1 dice, {l2_dice} L2 dice, {mean_dice:.3f} mean_dice, {eval_loss:5.2f} eval_loss")

        if WANDB:
          wandb.log({"eval/loss": eval_loss, "eval/mean_dice": mean_dice, "epoch": epoch})

        if mean_dice >= TARGET_METRIC:
          is_successful = True
          save_checkpoint(get_state_dict(model), "./ckpts/unet3d.safe")
        elif mean_dice < 1e-6:
          print("Model diverging. Aborting.")
          diverged = True

    if not is_successful and CKPT:
      if WANDB and wandb.run is not None:
        fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{epoch}.safe"
      else:
        fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{epoch}.safe"

      save_checkpoint(get_state_dict(model), fn)

    if is_successful or diverged:
      break

def train_rnnt():
  # TODO: RNN-T
  pass

@TinyJit
def train_step_bert(model, optimizer, scheduler, loss_scaler:float, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor,
                    masked_positions:Tensor, masked_lm_ids:Tensor, masked_lm_weights:Tensor, next_sentence_labels:Tensor, GPUS):
  for t in [input_ids, segment_ids, attention_mask, masked_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels]:
    if len(GPUS) > 1: t.shard_(GPUS, axis=0)
    else: t.to_(GPUS[0])
  optimizer.zero_grad()

  lm_logits, seq_relationship_logits = model(input_ids, attention_mask, masked_positions, segment_ids)
  loss = model.loss(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
  (loss * loss_scaler).backward()

  global_norm = Tensor([0.0], dtype=dtypes.float32, device=optimizer[0].device)
  for p in optimizer.params:
    p.grad = p.grad / loss_scaler
    global_norm += p.grad.float().square().sum()
  global_norm = global_norm.sqrt().contiguous()
  for p in optimizer.params:
    p.grad = (global_norm > 1.0).where((p.grad/global_norm).cast(p.grad.dtype), p.grad)

  optimizer.step()
  scheduler.step()
  # TODO: no to("CPU") here because it blocks and messes the python time
  Tensor.realize(loss, global_norm, optimizer.optimizers[0].lr)
  return loss, global_norm, optimizer.optimizers[0].lr

@TinyJit
def eval_step_bert(model, input_ids:Tensor, segment_ids:Tensor, attention_mask:Tensor, masked_positions:Tensor, masked_lm_ids:Tensor,
                   masked_lm_weights:Tensor, next_sentence_labels:Tensor, GPUS):
  for t in [input_ids, segment_ids, attention_mask, masked_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels]:
    if len(GPUS) > 1: t.shard_(GPUS, axis=0)
    else: t.to_(GPUS[0])
  lm_logits, seq_relationship_logits = model(input_ids, attention_mask, masked_positions, segment_ids)
  masked_lm_accuracy, seq_relationship_accuracy, masked_lm_loss, next_sentence_loss = \
    model.accuracy(lm_logits, seq_relationship_logits, masked_lm_ids, masked_lm_weights, next_sentence_labels)
  for t in [masked_lm_accuracy, seq_relationship_accuracy, masked_lm_loss, next_sentence_loss]:
    t.to_("CPU")
  Tensor.realize(masked_lm_accuracy, seq_relationship_accuracy, masked_lm_loss, next_sentence_loss)
  return masked_lm_accuracy, seq_relationship_accuracy, masked_lm_loss, next_sentence_loss

def train_bert():
  # NOTE: pip install tensorflow, wandb required
  from examples.mlperf.dataloader import batch_load_train_bert, batch_load_val_bert
  from examples.mlperf.helpers import get_mlperf_bert_model, get_fake_data_bert
  from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup

  config = {}
  BASEDIR = getenv("BASEDIR", Path(__file__).parent.parents[1] / "extra" / "datasets" / "wiki")

  GPUS = config["GPUS"] = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  print(f"training on {GPUS}")
  for x in GPUS: Device[x]
  seed = config["seed"] = getenv("SEED", 12345)

  INITMLPERF = getenv("INITMLPERF")
  RUNMLPERF = getenv("RUNMLPERF")
  BENCHMARK = getenv("BENCHMARK")
  if getenv("LOGMLPERF"):
    from mlperf_logging import mllog
    import mlperf_logging.mllog.constants as mllog_constants

    mllog.config(filename=f"result_bert_{seed}.log")
    mllog.config(root_dir=Path(__file__).parents[3].as_posix())
    MLLOGGER = mllog.get_mllogger()
    MLLOGGER.logger.propagate = False

    if INITMLPERF:
      assert BENCHMARK, "BENCHMARK must be set for INITMLPERF"
      MLLOGGER.event(key=mllog_constants.SUBMISSION_ORG, value="tinycorp")
      MLLOGGER.event(key=mllog_constants.SUBMISSION_PLATFORM, value=getenv("SUBMISSION_PLATFORM", "tinybox"))
      MLLOGGER.event(key=mllog_constants.SUBMISSION_DIVISION, value=mllog_constants.CLOSED)
      MLLOGGER.event(key=mllog_constants.SUBMISSION_STATUS, value=mllog_constants.ONPREM)

      MLLOGGER.event(key=mllog_constants.SUBMISSION_BENCHMARK, value=mllog_constants.BERT)

      diskcache_clear()
      MLLOGGER.event(key=mllog_constants.CACHE_CLEAR, value=True)
      MLLOGGER.start(key=mllog_constants.INIT_START, value=None)

    if RUNMLPERF:
      MLLOGGER.start(key=mllog_constants.RUN_START, value=None)
      MLLOGGER.event(key=mllog_constants.SEED, value=seed)
  else:
    MLLOGGER = None

  # ** hyperparameters **
  BS                 = config["GLOBAL_BATCH_SIZE"]      = getenv("BS", 11 * len(GPUS) if dtypes.default_float in (dtypes.float16, dtypes.bfloat16) else 8 * len(GPUS))
  EVAL_BS            = config["EVAL_BS"]                = getenv("EVAL_BS", 1 * len(GPUS))
  max_lr             = config["OPT_BASE_LEARNING_RATE"] = getenv("OPT_BASE_LEARNING_RATE", 0.000175 * math.sqrt(BS/96))
  opt_lamb_beta_1    = config["OPT_LAMB_BETA_1"]        = getenv("OPT_LAMB_BETA_1", 0.9)
  opt_lamb_beta_2    = config["OPT_LAMB_BETA_2"]        = getenv("OPT_LAMB_BETA_2", 0.999)

  train_steps        = config["TRAIN_STEPS"]            = getenv("TRAIN_STEPS", 3600000 // BS)
  warmup_steps       = config["NUM_WARMUP_STEPS"]       = getenv("NUM_WARMUP_STEPS", 1)
  max_eval_steps     = config["MAX_EVAL_STEPS"]         = getenv("MAX_EVAL_STEPS", (10000 + EVAL_BS - 1) // EVAL_BS) # EVAL_BS * MAX_EVAL_STEPS >= 10000
  eval_step_freq     = config["EVAL_STEP_FREQ"]         = getenv("EVAL_STEP_FREQ", int((math.floor(0.05 * (230.23 * BS + 3000000) / 25000) * 25000) / BS)) # Round down
  save_ckpt_freq     = config["SAVE_CKPT_FREQ"]         = getenv("SAVE_CKPT_FREQ", 1000)
  keep_ckpt_amount   = config["KEEP_CKPT_AMOUNT"]       = getenv("KEEP_CKPT_AMOUNT", 5)
  save_ckpt_dir      = config["SAVE_CKPT_DIR"]          = getenv("SAVE_CKPT_DIR", "./ckpts")
  init_ckpt          = config["INIT_CKPT_DIR"]          = getenv("INIT_CKPT_DIR", BASEDIR)

  loss_scaler        = config["LOSS_SCALER"]            = getenv("LOSS_SCALER", 2.0**11 if dtypes.default_float == dtypes.float16 else 1.0)
  decay              = config["DECAY"]                  = getenv("DECAY", 0.01)
  epsilon            = config["EPSILON"]                = getenv("EPSILON", 1e-6)
  poly_power         = config["POLY_POWER"]             = getenv("POLY_POWER", 1.0)

  target, achieved                                      = getenv("TARGET", 0.72), False

  config["DEFAULT_FLOAT"] = dtypes.default_float.name
  config["DISABLE_DROPOUT"] = getenv("DISABLE_DROPOUT", 0)
  config["TRAIN_BEAM"]    = TRAIN_BEAM = getenv("TRAIN_BEAM", BEAM.value)
  config["EVAL_BEAM"]     = EVAL_BEAM  = getenv("EVAL_BEAM", BEAM.value)

  Tensor.manual_seed(seed)  # seed for weight initialization

  assert 10000 <= (EVAL_BS * max_eval_steps), "Evaluation batchsize * max_eval_steps must greater or equal 10000 to iterate over full eval dataset"

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    import wandb
    wandb_args = {"id": wandb_id, "resume": "must"} if (wandb_id := getenv("WANDB_RESUME", "")) else {}
    wandb.init(config=config, **wandb_args, project="MLPerf-BERT")

  # ** init model **

  model = get_mlperf_bert_model()
  if RUNMLPERF:
    model.load_from_pretrained(init_ckpt)
  else:
    # for init, zero out all weights
    for p in get_parameters(model):
      p = p.assign(Tensor.zeros_like(p).contiguous()).realize()

  parameters = get_parameters(model)
  if len(GPUS) > 1:
    for p in parameters:
      p.to_(GPUS)

  # ** Log run config **
  for key, value in config.items(): print(f'HParam: "{key}": {value}')

  # ** Optimizer **
  parameters_no_wd = [v for k, v in get_state_dict(model).items() if "bias" in k or "LayerNorm" in k]
  parameters = [x for x in parameters if x not in set(parameters_no_wd)]
  optimizer_wd = LAMB(parameters, lr=max_lr, b1=opt_lamb_beta_1, b2=opt_lamb_beta_2, eps=epsilon, weight_decay=decay, adam=False)
  optimizer_no_wd = LAMB(parameters_no_wd, lr=max_lr, b1=opt_lamb_beta_1, b2=opt_lamb_beta_2, eps=epsilon, weight_decay=0.0, adam=False)
  optimizer_group = OptimizerGroup(optimizer_wd, optimizer_no_wd)

  # ** LR scheduler **
  scheduler_wd = PolynomialDecayWithWarmup(optimizer_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_no_wd = PolynomialDecayWithWarmup(optimizer_no_wd, max_lr, 0, train_steps, warmup_steps, power=poly_power)
  scheduler_group = LRSchedulerGroup(scheduler_wd, scheduler_no_wd)
  print(f"training with batch size {BS} for one epoch with {train_steps} steps")

  # log mlperf hparams
  if MLLOGGER:
    if RUNMLPERF:
      MLLOGGER.event(key=mllog_constants.GLOBAL_BATCH_SIZE, value=config["GLOBAL_BATCH_SIZE"])
      MLLOGGER.event(key=mllog_constants.MAX_SEQUENCE_LENGTH, value=512)
      MLLOGGER.event(key="max_predictions_per_seq", value=76)

      MLLOGGER.event(key=mllog_constants.OPT_NAME, value="LAMB")
      MLLOGGER.event(key=mllog_constants.OPT_BASE_LR, value=config["OPT_BASE_LEARNING_RATE"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_WEIGHT_DECAY, value=config["DECAY"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_BETA_1, value=config["OPT_LAMB_BETA_1"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_BETA_2, value=config["OPT_LAMB_BETA_2"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=config["POLY_POWER"])
      MLLOGGER.event(key=mllog_constants.OPT_LAMB_EPSILON, value=config["EPSILON"])

      MLLOGGER.event(key=mllog_constants.OPT_LR_WARMUP_STEPS, value=config["NUM_WARMUP_STEPS"])
      MLLOGGER.event(key=mllog_constants.NUM_WARMUP_STEPS, value=config["NUM_WARMUP_STEPS"])
      MLLOGGER.event(key='start_warmup_step', value=0)
      MLLOGGER.event(key='opt_learning_rate_training_steps', value=config["TRAIN_STEPS"])
      MLLOGGER.event(key=mllog_constants.GRADIENT_ACCUMULATION_STEPS, value=1)
      MLLOGGER.event(key=mllog_constants.EVAL_SAMPLES, value=config["EVAL_BS"] * config["MAX_EVAL_STEPS"])
      MLLOGGER.event(key=mllog_constants.TRAIN_SAMPLES, value=config["GLOBAL_BATCH_SIZE"] * config["TRAIN_STEPS"])

  # ** resume from checkpointing **
  start_step = 0
  previous_step = None
  if ckpt:=getenv("RESUME", ""):
    load_training_state(model, optimizer_group, scheduler_group, safe_load(ckpt))
    start_step = int(scheduler_wd.epoch_counter.item())
    print(f"resuming from {ckpt} at step {start_step}")

  if RUNMLPERF:
    # only load real data with RUNMLPERF
    eval_it = iter(batch_load_val_bert(EVAL_BS))
    train_it = iter(tqdm(batch_load_train_bert(BS), total=train_steps, disable=BENCHMARK))
    for _ in range(start_step): next(train_it) # Fast forward
  else:
    # repeat fake data
    def repeat_fake(bs):
      while True: yield get_fake_data_bert(bs)
    eval_it = iter(repeat_fake(EVAL_BS))
    train_it = iter(repeat_fake(BS))

  step_times = []
  # ** train loop **
  wc_start = time.perf_counter()

  i, train_data = start_step, next(train_it)

  if RUNMLPERF:
    if MLLOGGER:
      MLLOGGER.start(key=mllog_constants.EPOCH_START, value=i*BS, metadata={"epoch_num": i*BS})

  while train_data is not None and i < train_steps and not achieved:
    if getenv("TRAIN", 1):
      Tensor.training = True
      BEAM.value = TRAIN_BEAM
      st = time.perf_counter()
      GlobalCounters.reset()
      loss, global_norm, lr = train_step_bert(model, optimizer_group, scheduler_group, loss_scaler,
        train_data["input_ids"], train_data["segment_ids"], train_data["input_mask"], train_data["masked_lm_positions"], \
        train_data["masked_lm_ids"], train_data["masked_lm_weights"], train_data["next_sentence_labels"], GPUS)

      pt = time.perf_counter()

      try:
        next_data = next(train_it)
      except StopIteration:
        next_data = None

      dt = time.perf_counter()

      device_str = parameters[0].device if isinstance(parameters[0].device, str) else f"{parameters[0].device[0]} * {len(parameters[0].device)}"
      loss = loss.item()
      assert not math.isnan(loss)
      lr = lr.item()

      cl = time.perf_counter()
      if BENCHMARK: step_times.append(cl - st)

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(pt - st) * 1000.0:7.2f} ms python, {(dt - pt) * 1000.0:6.2f} ms fetch data, "
        f"{(cl - dt) * 1000.0:7.2f} ms {device_str}, {loss:5.2f} loss, {lr:.6f} LR, "
        f"{GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": lr, "train/loss": loss, "train/global_norm": global_norm.item(), "train/step_time": cl - st,
                    "train/python_time": pt - st, "train/data_time": dt - pt, "train/cl_time": cl - dt,
                    "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st), "epoch": (i+1)*BS})

      train_data, next_data = next_data, None
      i += 1

      if i == BENCHMARK:
        median_step_time = sorted(step_times)[(BENCHMARK + 1) // 2]  # in seconds
        estimated_total_minutes = int(median_step_time * train_steps / 60)
        print(f"Estimated training time: {estimated_total_minutes // 60}h{estimated_total_minutes % 60}m")
        print(f"epoch global_ops: {train_steps * GlobalCounters.global_ops:_}, "
              f"epoch global_mem: {train_steps * GlobalCounters.global_mem:_}")

    # ** eval loop **
    if i % eval_step_freq == 0 or (BENCHMARK and i == BENCHMARK) or i == train_steps:
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.start(key=mllog_constants.EVAL_START, value=None, metadata={"epoch_num": i*BS, "step_num": i})
      if getenv("RESET_STEP"): train_step_bert.reset()
      elif getenv("FREE_INTERMEDIATE", 1) and train_step_bert.captured is not None: train_step_bert.captured.free_intermediates()
      eval_lm_losses = []
      eval_clsf_losses = []
      eval_lm_accs = []
      eval_clsf_accs = []
      eval_times = []
      Tensor.training = False
      BEAM.value = EVAL_BEAM

      for j in tqdm(range(max_eval_steps), desc="Evaluating", total=max_eval_steps, disable=BENCHMARK):
        eval_data = next(eval_it)
        GlobalCounters.reset()
        st = time.time()

        lm_acc, clsf_acc, lm_loss, clsf_loss = eval_step_bert(model,
          eval_data["input_ids"], eval_data["segment_ids"], eval_data["input_mask"], eval_data["masked_lm_positions"],
          eval_data["masked_lm_ids"], eval_data["masked_lm_weights"], eval_data["next_sentence_labels"], GPUS)
        lm_acc, clsf_acc, lm_loss, clsf_loss = lm_acc.item(), clsf_acc.item(), lm_loss.item(), clsf_loss.item()

        eval_lm_losses.append(lm_loss)
        eval_clsf_losses.append(clsf_loss)
        eval_lm_accs.append(lm_acc)
        eval_clsf_accs.append(clsf_acc)

        et = time.time()
        eval_times.append(et - st)

        if BENCHMARK and (j+1) == min(BENCHMARK, max_eval_steps):
          # assume INITMLPERF has BENCHMARK set
          if MLLOGGER and INITMLPERF:
            MLLOGGER.event(key=mllog_constants.INIT_STOP, value=None)
          return

      if getenv("RESET_STEP"): eval_step_bert.reset()
      elif getenv("FREE_INTERMEDIATE", 1) and eval_step_bert.captured is not None: eval_step_bert.captured.free_intermediates()

      del eval_data
      avg_lm_loss = sum(eval_lm_losses) / len(eval_lm_losses)
      avg_clsf_loss = sum(eval_clsf_losses) / len(eval_clsf_losses)
      avg_lm_acc = sum(eval_lm_accs) / len(eval_lm_accs)
      avg_clsf_acc = sum(eval_clsf_accs) / len(eval_clsf_accs)
      avg_fw_time = sum(eval_times) / len(eval_times)
      results = f"eval lm loss: {avg_lm_loss:.2f}, eval clsf loss: {avg_clsf_loss:.2f}, eval lm accuracy: {avg_lm_acc:.6f}, \
                  eval clsf accuracy: {avg_clsf_acc:.2f}, avg eval step time: {avg_fw_time:.2f}"
      tqdm.write(results)

      if WANDB:
        wandb.log({"eval/lm_loss": avg_lm_loss, "eval/clsf_loss": avg_clsf_loss, "eval/lm_accuracy": avg_lm_acc, \
                    "eval/clsf_accuracy": avg_clsf_acc, "eval/forward_time": avg_fw_time, "epoch": (i+1)*BS})

      if MLLOGGER and RUNMLPERF:
        MLLOGGER.end(key=mllog_constants.EVAL_STOP, value=i*BS, metadata={"epoch_count": i*BS, "step_num": i, "samples_count": config["EVAL_BS"] * config["MAX_EVAL_STEPS"]})
        MLLOGGER.event(key=mllog_constants.EVAL_ACCURACY, value=avg_lm_acc, metadata={"epoch_num": i*BS, "masked_lm_accuracy": avg_lm_acc})

      # save model if achieved target
      if not achieved and avg_lm_acc >= target:
        wc_end = time.perf_counter()
        if getenv("CKPT"):
          if not os.path.exists(ckpt_dir := save_ckpt_dir): os.mkdir(ckpt_dir)
          fn = f"{ckpt_dir}/bert-large.safe"
          safe_save(get_state_dict(model), fn)
          print(f" *** Model saved to {fn} ***")

        total_seconds = wc_end - wc_start
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        print(f"Reference Convergence point reached after {i * BS} datasamples and {hours}h{minutes}m{seconds:.2f}s.")
        achieved = True
        if MLLOGGER and RUNMLPERF:
          MLLOGGER.event(key=mllog_constants.EPOCH_STOP, value=i*BS, metadata={"epoch_num": i*BS})
          MLLOGGER.end(key=mllog_constants.RUN_STOP, metadata=dict(status=mllog_constants.SUCCESS))
        # stop once hitting the target
        break

    # should not happen, BENCHMARK not properly terminated
    if BENCHMARK: assert i < BENCHMARK, i

    if getenv("CKPT") and i % save_ckpt_freq == 0:
      if MLLOGGER and RUNMLPERF:
        if previous_step:
          MLLOGGER.end(key=mllog_constants.BLOCK_STOP, value=None, metadata={"first_epoch_num": 1, "epoch_num": 1, "first_step_num": i, "step_num": i, "step_count": i - previous_step})
        MLLOGGER.start(key="checkpoint_start", value=None, metadata={"step_num": i})
      if not os.path.exists(ckpt_dir := save_ckpt_dir): os.mkdir(ckpt_dir)
      if WANDB and wandb.run is not None:
        fn = f"{ckpt_dir}/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}.safe"
      else:
        fn = f"{ckpt_dir}/{time.strftime('%Y%m%d_%H%M%S')}.safe"
      print(f"saving ckpt to {fn}")
      safe_save(get_training_state(model, optimizer_group, scheduler_group), fn)
      ckpt_files = [f for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f))]
      ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)))
      while len(ckpt_files) > keep_ckpt_amount:
        last = ckpt_files.pop(0)
        print(f"Removing old ckpt {last}")
        os.remove(os.path.join(ckpt_dir, last))
      if MLLOGGER and RUNMLPERF:
        MLLOGGER.end(key="checkpoint_stop", value=None, metadata={"step_num": i})
        MLLOGGER.start(key=mllog_constants.BLOCK_START, value=None, metadata={"first_epoch_num": 1, "epoch_num": 1, "epoch_count": 1, "samples_count": i * BS, "step_num": i, "first_step_num": i+1})
        previous_step = i

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn')
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        with Profiling(enabled=getenv("PYPROFILE")): globals()[nm]()
