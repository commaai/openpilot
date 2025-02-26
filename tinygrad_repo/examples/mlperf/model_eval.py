import time
start = time.perf_counter()
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Device, dtypes, GlobalCounters, TinyJit
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load
from tinygrad.helpers import getenv
def tlog(x): print(f"{x:25s}  @ {time.perf_counter()-start:5.2f}s")

def eval_resnet():
  Tensor.no_grad = True
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
  from extra.models.resnet import ResNeXt50_32X4D
  from extra.models.retinanet import RetinaNet
  mdl = RetinaNet(ResNeXt50_32X4D())
  mdl.load_from_pretrained()

  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  def input_fixup(x):
    x = x.permute([0,3,1,2]) / 255.0
    x -= input_mean
    x /= input_std
    return x

  from extra.datasets.openimages import download_dataset, iterate, BASEDIR
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  from contextlib import redirect_stdout
  coco = COCO(download_dataset(base_dir:=getenv("BASE_DIR", BASEDIR), 'validation'))
  coco_eval = COCOeval(coco, iouType="bbox")
  coco_evalimgs, evaluated_imgs, ncats, narea = [], [], len(coco_eval.params.catIds), len(coco_eval.params.areaRng)

  from tinygrad.engine.jit import TinyJit
  mdlrun = TinyJit(lambda x: mdl(input_fixup(x)).realize())

  n, bs = 0, 8
  st = time.perf_counter()
  for x, targets in iterate(coco, base_dir, bs):
    dat = Tensor(x.astype(np.float32))
    mt = time.perf_counter()
    if dat.shape[0] == bs:
      outs = mdlrun(dat).numpy()
    else:
      mdlrun._jit_cache = []
      outs =  mdl(input_fixup(dat)).numpy()
    et = time.perf_counter()
    predictions = mdl.postprocess_detections(outs, input_size=dat.shape[1:3], orig_image_sizes=[t["image_size"] for t in targets])
    ext = time.perf_counter()
    n += len(targets)
    print(f"[{n}/{len(coco.imgs)}] == {(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:.2f} ms to run model, {(ext-et)*1000:.2f} ms for postprocessing")
    img_ids = [t["image_id"] for t in targets]
    coco_results  = [{"image_id": targets[i]["image_id"], "category_id": label, "bbox": box.tolist(), "score": score}
      for i, prediction in enumerate(predictions) for box, score, label in zip(*prediction.values())]
    with redirect_stdout(None):
      coco_eval.cocoDt = coco.loadRes(coco_results)
      coco_eval.params.imgIds = img_ids
      coco_eval.evaluate()
    evaluated_imgs.extend(img_ids)
    coco_evalimgs.append(np.array(coco_eval.evalImgs).reshape(ncats, narea, len(img_ids)))
    st = time.perf_counter()

  coco_eval.params.imgIds = evaluated_imgs
  coco_eval._paramsEval.imgIds = evaluated_imgs
  coco_eval.evalImgs = list(np.concatenate(coco_evalimgs, -1).flatten())
  coco_eval.accumulate()
  coco_eval.summarize()

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

def eval_mrcnn():
  from tqdm import tqdm
  from extra.models.mask_rcnn import MaskRCNN
  from extra.models.resnet import ResNet
  from extra.datasets.coco import BASEDIR, images, convert_prediction_to_coco_bbox, convert_prediction_to_coco_mask, accumulate_predictions_for_coco, evaluate_predictions_on_coco, iterate
  from examples.mask_rcnn import compute_prediction_batched, Image
  mdl = MaskRCNN(ResNet(50, num_classes=None, stride_in_1x1=True))
  mdl.load_from_pretrained()

  bbox_output = '/tmp/results_bbox.json'
  mask_output = '/tmp/results_mask.json'

  accumulate_predictions_for_coco([], bbox_output, rm=True)
  accumulate_predictions_for_coco([], mask_output, rm=True)

  #TODO: bs > 1 not as accurate
  bs = 1

  for batch in tqdm(iterate(images, bs=bs), total=len(images)//bs):
    batch_imgs = []
    for image_row in batch:
      image_name = image_row['file_name']
      img = Image.open(BASEDIR/f'val2017/{image_name}').convert("RGB")
      batch_imgs.append(img)
    batch_result = compute_prediction_batched(batch_imgs, mdl)
    for image_row, result in zip(batch, batch_result):
      image_name = image_row['file_name']
      box_pred = convert_prediction_to_coco_bbox(image_name, result)
      mask_pred = convert_prediction_to_coco_mask(image_name, result)
      accumulate_predictions_for_coco(box_pred, bbox_output)
      accumulate_predictions_for_coco(mask_pred, mask_output)
    del batch_imgs
    del batch_result

  evaluate_predictions_on_coco(bbox_output, iou_type='bbox')
  evaluate_predictions_on_coco(mask_output, iou_type='segm')

if __name__ == "__main__":
  # inference only
  Tensor.training = False
  Tensor.no_grad = True

  models = getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,mrcnn").split(",")
  for m in models:
    nm = f"eval_{m}"
    if nm in globals():
      print(f"eval {m}")
      globals()[nm]()
