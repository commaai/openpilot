import json
import pathlib
import zipfile
import numpy as np
from tinygrad.helpers import fetch
import pycocotools._mask as _mask
from examples.mask_rcnn import Masker
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

iou         = _mask.iou
merge       = _mask.merge
frPyObjects = _mask.frPyObjects

BASEDIR = pathlib.Path(__file__).parent / "COCO"
BASEDIR.mkdir(exist_ok=True)

def create_dict(key_row, val_row, rows): return {row[key_row]:row[val_row] for row in rows}


if not pathlib.Path(BASEDIR/'val2017').is_dir():
  fn = fetch('http://images.cocodataset.org/zips/val2017.zip')
  with zipfile.ZipFile(fn, 'r') as zip_ref:
    zip_ref.extractall(BASEDIR)
  fn.unlink()


if not pathlib.Path(BASEDIR/'annotations').is_dir():
  fn = fetch('http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
  with zipfile.ZipFile(fn, 'r') as zip_ref:
    zip_ref.extractall(BASEDIR)
  fn.unlink()

with open(BASEDIR/'annotations/instances_val2017.json', 'r') as f:
  annotations_raw = json.loads(f.read())
images = annotations_raw['images']
categories = annotations_raw['categories']
annotations = annotations_raw['annotations']
file_name_to_id = create_dict('file_name', 'id', images)
id_to_width = create_dict('id', 'width', images)
id_to_height = create_dict('id', 'height', images)
json_category_id_to_contiguous_id = {v['id']: i + 1 for i, v in enumerate(categories)}
contiguous_category_id_to_json_id = {v:k for k,v in json_category_id_to_contiguous_id.items()}


def encode(bimask):
  if len(bimask.shape) == 3:
    return _mask.encode(bimask)
  elif len(bimask.shape) == 2:
    h, w = bimask.shape
    return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]

def decode(rleObjs):
  if type(rleObjs) == list:
    return _mask.decode(rleObjs)
  else:
    return _mask.decode([rleObjs])[:,:,0]

def area(rleObjs):
  if type(rleObjs) == list:
    return _mask.area(rleObjs)
  else:
    return _mask.area([rleObjs])[0]

def toBbox(rleObjs):
  if type(rleObjs) == list:
    return _mask.toBbox(rleObjs)
  else:
    return _mask.toBbox([rleObjs])[0]


def convert_prediction_to_coco_bbox(file_name, prediction):
  coco_results = []
  try:
    original_id = file_name_to_id[file_name]
    if len(prediction) == 0:
      return coco_results

    image_width = id_to_width[original_id]
    image_height = id_to_height[original_id]
    prediction = prediction.resize((image_width, image_height))
    prediction = prediction.convert("xywh")

    boxes = prediction.bbox.numpy().tolist()
    scores = prediction.get_field("scores").numpy().tolist()
    labels = prediction.get_field("labels").numpy().tolist()

    mapped_labels = [contiguous_category_id_to_json_id[int(i)] for i in labels]

    coco_results.extend(
      [
        {
          "image_id": original_id,
          "category_id": mapped_labels[k],
          "bbox": box,
          "score": scores[k],
        }
          for k, box in enumerate(boxes)
      ]
    )
  except Exception as e:
    print(file_name, e)
  return coco_results

masker = Masker(threshold=0.5, padding=1)

def convert_prediction_to_coco_mask(file_name, prediction):
  coco_results = []
  try:
    original_id = file_name_to_id[file_name]
    if len(prediction) == 0:
      return coco_results

    image_width = id_to_width[original_id]
    image_height = id_to_height[original_id]
    prediction = prediction.resize((image_width, image_height))
    masks = prediction.get_field("mask")

    scores = prediction.get_field("scores").numpy().tolist()
    labels = prediction.get_field("labels").numpy().tolist()

    masks = masker([masks], [prediction])[0].numpy()

    rles = [
      encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
      for mask in masks
    ]
    for rle in rles:
      rle["counts"] = rle["counts"].decode("utf-8")

    mapped_labels = [contiguous_category_id_to_json_id[int(i)] for i in labels]

    coco_results.extend(
      [
        {
          "image_id": original_id,
          "category_id": mapped_labels[k],
          "segmentation": rle,
          "score": scores[k],
        }
          for k, rle in enumerate(rles)
      ]
    )
  except Exception as e:
    print(file_name, e)
  return coco_results



def accumulate_predictions_for_coco(coco_results, json_result_file, rm=False):
  path = pathlib.Path(json_result_file)
  if rm and path.exists(): path.unlink()
  with open(path, "a") as f:
    for s in coco_results:
      f.write(json.dumps(s))
      f.write('\n')

def remove_dup(l):
  seen = set()
  seen_add = seen.add
  return [x for x in l if not (x in seen or seen_add(x))]

class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NpEncoder, self).default(obj)


def evaluate_predictions_on_coco(json_result_file, iou_type="bbox"):
  coco_results = []
  with open(json_result_file, "r") as f:
    for line in f:
      coco_results.append(json.loads(line))

  coco_gt = COCO(str(BASEDIR/'annotations/instances_val2017.json'))
  set_of_json = remove_dup([json.dumps(d, cls=NpEncoder) for d in coco_results])
  unique_list = [json.loads(s) for s in set_of_json]

  with open(f'{json_result_file}.flattend', "w") as f:
    json.dump(unique_list, f)

  coco_dt = coco_gt.loadRes(str(f'{json_result_file}.flattend'))
  coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
  coco_eval.evaluate()
  coco_eval.accumulate()
  coco_eval.summarize()
  return coco_eval

def iterate(files, bs=1):
  batch = []
  for file in files:
    batch.append(file)
    if len(batch) >= bs: yield batch; batch = []
  if len(batch) > 0: yield batch; batch = []
