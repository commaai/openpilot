import glob
import sys
import json
import numpy as np
from PIL import Image
from pathlib import Path
import boto3, botocore
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch, tqdm, getenv
import pandas as pd
import concurrent.futures

BASEDIR = Path(__file__).parent / "open-images-v6-mlperf"
BUCKET_NAME = "open-images-dataset"
TRAIN_BBOX_ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
VALIDATION_BBOX_ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
MAP_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
MLPERF_CLASSES = ['Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
  'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench', 'Bicycle',
  'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book', 'Bookcase', 'Boot',
  'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread',
  'Broccoli', 'Bronze sculpture', 'Bull', 'Bus', 'Bust', 'Butterfly', 'Cabinetry',
  'Cake', 'Camel', 'Camera', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart',
  'Castle', 'Cat', 'Cattle', 'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
  'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup', 'Coffee table', 'Coin',
  'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Convenience store',
  'Cookie', 'Countertop', 'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard',
  'Curtain', 'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door', 'Dragonfly',
  'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant',
  'Falcon', 'Fedora', 'Flag', 'Flowerpot', 'Football', 'Football helmet', 'Fork',
  'Fountain', 'French fries', 'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses',
  'Goat', 'Goggles', 'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
  'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter', 'High heels',
  'Hiking equipment', 'Horse', 'House', 'Houseplant', 'Human arm', 'Human beard',
  'Human body', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
  'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream',
  'Jacket', 'Jeans', 'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite',
  'Lamp', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb', 'Lighthouse',
  'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple', 'Microphone', 'Mirror',
  'Mixing bowl', 'Mobile phone', 'Monkey', 'Motorcycle', 'Muffin', 'Mug', 'Mule',
  'Mushroom', 'Musical keyboard', 'Necklace', 'Nightstand', 'Office building',
  'Orange', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen',
  'Penguin', 'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
  'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit', 'Rifle',
  'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Sea lion',
  'Sea turtle', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard',
  'Ski', 'Skull', 'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
  'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry', 'Street light',
  'Studio couch', 'Suit', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan',
  'Swimming pool', 'Swimwear', 'Tank', 'Tap', 'Taxi', 'Tea', 'Teddy bear', 'Television',
  'Tent', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
  'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella', 'Van', 'Vase',
  'Vehicle registration plate', 'Violin', 'Wall clock', 'Waste container', 'Watch',
  'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Window', 'Wine', 'Wine glass', 'Woman',
  'Zebra', 'Zucchini',
]


def openimages(base_dir:Path, subset:str, ann_file:Path):
  valid_subsets = ['train', 'validation']
  if subset not in valid_subsets:
    raise ValueError(f"{subset=} must be one of {valid_subsets}")

  fetch_openimages(ann_file, base_dir, subset)

# this slows down the conversion a lot!
# maybe use https://raw.githubusercontent.com/scardine/image_size/master/get_image_size.py
def extract_dims(path): return Image.open(path).size[::-1]

def export_to_coco(class_map, annotations, image_list, dataset_path, output_path, subset, classes=MLPERF_CLASSES):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cats = [{"id": i, "name": c, "supercategory": None} for i, c in enumerate(classes)]
  categories_map = pd.DataFrame([(i, c) for i, c in enumerate(classes)], columns=["category_id", "category_name"])
  class_map = class_map.merge(categories_map, left_on="DisplayName", right_on="category_name", how="inner")
  annotations = annotations[annotations["ImageID"].isin(image_list)]
  annotations = annotations.merge(class_map, on="LabelName", how="inner")
  annotations["image_id"] = pd.factorize(annotations["ImageID"].tolist())[0]
  annotations[["height", "width"]] = annotations.apply(lambda x: extract_dims(dataset_path / f"{x['ImageID']}.jpg"), axis=1, result_type="expand")

  # Images
  imgs = [{"id": int(id + 1), "file_name": f"{image_id}.jpg", "height": row["height"], "width": row["width"], "subset": subset, "license": None, "coco_url": None}
    for (id, image_id), row in (annotations.groupby(["image_id", "ImageID"]).first().iterrows())
  ]

  # Annotations
  annots = []
  for i, row in annotations.iterrows():
    xmin, ymin, xmax, ymax, img_w, img_h = [row[k] for k in ["XMin", "YMin", "XMax", "YMax", "width", "height"]]
    x, y, w, h = xmin * img_w, ymin * img_h, (xmax - xmin) * img_w, (ymax - ymin) * img_h
    coco_annot = {"id": int(i) + 1, "image_id": int(row["image_id"] + 1), "category_id": int(row["category_id"]), "bbox": [x, y, w, h], "area": w * h}
    coco_annot.update({k: row[k] for k in ["IsOccluded", "IsInside", "IsDepiction", "IsTruncated", "IsGroupOf"]})
    coco_annot["iscrowd"] = int(row["IsGroupOf"])
    annots.append(coco_annot)

  info = {"dataset": "openimages_mlperf", "version": "v6"}
  coco_annotations = {"info": info, "licenses": [], "categories": cats, "images": imgs, "annotations": annots}
  with open(output_path, "w") as fp:
    json.dump(coco_annotations, fp)

def get_image_list(class_map, annotations, classes=MLPERF_CLASSES):
  labels = class_map[class_map["DisplayName"].isin(classes)]["LabelName"]
  image_ids = annotations[annotations["LabelName"].isin(labels)]["ImageID"].unique()
  return image_ids

def download_image(bucket, subset, image_id, data_dir):
  try:
    bucket.download_file(f"{subset}/{image_id}.jpg", f"{data_dir}/{image_id}.jpg")
  except botocore.exceptions.ClientError as exception:
    sys.exit(f"ERROR when downloading image `validation/{image_id}`: {str(exception)}")

def fetch_openimages(output_fn:str, base_dir:Path, subset:str):
  bucket = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)

  annotations_dir, data_dir = base_dir / "annotations", base_dir / f"{subset}/data"
  annotations_dir.mkdir(parents=True, exist_ok=True)
  data_dir.mkdir(parents=True, exist_ok=True)

  if subset == "train":
    annotations_fn = annotations_dir / TRAIN_BBOX_ANNOTATIONS_URL.split('/')[-1]
    fetch(TRAIN_BBOX_ANNOTATIONS_URL, annotations_fn)
  else:  # subset == validation
    annotations_fn = annotations_dir / VALIDATION_BBOX_ANNOTATIONS_URL.split('/')[-1]
    fetch(VALIDATION_BBOX_ANNOTATIONS_URL, annotations_fn)

  annotations = pd.read_csv(annotations_fn)

  classmap_fn = annotations_dir / MAP_CLASSES_URL.split('/')[-1]
  fetch(MAP_CLASSES_URL, classmap_fn)
  class_map = pd.read_csv(classmap_fn, names=["LabelName", "DisplayName"])

  image_list = get_image_list(class_map, annotations)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_image, bucket, subset, image_id, data_dir) for image_id in image_list]
    for future in (t := tqdm(concurrent.futures.as_completed(futures), total=len(image_list))):
      t.set_description(f"Downloading images")
      future.result()

  print("Converting annotations to COCO format...")
  export_to_coco(class_map, annotations, image_list, data_dir, output_fn, subset)

def image_load(base_dir, subset, fn):
  img_folder = base_dir / f"{subset}/data"
  return Image.open(img_folder / fn).convert('RGB')

def prepare_target(annotations, img_id, img_size):
  boxes = [annot["bbox"] for annot in annotations]
  boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
  boxes[:, 2:] += boxes[:, :2]
  boxes[:, 0::2] = boxes[:, 0::2].clip(0, img_size[1])
  boxes[:, 1::2] = boxes[:, 1::2].clip(0, img_size[0])
  keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
  boxes = boxes[keep]
  classes = [annot["category_id"] for annot in annotations]
  classes = np.array(classes, dtype=np.int64)
  classes = classes[keep]
  return {"boxes": boxes, "labels": classes, "image_id": img_id, "image_size": img_size}

def download_dataset(base_dir:Path, subset:str) -> Path:
  if (ann_file:=base_dir / f"{subset}/labels/openimages-mlperf.json").is_file(): print(f"{subset} dataset is already available")
  else:
    print(f"Downloading {subset} dataset...")
    openimages(base_dir, subset, ann_file)
    print("Done")

  return ann_file

def random_horizontal_flip(img, tgt, prob=0.5):
  import torch
  import torchvision.transforms.functional as F
  if torch.rand(1) < prob:
    w = img.size[0]
    img = F.hflip(img)
    tgt["boxes"][:, [0, 2]] = w - tgt["boxes"][:, [2, 0]]
  return img, tgt

def resize(img:Image, tgt:dict[str, np.ndarray|tuple]|None=None, size:tuple[int, int]=(800, 800)) -> tuple[np.ndarray, np.ndarray, tuple]|tuple[np.ndarray, tuple]:
  import torchvision.transforms.functional as F
  img_size = img.size[::-1]
  img = F.resize(img, size=size)
  img = np.array(img)

  if tgt is not None:
    ratios = [s / s_orig for s, s_orig in zip(size, img_size)]
    ratio_h, ratio_w = ratios
    x_min, y_min, x_max, y_max = [tgt["boxes"][:, i] for i in range(tgt["boxes"].shape[-1])]
    x_min = x_min * ratio_w
    x_max = x_max * ratio_w
    y_min = y_min * ratio_h
    y_max = y_max * ratio_h

    tgt["boxes"] = np.stack([x_min, y_min, x_max, y_max], axis=1)
    return img, tgt, img_size

  return img, img_size

def normalize(img:Tensor, device:list[str]|None = None):
  mean = Tensor([0.485, 0.456, 0.406], device=device, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  std = Tensor([0.229, 0.224, 0.225], device=device, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  img = ((img.permute([0, 3, 1, 2]) / 255.0) - mean) / std
  return img.cast(dtypes.default_float)

def get_dataset_count(base_dir:Path, val:bool) -> int:
  if not (files:=glob.glob(p:=str(base_dir / f"{'validation' if val else 'train'}/data/*.jpg"))): raise FileNotFoundError(f"No files in {p}")
  return len(files)

if __name__ == "__main__":
  download_dataset(base_dir:=getenv("BASEDIR", BASEDIR), "train")
  download_dataset(base_dir, "validation")
