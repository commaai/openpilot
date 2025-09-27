from extra.models.mask_rcnn import MaskRCNN
from extra.models.resnet import ResNet
from extra.models.mask_rcnn import BoxList
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as Ft
import random
from tinygrad.tensor import Tensor
from PIL import Image
import numpy as np
import torch
import argparse
import cv2


class Resize:
  def __init__(self, min_size, max_size):
    if not isinstance(min_size, (list, tuple)):
      min_size = (min_size,)
    self.min_size = min_size
    self.max_size = max_size

  # modified from torchvision to add support for max size
  def get_size(self, image_size):
    w, h = image_size
    size = random.choice(self.min_size)
    max_size = self.max_size
    if max_size is not None:
      min_original_size = float(min((w, h)))
      max_original_size = float(max((w, h)))
      if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

      if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

      if w < h:
        ow = size
        oh = int(size * h / w)
      else:
        oh = size
        ow = int(size * w / h)

      return (oh, ow)

  def __call__(self, image):
    size = self.get_size(image.size)
    image = Ft.resize(image, size)
    return image


class Normalize:
  def __init__(self, mean, std, to_bgr255=True):
    self.mean = mean
    self.std = std
    self.to_bgr255 = to_bgr255

  def __call__(self, image):
    if self.to_bgr255:
      image = image[[2, 1, 0]] * 255
    else:
      image = image[[0, 1, 2]] * 255
    image = Ft.normalize(image, mean=self.mean, std=self.std)
    return image

transforms = lambda size_scale: T.Compose(
  [
    Resize(int(800*size_scale), int(1333*size_scale)),
    T.ToTensor(),
    Normalize(
      mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.], to_bgr255=True
    ),
  ]
)

def expand_boxes(boxes, scale):
  w_half = (boxes[:, 2] - boxes[:, 0]) * .5
  h_half = (boxes[:, 3] - boxes[:, 1]) * .5
  x_c = (boxes[:, 2] + boxes[:, 0]) * .5
  y_c = (boxes[:, 3] + boxes[:, 1]) * .5

  w_half *= scale
  h_half *= scale

  boxes_exp = torch.zeros_like(boxes)
  boxes_exp[:, 0] = x_c - w_half
  boxes_exp[:, 2] = x_c + w_half
  boxes_exp[:, 1] = y_c - h_half
  boxes_exp[:, 3] = y_c + h_half
  return boxes_exp


def expand_masks(mask, padding):
  N = mask.shape[0]
  M = mask.shape[-1]
  pad2 = 2 * padding
  scale = float(M + pad2) / M
  padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
  padded_mask[:, :, padding:-padding, padding:-padding] = mask
  return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
  # TODO: remove torch
  mask = torch.tensor(mask.numpy())
  box = torch.tensor(box.numpy())
  padded_mask, scale = expand_masks(mask[None], padding=padding)
  mask = padded_mask[0, 0]
  box = expand_boxes(box[None], scale)[0]
  box = box.to(dtype=torch.int32)

  TO_REMOVE = 1
  w = int(box[2] - box[0] + TO_REMOVE)
  h = int(box[3] - box[1] + TO_REMOVE)
  w = max(w, 1)
  h = max(h, 1)

  mask = mask.expand((1, 1, -1, -1))

  mask = mask.to(torch.float32)
  mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
  mask = mask[0][0]

  if thresh >= 0:
    mask = mask > thresh
  else:
    mask = (mask * 255).to(torch.uint8)

  im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
  x_0 = max(box[0], 0)
  x_1 = min(box[2] + 1, im_w)
  y_0 = max(box[1], 0)
  y_1 = min(box[3] + 1, im_h)

  im_mask[y_0:y_1, x_0:x_1] = mask[
                              (y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])
                              ]
  return im_mask


class Masker:
  def __init__(self, threshold=0.5, padding=1):
    self.threshold = threshold
    self.padding = padding

  def forward_single_image(self, masks, boxes):
    boxes = boxes.convert("xyxy")
    im_w, im_h = boxes.size
    res = [
      paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
      for mask, box in zip(masks, boxes.bbox)
    ]
    if len(res) > 0:
      res = torch.stack(*res, dim=0)[:, None]
    else:
      res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
    return Tensor(res.numpy())

  def __call__(self, masks, boxes):
    if isinstance(boxes, BoxList):
      boxes = [boxes]

    results = []
    for mask, box in zip(masks, boxes):
      result = self.forward_single_image(mask, box)
      results.append(result)
    return results


masker = Masker(threshold=0.5, padding=1)

def select_top_predictions(predictions, confidence_threshold=0.9):
  scores = predictions.get_field("scores").numpy()
  keep = [idx for idx, score in enumerate(scores) if score > confidence_threshold]
  return predictions[keep]

def compute_prediction(original_image, model, confidence_threshold, size_scale=1.0):
  image = transforms(size_scale)(original_image).numpy()
  image = Tensor(image, requires_grad=False)
  predictions = model(image)
  prediction = predictions[0]
  prediction = select_top_predictions(prediction, confidence_threshold)
  width, height = original_image.size
  prediction = prediction.resize((width, height))

  if prediction.has_field("mask"):
    masks = prediction.get_field("mask")
    masks = masker([masks], [prediction])[0]
    prediction.add_field("mask", masks)
  return prediction

def compute_prediction_batched(batch, model, size_scale=1.0):
  imgs = []
  for img in batch:
    imgs.append(transforms(size_scale)(img).numpy())
  image = [Tensor(image, requires_grad=False) for image in imgs]
  predictions = model(image)
  del image
  return predictions

palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

def findContours(*args, **kwargs):
  if cv2.__version__.startswith('4'):
    contours, hierarchy = cv2.findContours(*args, **kwargs)
  elif cv2.__version__.startswith('3'):
    _, contours, hierarchy = cv2.findContours(*args, **kwargs)
  return contours, hierarchy

def compute_colors_for_labels(labels):
  l = labels[:, None]
  colors = l * palette
  colors = (colors % 255).astype("uint8")
  return colors

def overlay_mask(image, predictions):
  image = np.asarray(image)
  masks = predictions.get_field("mask").numpy()
  labels = predictions.get_field("labels").numpy()

  colors = compute_colors_for_labels(labels).tolist()

  for mask, color in zip(masks, colors):
    thresh = mask[0, :, :, None]
    contours, hierarchy = findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    image = cv2.drawContours(image, contours, -1, color, 3)

  composite = image

  return composite

CATEGORIES = [
    "__background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

def overlay_boxes(image, predictions):
  labels = predictions.get_field("labels").numpy()
  boxes = predictions.bbox
  image = np.asarray(image)
  colors = compute_colors_for_labels(labels).tolist()

  for box, color in zip(boxes, colors):
    box = torch.tensor(box.numpy())
    box = box.to(torch.int64)
    top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    image = cv2.rectangle(
        image, tuple(top_left), tuple(bottom_right), tuple(color), 1
    )

  return image

def overlay_class_names(image, predictions):
  scores = predictions.get_field("scores").numpy().tolist()
  labels = predictions.get_field("labels").numpy().tolist()
  labels = [CATEGORIES[int(i)] for i in labels]
  boxes = predictions.bbox.numpy()
  image = np.asarray(image)
  template = "{}: {:.2f}"
  for box, score, label in zip(boxes, scores, labels):
    x, y = box[:2]
    s = template.format(label, score)
    x, y = int(x), int(y)
    cv2.putText(
        image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
    )

  return image


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run MaskRCNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--image', type=str, help="Path of the image to run")
  parser.add_argument('--threshold', type=float, default=0.7, help="Detector threshold")
  parser.add_argument('--size_scale', type=float, default=1.0, help="Image resize multiplier")
  parser.add_argument('--out', type=str, default="/tmp/rendered.png", help="Output filename")
  args = parser.parse_args()

  resnet = ResNet(50, num_classes=None, stride_in_1x1=True)
  model_tiny = MaskRCNN(resnet)
  model_tiny.load_from_pretrained()
  img = Image.open(args.image)
  top_result_tiny = compute_prediction(img, model_tiny, confidence_threshold=args.threshold, size_scale=args.size_scale)
  bbox_image = overlay_boxes(img, top_result_tiny)
  mask_image = overlay_mask(bbox_image, top_result_tiny)
  final_image = overlay_class_names(mask_image, top_result_tiny)

  im = Image.fromarray(final_image)
  print(f"saving {args.out}")
  im.save(args.out)
  im.show()
