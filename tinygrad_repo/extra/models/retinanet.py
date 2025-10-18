import math
from tinygrad import Tensor, dtypes
from tinygrad.helpers import flatten, get_child
from examples.mlperf.helpers import generate_anchors, BoxCoder
from examples.mlperf.losses import sigmoid_focal_loss, l1_loss
from extra.models.resnet import ResNet
import tinygrad.nn as nn
import numpy as np

ConvFPN = ConvHead = ConvClassificationHeadLogits = nn.Conv2d

def nms(boxes, scores, thresh=0.5):
  x1, y1, x2, y2 = np.rollaxis(boxes, 1)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort()[::-1], []
  while to_process.size > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = np.maximum(x1[cur], x1[to_process])
    inter_y1 = np.maximum(y1[cur], y1[to_process])
    inter_x2 = np.minimum(x2[cur], x2[to_process])
    inter_y2 = np.minimum(y2[cur], y2[to_process])
    inter_area = np.maximum(0, inter_x2 - inter_x1 + 1) * np.maximum(0, inter_y2 - inter_y1 + 1)
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[np.where(iou <= thresh)[0]]
  return keep

def decode_bbox(offsets, anchors):
  dx, dy, dw, dh = np.rollaxis(offsets, 1)
  widths, heights = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
  cx, cy = anchors[:, 0] + 0.5 * widths, anchors[:, 1] + 0.5 * heights
  pred_cx, pred_cy = dx * widths + cx, dy * heights + cy
  pred_w, pred_h = np.exp(dw) * widths, np.exp(dh) * heights
  pred_x1, pred_y1 = pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h
  pred_x2, pred_y2 = pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h
  return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1, dtype=np.float32)

class RetinaNet:
  def __init__(self, backbone:ResNet, num_classes:int=264, num_anchors:int=9, scales:list[int]|None=None, aspect_ratios:list[float]|None=None):
    assert isinstance(backbone, ResNet)
    scales = tuple((i, int(i*2**(1/3)), int(i*2**(2/3))) for i in 2**np.arange(5, 10)) if scales is None else scales
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales) if aspect_ratios is None else aspect_ratios
    self.num_anchors, self.num_classes = num_anchors, num_classes
    assert len(scales) == len(aspect_ratios) and all(self.num_anchors == len(s) * len(ar) for s, ar in zip(scales, aspect_ratios))

    self.backbone = ResNetFPN(backbone)
    self.head = RetinaHead(self.backbone.out_channels, num_anchors=num_anchors, num_classes=num_classes)

  def __call__(self, x:Tensor, **kwargs):
    return self.forward(x, **kwargs)

  def forward(self, x:Tensor, **kwargs):
    return self.head(self.backbone(x), **kwargs)

  def load_from_pretrained(self):
    model_urls = {
      (50, 1, 64): "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
      (50, 32, 4): "https://zenodo.org/record/6605272/files/retinanet_model_10.zip",
    }
    self.url = model_urls[(self.backbone.body.num, self.backbone.body.groups, self.backbone.body.base_width)]
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

  # predictions: (BS, (H1W1+...+HmWm)A, 4 + K)
  def postprocess_detections(self, predictions, input_size=(800, 800), image_sizes=None, orig_image_sizes=None, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5):
    anchors = generate_anchors(input_size)
    grid_sizes = self.backbone.compute_grid_sizes(input_size)
    split_idx = np.cumsum([int(self.num_anchors * sz[0] * sz[1]) for sz in grid_sizes[:-1]])
    detections = []
    for i, predictions_per_image in enumerate(predictions):
      h, w = input_size if image_sizes is None else image_sizes[i]

      predictions_per_image = np.split(predictions_per_image, split_idx)
      offsets_per_image = [br[:, :4] for br in predictions_per_image]
      scores_per_image = [cl[:, 4:] for cl in predictions_per_image]

      image_boxes, image_scores, image_labels = [], [], []
      for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
        # remove low scoring boxes
        scores_per_level = scores_per_level.flatten()
        keep_idxs = scores_per_level > score_thresh
        scores_per_level = scores_per_level[keep_idxs]

        # keep topk
        topk_idxs = np.where(keep_idxs)[0]
        num_topk = min(len(topk_idxs), topk_candidates)
        sort_idxs = scores_per_level.argsort()[-num_topk:][::-1]
        topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]

        # bbox coords from offsets
        anchor_idxs = topk_idxs // self.num_classes
        labels_per_level = topk_idxs % self.num_classes
        boxes_per_level = decode_bbox(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
        # clip to image size
        clipped_x = boxes_per_level[:, 0::2].clip(0, w)
        clipped_y = boxes_per_level[:, 1::2].clip(0, h)
        boxes_per_level = np.stack([clipped_x, clipped_y], axis=2).reshape(-1, 4)

        image_boxes.append(boxes_per_level)
        image_scores.append(scores_per_level)
        image_labels.append(labels_per_level)

      image_boxes = np.concatenate(image_boxes)
      image_scores = np.concatenate(image_scores)
      image_labels = np.concatenate(image_labels)

      # nms for each class
      keep_mask = np.zeros_like(image_scores, dtype=bool)
      for class_id in np.unique(image_labels):
        curr_indices = np.where(image_labels == class_id)[0]
        curr_keep_indices = nms(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
        keep_mask[curr_indices[curr_keep_indices]] = True
      keep = np.where(keep_mask)[0]
      keep = keep[image_scores[keep].argsort()[::-1]]

      # resize bboxes back to original size
      image_boxes = image_boxes[keep]
      if orig_image_sizes is not None:
        resized_x = image_boxes[:, 0::2] * orig_image_sizes[i][1] / w
        resized_y = image_boxes[:, 1::2] * orig_image_sizes[i][0] / h
        image_boxes = np.stack([resized_x, resized_y], axis=2).reshape(-1, 4)
      # xywh format
      image_boxes = np.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], axis=1)

      detections.append({"boxes":image_boxes, "scores":image_scores[keep], "labels":image_labels[keep]})
    return detections

class ClassificationHead:
  def __init__(self, in_channels:int, num_anchors:int, num_classes:int):
    self.num_classes = num_classes
    self.conv = flatten([(ConvHead(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.cls_logits = ConvClassificationHeadLogits(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

  def __call__(self, x:Tensor, labels:Tensor|None=None, matches:Tensor|None=None):
    out = [self.cls_logits(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.num_classes) for feat in x]
    out = out[0].cat(*out[1:], dim=1)

    if Tensor.training:
      assert labels is not None and matches is not None, "labels and matches should be passed in when training"
      return self._compute_loss(out.cast(dtypes.float32), labels, matches)

    return out.sigmoid()

  def _compute_loss(self, x:Tensor, labels:Tensor, matches:Tensor) -> Tensor:
    labels = ((labels + 1) * (fg_idxs := matches >= 0) - 1).one_hot(num_classes=x.shape[-1])
    valid_idxs = (matches != -2).reshape(matches.shape[0], -1, 1)
    loss = valid_idxs.where(sigmoid_focal_loss(x, labels), 0).sum((-1, -2))
    loss = (loss / fg_idxs.sum(-1)).sum() / matches.shape[0]
    return loss

class RegressionHead:
  def __init__(self, in_channels:int, num_anchors:int, box_coder:BoxCoder|None=None):
    self.conv = flatten([(ConvHead(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.bbox_reg = ConvHead(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    if box_coder is None:
      box_coder = BoxCoder((1.0, 1.0, 1.0, 1.0))
    self.box_coder = box_coder

  def __call__(self, x:Tensor, bboxes:Tensor|None=None, matches:Tensor|None=None, anchors:Tensor|None=None):
    out = [self.bbox_reg(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, 4) for feat in x]
    out = out[0].cat(*out[1:], dim=1)

    if Tensor.training:
      assert bboxes is not None and matches is not None and anchors is not None, "bboxes, matches, and anchors should be passed in when training"
      return self._compute_loss(out, bboxes, matches, anchors)

    return out

  def _compute_loss(self, x:Tensor, bboxes:Tensor, matches:Tensor, anchors:Tensor) -> Tensor:
    mask = (fg_idxs := matches >= 0).reshape(matches.shape[0], -1, 1)
    x = x * mask
    tgt = self.box_coder.encode(bboxes, anchors) * mask
    loss = l1_loss(x, tgt).sum((-1, -2))
    loss = (loss / fg_idxs.sum(-1)).sum() / matches.shape[0]
    return loss

class RetinaHead:
  def __init__(self, in_channels:int, num_anchors:int, num_classes:int):
    self.classification_head = ClassificationHead(in_channels, num_anchors, num_classes)
    self.regression_head = RegressionHead(in_channels, num_anchors)

  def __call__(self, x:Tensor, **kwargs) -> Tensor|dict[str, Tensor]:
    if Tensor.training:
      return {
        "classification_loss": self.classification_head(x, labels=kwargs["labels"], matches=kwargs["matches"]),
        "regression_loss": self.regression_head(x, bboxes=kwargs["bboxes"], matches=kwargs["matches"], anchors=kwargs["anchors"])
      }

    pred_bbox, pred_class = self.regression_head(x), self.classification_head(x)
    out = pred_bbox.cat(pred_class, dim=-1)
    return out

class ResNetFPN:
  def __init__(self, resnet:ResNet, out_channels:int=256, returned_layers:list[int]=[2, 3, 4]):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_list = [(self.body.in_planes // 8) * 2 ** (i - 1) for i in returned_layers]
    self.fpn = FPN(in_channels_list, out_channels)

  # this is needed to decouple inference from postprocessing (anchors generation)
  def compute_grid_sizes(self, input_size):
    return np.ceil(np.array(input_size)[None, :] / 2 ** np.arange(3, 8)[:, None])

  def __call__(self, x:Tensor):
    out = self.body.bn1(self.body.conv1(x)).relu()
    out = out.pad([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.body.layer1)
    p3 = out.sequential(self.body.layer2)
    p4 = p3.sequential(self.body.layer3)
    p5 = p4.sequential(self.body.layer4)
    return self.fpn([p3, p4, p5])

class ExtraFPNBlock:
  def __init__(self, in_channels:int, out_channels:int):
    self.p6 = ConvFPN(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.p7 = ConvFPN(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.use_P5 = in_channels == out_channels

  def __call__(self, p:Tensor, c:Tensor):
    p5, c5 = p[-1], c[-1]
    x = p5 if self.use_P5 else c5
    p6 = self.p6(x)
    p7 = self.p7(p6.relu())
    p.extend([p6, p7])
    return p

class FPN:
  def __init__(self, in_channels_list:list[int], out_channels:int, extra_blocks:ExtraFPNBlock|None=None):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(ConvFPN(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(ConvFPN(out_channels, out_channels, kernel_size=3, padding=1))
    self.extra_blocks = ExtraFPNBlock(256, 256) if extra_blocks is None else extra_blocks

  def __call__(self, x:Tensor):
    last_inner = self.inner_blocks[-1](x[-1])
    results = [self.layer_blocks[-1](last_inner)]
    for idx in range(len(x) - 2, -1, -1):
      inner_lateral = self.inner_blocks[idx](x[idx])

      # upsample to inner_lateral's shape
      (ih, iw), (oh, ow), prefix = last_inner.shape[-2:], inner_lateral.shape[-2:], last_inner.shape[:-2]
      eh, ew = math.ceil(oh / ih), math.ceil(ow / iw)
      inner_top_down = last_inner.reshape(*prefix, ih, 1, iw, 1).expand(*prefix, ih, eh, iw, ew).reshape(*prefix, ih*eh, iw*ew)[:, :, :oh, :ow]

      last_inner = inner_lateral + inner_top_down
      results.insert(0, self.layer_blocks[idx](last_inner))
    if self.extra_blocks is not None:
      results = self.extra_blocks(results, x)
    return results

if __name__ == "__main__":
  from extra.models.resnet import ResNeXt50_32X4D
  backbone = ResNeXt50_32X4D()
  retina = RetinaNet(backbone)
  retina.load_from_pretrained()
