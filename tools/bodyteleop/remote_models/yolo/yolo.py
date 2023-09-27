#!/usr/bin/env python3
import cv2
import numpy as np
import ast
import os
import argparse
import json

from openpilot.selfdrive.modeld.runners.onnxmodel import create_ort_session
from cereal.visionipc import VisionIpcClient, VisionStreamType
from cereal import messaging

N_CLASSES = 80

def xywh2xyxy(x):
  y = x.copy()
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y


def non_max_suppression(boxes, scores, threshold):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == scores.shape[0]
  ys1 = boxes[:, 0]
  xs1 = boxes[:, 1]
  ys2 = boxes[:, 2]
  xs2 = boxes[:, 3]
  areas = (ys2 - ys1) * (xs2 - xs1)
  scores_indexes = scores.argsort().tolist()
  boxes_keep_index = []
  while len(scores_indexes):
    index = scores_indexes.pop()
    boxes_keep_index.append(index)
    if not len(scores_indexes):
      break
    ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
    filtered_indexes = set((ious > threshold).nonzero()[0])
    scores_indexes = [
      v for (i, v) in enumerate(scores_indexes)
      if i not in filtered_indexes
    ]
  return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == boxes_area.shape[0]
  ys1 = np.maximum(box[0], boxes[:, 0])
  xs1 = np.maximum(box[1], boxes[:, 1])
  ys2 = np.minimum(box[2], boxes[:, 2])
  xs2 = np.minimum(box[3], boxes[:, 3])
  intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
  unions = box_area + boxes_area - intersections
  ious = intersections / unions
  return ious


def nms(prediction, conf_thres=0.3, iou_thres=0.45):
  prediction = prediction[prediction[..., 4] > conf_thres]
  boxes = xywh2xyxy(prediction[:, :4])
  res = non_max_suppression(boxes, prediction[:, 4], iou_thres)
  result_boxes = []
  result_scores = []
  result_classes = []
  for r in res:
    result_boxes.append(boxes[r])
    result_scores.append(prediction[r, 4])
    result_classes.append(np.argmax(prediction[r, 5:]))
  return np.c_[result_boxes, result_scores, result_classes]


class YoloRunner:
  def __init__(self, onnx_path):
    self.sess = create_ort_session(onnx_path)
    self.class_names = [ast.literal_eval(self.sess.get_modelmeta().custom_metadata_map['names'])[i] for i in range(N_CLASSES)]
    self.width, self.height = 640, 416

  def preprocess_image(self, img):
    img = cv2.resize(img, (self.width, self.height))
    img = np.expand_dims(img, 0).astype(np.float32) # add batch dim
    img = img.transpose(0, 3, 1, 2) # NHWC -> NCHW
    img = img[:, [2,1,0], :, :] # BGR -> RGB
    img /= 255 # 0-255 -> 0-1
    return img

  def run(self, img):
    img = self.preprocess_image(img)
    res = self.sess.run(None, {'images': img})
    res = nms(res[0])
    return [{
        "pred_class": self.class_names[int(opt[-1])],
        "prob": opt[-2],
        "pt1": opt[:2].astype(int).tolist(),
        "pt2": opt[2:4].astype(int).tolist()}
      for opt in res]

  def draw_boxes(self, img, objects):
    img = cv2.resize(img, (self.width, self.height))
    for obj in objects:
      pt1 = obj['pt1']
      pt2 = tuple(obj['pt2'])
      cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
      cv2.putText(img, f"{obj['pred_class']} {obj['prob']:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img


def main(yolo_runner, debug):
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)
  pm = messaging.PubMaster(['customReservedRawData1'])

  while True:
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue
    imgff = np.frombuffer(yuv_img_raw.data, dtype=np.uint8).reshape((vipc_client.height * 3 // 2, vipc_client.width))
    img = cv2.cvtColor(imgff, cv2.COLOR_YUV2BGR_I420)
    outputs = yolo_runner.run(img)
    msg = messaging.new_message()
    msg.customReservedRawData1 = json.dumps(outputs).encode()
    pm.send('customReservedRawData1', msg)
    if debug:
      cv2.imwrite('yolo.jpg', yolo_runner.draw_boxes(img, outputs))
      print(f"found: {','.join([x['pred_class'] for x in outputs])}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Receive VisionIPC frames, run YOLO and publish outputs")
  parser.add_argument("--debug", action="store_true", help="debug output")
  args = parser.parse_args()

  if not os.path.isfile('yolov5n.onnx'):
    os.system('wget https://github.com/YassineYousfi/yolov5n.onnx/releases/download/yolov5n.onnx/yolov5n.onnx')
  yolo_runner = YoloRunner('yolov5n.onnx')
  main(yolo_runner, debug=args.debug)
