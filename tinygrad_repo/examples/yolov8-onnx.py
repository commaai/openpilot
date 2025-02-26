#!/usr/bin/env python3
import os
from ultralytics import YOLO
import onnx
from pathlib import Path
from extra.onnx import OnnxRunner
from extra.onnx_helpers import get_example_inputs
from tinygrad.tensor import Tensor

os.chdir("/tmp")
if not Path("yolov8n-seg.onnx").is_file():
  model = YOLO("yolov8n-seg.pt")
  model.export(format="onnx", imgsz=[480,640])
onnx_model = onnx.load(open("yolov8n-seg.onnx", "rb"))
run_onnx = OnnxRunner(onnx_model)
run_onnx(get_example_inputs(run_onnx.graph_inputs), debug=True)
