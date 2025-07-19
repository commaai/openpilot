#!/usr/bin/env python3
import os
from ultralytics import YOLO
from pathlib import Path
from tinygrad.frontend.onnx import OnnxRunner
from extra.onnx_helpers import get_example_inputs

os.chdir("/tmp")
if not Path("yolov8n-seg.onnx").is_file():
  model = YOLO("yolov8n-seg.pt")
  model.export(format="onnx", imgsz=[480,640])
run_onnx = OnnxRunner("yolov8n-seg.onnx")
run_onnx(get_example_inputs(run_onnx.graph_inputs), debug=True)
