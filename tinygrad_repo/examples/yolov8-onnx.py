#!/usr/bin/env python3
import os
from ultralytics import YOLO
import onnx
from pathlib import Path
from extra.onnx import get_run_onnx
from tinygrad.tensor import Tensor

os.chdir("/tmp")
if not Path("yolov8n-seg.onnx").is_file():
  model = YOLO("yolov8n-seg.pt")
  model.export(format="onnx", imgsz=[480,640])
onnx_model = onnx.load(open("yolov8n-seg.onnx", "rb"))
# TODO: move get example inputs to onnx
input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
print(input_shapes)
run_onnx = get_run_onnx(onnx_model)
run_onnx({"images": Tensor.zeros(1,3,480,640)}, debug=True)
