import os
from openpilot.system.hardware import TICI

#
if TICI:
  from tinygrad.tensor import Tensor
  from tinygrad.dtype import dtypes
  from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
  os.environ['QCOM'] = '1'
else:
  from openpilot.selfdrive.modeld.runners.ort_helpers import make_onnx_cpu_runner
import pickle
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLMem

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
MODEL_PATH = Path(__file__).parent / '../models/supercombo.onnx'
MODEL_PKL_PATH = Path(__file__).parent / '../models/supercombo_tinygrad.pkl'
METADATA_PATH = Path(__file__).parent / '../models/supercombo_metadata.pkl'


class ModelRunner(ABC):
  """Abstract base class for model runners that defines the interface for running ML models."""

  def __init__(self):
    """Initialize the model runner with paths to model and metadata files."""
    with open(METADATA_PATH, 'rb') as f:
      self.model_metadata = pickle.load(f)
    self.input_shapes = self.model_metadata['input_shapes']
    self.output_slices = self.model_metadata['output_slices']
    self.inputs: dict = {}

  @abstractmethod
  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray])-> dict:
    """Prepare inputs for model inference."""

  @abstractmethod
  def run_model(self):
    """Run model inference with prepared inputs."""

  def slice_outputs(self, model_outputs: np.ndarray) -> dict:
    """Slice model outputs according to metadata configuration."""
    parsed_outputs = {k: model_outputs[np.newaxis, v] for k, v in self.output_slices.items()}
    if SEND_RAW_PRED:
      parsed_outputs['raw_pred'] = model_outputs.copy()
    return parsed_outputs


class TinygradRunner(ModelRunner):
  """Tinygrad implementation of model runner for TICI hardware."""

  def __init__(self):
    super().__init__()
    # Load Tinygrad model
    with open(MODEL_PKL_PATH, "rb") as f:
      self.model_run = pickle.load(f)

  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray]) -> dict:
    # Initialize image tensors if not already done
    for key in imgs_cl:
      if key not in self.inputs:
        self.inputs[key] = qcom_tensor_from_opencl_address(imgs_cl[key].mem_address, self.input_shapes[key], dtype=dtypes.uint8)

    # Update numpy inputs
    for k, v in numpy_inputs.items():
      if k not in self.inputs:
        self.inputs[k] = Tensor(v, device='NPY').realize()

    return self.inputs

  def run_model(self):
    return self.model_run(**self.inputs).numpy().flatten()


class ONNXRunner(ModelRunner):
  """ONNX implementation of model runner for non-TICI hardware."""

  def __init__(self, frames: dict[str, DrivingModelFrame]):
    super().__init__()
    self.runner = make_onnx_cpu_runner(MODEL_PATH)
    self.frames = frames

  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray]) -> dict:
    self.inputs = numpy_inputs.copy()
    for key in imgs_cl:
      self.inputs[key] = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.input_shapes[key])
    return self.inputs

  def run_model(self):
    return self.runner.run(None, self.inputs)[0].flatten()
