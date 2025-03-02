import os
import pickle
from abc import ABC, abstractmethod
import numpy as np

from cereal import custom
from openpilot.sunnypilot.modeld_v2 import MODEL_PATH, MODEL_PKL_PATH, METADATA_PATH
from openpilot.sunnypilot.modeld_v2.models.commonmodel_pyx import DrivingModelFrame, CLMem
from openpilot.sunnypilot.modeld_v2.runners.ort_helpers import make_onnx_cpu_runner, ORT_TYPES_TO_NP_TYPES
from openpilot.sunnypilot.modeld_v2.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
from openpilot.system.hardware import TICI
from openpilot.system.hardware.hw import Paths

from openpilot.sunnypilot.models.helpers import get_active_bundle
from tinygrad.tensor import Tensor

if TICI:
  os.environ['QCOM'] = '1'

SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
CUSTOM_MODEL_PATH = Paths.model_root()
ModelManager = custom.ModelManagerSP


class ModelRunner(ABC):
  """Abstract base class for model runners that defines the interface for running ML models."""

  def __init__(self):
    """Initialize the model runner with paths to model and metadata files."""
    metadata_path = METADATA_PATH
    self.is_20hz = None
    self._drive_model = None
    self._metadata_model = None

    if bundle := get_active_bundle():
      bundle_models = {model.type.raw: model for model in bundle.models}
      self._drive_model = bundle_models.get(ModelManager.Type.drive)
      self._metadata_model = bundle_models.get(ModelManager.Type.metadata)
      self.is_20hz = bundle.is20hz

    # Override the metadata path if a metadata model is found in the active bundle
    if self._metadata_model:
      metadata_path = f"{CUSTOM_MODEL_PATH}/{self._metadata_model.fileName}"

    with open(metadata_path, 'rb') as f:
      self.model_metadata = pickle.load(f)

    self.input_shapes = self.model_metadata['input_shapes']
    self.output_slices = self.model_metadata['output_slices']
    self.inputs: dict = {}

  @abstractmethod
  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray], frames: dict[str, DrivingModelFrame]) -> dict:
    """Prepare inputs for model inference."""
    raise NotImplementedError

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

    model_pkl_path = MODEL_PKL_PATH
    if self._drive_model:
      model_pkl_path = f"{CUSTOM_MODEL_PATH}/{self._drive_model.fileName}"
      assert model_pkl_path.endswith('_tinygrad.pkl'), f"Invalid model file: {model_pkl_path} for TinygradRunner"

    # Load Tinygrad model
    with open(model_pkl_path, "rb") as f:
      try:
        self.model_run = pickle.load(f)
      except FileNotFoundError as e:
        assert "/dev/kgsl-3d0" not in str(e), "Model was built on C3 or C3X, but is being loaded on PC"
        raise

    self.input_to_dtype = {}
    self.input_to_device = {}

    for idx, name in enumerate(self.model_run.captured.expected_names):
      self.input_to_dtype[name] = self.model_run.captured.expected_st_vars_dtype_device[idx][2]  # 2 is the dtype
      self.input_to_device[name] = self.model_run.captured.expected_st_vars_dtype_device[idx][3]  # 3 is the device

  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray], frames: dict[str, DrivingModelFrame]) -> dict:
    # Initialize image tensors if not already done
    for key in imgs_cl:
      if TICI and key not in self.inputs:
        self.inputs[key] = qcom_tensor_from_opencl_address(imgs_cl[key].mem_address, self.input_shapes[key], dtype=self.input_to_dtype[key])
      elif not TICI:
        shape = frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.input_shapes[key])
        self.inputs[key] = Tensor(shape, device=self.input_to_device[key], dtype=self.input_to_dtype[key]).realize()

    # Update numpy inputs
    for key, value in numpy_inputs.items():
      if key not in imgs_cl:
        self.inputs[key] = Tensor(value, device=self.input_to_device[key], dtype=self.input_to_dtype[key]).realize()

    return self.inputs

  def run_model(self):
    return self.model_run(**self.inputs).numpy().flatten()


class ONNXRunner(ModelRunner):
  """ONNX implementation of model runner for non-TICI hardware."""

  def __init__(self):
    super().__init__()
    self.runner = make_onnx_cpu_runner(MODEL_PATH)

    self.input_to_nptype = {
      model_input.name: ORT_TYPES_TO_NP_TYPES[model_input.type]
      for model_input in self.runner.get_inputs()
    }

  def prepare_inputs(self, imgs_cl: dict[str, CLMem], numpy_inputs: dict[str, np.ndarray], frames: dict[str, DrivingModelFrame]) -> dict:
    self.inputs = numpy_inputs
    for key in imgs_cl:
      self.inputs[key] = frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.input_shapes[key]).astype(dtype=self.input_to_nptype[key])
    return self.inputs

  def run_model(self):
    return self.runner.run(None, self.inputs)[0].flatten()
