import logging
import numpy as np
from typing import Dict
from cereal.visionipc.visionipc_pyx import VisionBuf # pylint: disable=no-name-in-module
from selfdrive.modeld.runners.runmodel_pyx import ONNXModel, Runtime # pylint: disable=no-name-in-module
from selfdrive.modeld.models.commonmodel_pyx import ModelFrame # pylint: disable=no-name-in-module
from selfdrive.modeld.models.cl_pyx import CLContext # pylint: disable=no-name-in-module

FEATURE_LEN = 128
HISTORY_BUFFER_LEN = 99
DESIRE_LEN = 8
TRAFFIC_CONVENTION_LEN = 2
DRIVING_STYLE_LEN = 12
NAV_FEATURE_LEN = 256
OUTPUT_SIZE = 5990
MODEL_OUTPUT_SIZE = 6120
MODEL_FREQ = 20

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
BUF_SIZE = MODEL_FRAME_SIZE * 2

class ModelState:
  frame: ModelFrame
  wide_frame: ModelFrame
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse
  model: ONNXModel

  def __init__(self, context:CLContext):
    self.frame = ModelFrame(context)
    self.wide_frame = ModelFrame(context)
    self.prev_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
    self.output = np.zeros(MODEL_OUTPUT_SIZE, dtype=np.float32)
    self.inputs = {
      'desire_pulse': np.zeros(DESIRE_LEN * (HISTORY_BUFFER_LEN+1), dtype=np.float32),
      'traffic_convention': np.zeros(TRAFFIC_CONVENTION_LEN, dtype=np.float32),
      'nav_features': np.zeros(NAV_FEATURE_LEN, dtype=np.float32),
      'feature_buffer': np.zeros(HISTORY_BUFFER_LEN * FEATURE_LEN, dtype=np.float32),
    }

    self.model = ONNXModel("models/supercombo.onnx", self.output, Runtime.GPU, False, context)
    self.model.addInput("input_imgs", None)
    self.model.addInput("big_input_imgs", None)
    for k,v in self.inputs.items():
      self.model.addInput(k, v)

  def eval(self, buf:VisionBuf, wbuf:VisionBuf, transform:np.ndarray, transform_wide:np.ndarray, inputs:Dict[str, np.ndarray], prepare_only:bool):
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.inputs['desire_pulse'][:-DESIRE_LEN] = self.inputs['desire_pulse'][DESIRE_LEN:]
    self.inputs['desire_pulse'][-DESIRE_LEN:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    logging.info("Desire enqueued")

    self.inputs['traffic_convention'][:] = inputs['traffic_convention']
    self.inputs['nav_features'][:] = inputs['nav_features']
    # self.inputs['driving_style'][:] = inputs['driving_style']

    # if getCLBuffer is not None, frame will be None
    frame = self.frame.prepare(buf, transform.astype(np.float32).flatten(), self.model.getCLBuffer("input_imgs"))
    self.model.setInputBuffer("input_imgs", frame)
    logging.info("Image added")

    if wbuf is not None:
      wide_frame = self.wide_frame.prepare(wbuf, transform_wide.astype(np.float32).flatten(), self.model.getCLBuffer("big_input_imgs"))
      self.model.setInputBuffer("big_input_imgs", wide_frame)
      logging.info("Extra image added")

    if prepare_only:
      return None

    self.model.execute()
    logging.info("Execution finished")

    self.inputs['feature_buffer'][:-FEATURE_LEN] = self.inputs['feature_buffer'][FEATURE_LEN:]
    self.inputs['feature_buffer'][-FEATURE_LEN:] = self.output[OUTPUT_SIZE:OUTPUT_SIZE+FEATURE_LEN]
    logging.info("Features enqueued")

    return self.output
