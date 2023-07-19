import numpy as np
from dataclasses import dataclass
from selfdrive.modeld.runners.runmodel_pyx import ONNXModel

FEATURE_LEN = 128
HISTORY_BUFFER_LEN = 99
DESIRE_LEN = 8
TRAFFIC_CONVENTION_LEN = 2
DRIVING_STYLE_LEN = 12
MODEL_OUTPUT_SIZE = 1000

MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_FRAME_SIZE = MODEL_WIDTH * MODEL_HEIGHT * 3 // 2
BUF_SIZE = MODEL_FRAME_SIZE * 2

@dataclass
class ModelState:
  model: ONNXModel
  inputs: Dict[str, np.ndarray]
  output: np.ndarray
  # frame: ModelFrame
  # wide_frame: ModelFrame


def model_init(device_id, context):
  frame = ModelFrame(device_id, context)
  wide_frame = ModelFrame(device_id, context)

  inputs = {}
  output = np.zeros(MODEL_OUTPUT_SIZE, dtype=np.float32)
  model = ONNXModel("models/supercombo.onnx", output, MODEL_OUTPUT_SIZE, Runtimes.GPU_RUNTIME, False, context)

  model.addInput("input_imgs", None)
  model.addInput("big_input_imgs", None)
  for k,v in inputs.items():
    model.addInput(k, v)

  return ModelState(model, inputs, output, frame, wide_frame)


def model_eval_frame(s:ModelState, buf:np.ndarray, wbuf:np.ndarray, transform:np.ndarray, transform_wide:np.ndarray,
                     desire_in:np.ndarray, is_rhd:bool, driving_style:np.ndarray, nav_features:np.ndarray, prepare_only:bool):

  return None
  """
  std::memmove(&s->pulse_desire[0], &s->pulse_desire[DESIRE_LEN], sizeof(float) * DESIRE_LEN*HISTORY_BUFFER_LEN);
  if (desire_in != NULL) {
    for (int i = 1; i < DESIRE_LEN; i++) {
      // Model decides when action is completed
      // so desire input is just a pulse triggered on rising edge
      if (desire_in[i] - s->prev_desire[i] > .99) {
        s->pulse_desire[DESIRE_LEN*HISTORY_BUFFER_LEN+i] = desire_in[i];
      } else {
        s->pulse_desire[DESIRE_LEN*HISTORY_BUFFER_LEN+i] = 0.0;
      }
      s->prev_desire[i] = desire_in[i];
    }
  }
  LOGT("Desire enqueued");

  std::memcpy(s->nav_features, nav_features, sizeof(float)*NAV_FEATURE_LEN);
  std::memcpy(s->driving_style, driving_style, sizeof(float)*DRIVING_STYLE_LEN);

  int rhd_idx = is_rhd;
  s->traffic_convention[rhd_idx] = 1.0;
  s->traffic_convention[1-rhd_idx] = 0.0;

  // if getInputBuf is not NULL, net_input_buf will be
  auto net_input_buf = s->frame->prepare(buf->buf_cl, buf->width, buf->height, buf->stride, buf->uv_offset, transform, static_cast<cl_mem*>(s->m->getCLBuffer("input_imgs")));
  s->m->setInputBuffer("input_imgs", net_input_buf, s->frame->buf_size);
  LOGT("Image added");

  if (wbuf != nullptr) {
    auto net_extra_buf = s->wide_frame->prepare(wbuf->buf_cl, wbuf->width, wbuf->height, wbuf->stride, wbuf->uv_offset, transform_wide, static_cast<cl_mem*>(s->m->getCLBuffer("big_input_imgs")));
    s->m->setInputBuffer("big_input_imgs", net_extra_buf, s->wide_frame->buf_size);
    LOGT("Extra image added");
  }

  if (prepare_only) {
    return nullptr;
  }

  s->m->execute();
  LOGT("Execution finished");

  #ifdef TEMPORAL
    std::memmove(&s->feature_buffer[0], &s->feature_buffer[FEATURE_LEN], sizeof(float) * FEATURE_LEN*(HISTORY_BUFFER_LEN-1));
    std::memcpy(&s->feature_buffer[FEATURE_LEN*(HISTORY_BUFFER_LEN-1)], &s->output[OUTPUT_SIZE], sizeof(float) * FEATURE_LEN);
    LOGT("Features enqueued");
  #endif
  """
