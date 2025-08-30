# Test whether pretrained weights from first BERT pretraining phase have been loaded correctly
# Usage:
# 1. Download the BERT checkoints with `wikipedia_download.py`
#    Command: BASEDIR=/path/to/wiki python3 wikipedia_download.py
# 2. Run this script. (Adjust EVAL_BS and GPUS as needed)
#    Command: EVAL_BEAM=4 DEFAULT_FLOAT=half GPUS=6 BASEDIR=/path/to/wiki python3 test/external/mlperf_bert/external_test_checkpoint_loading.py
import os
from tqdm import tqdm

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_state_dict

from examples.mlperf.helpers import get_mlperf_bert_model, get_data_bert
from examples.mlperf.dataloader import batch_load_val_bert
from examples.mlperf.model_train import eval_step_bert

if __name__ == "__main__":
  BASEDIR = os.environ["BASEDIR"] = getenv("BASEDIR", "/raid/datasets/wiki")
  INIT_CKPT_DIR = getenv("INIT_CKPT_DIR", BASEDIR)
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1))]
  EVAL_BS = getenv("EVAL_BS", 4 * len(GPUS))
  max_eval_steps = (10000 + EVAL_BS - 1) // EVAL_BS

  for i in range(10):
    assert os.path.exists(os.path.join(BASEDIR, "eval", f"{i}.pkl")), \
      f"File {i}.pkl does not exist in {os.path.join(BASEDIR, 'eval')}"

  required_files = ["checkpoint", "model.ckpt-28252.data-00000-of-00001", "model.ckpt-28252.index", "model.ckpt-28252.meta"]
  assert all(os.path.exists(os.path.join(INIT_CKPT_DIR, f)) for f in required_files), \
    f"Missing checkpoint files in INIT_CKPT_DIR: {required_files}"

  Tensor.training = False

  model = get_mlperf_bert_model(INIT_CKPT_DIR)

  for _, x in get_state_dict(model).items():
    x.realize().to_(GPUS)

  eval_accuracy = []
  eval_it = iter(batch_load_val_bert(EVAL_BS))

  for _ in tqdm(range(max_eval_steps), desc="Evaluating", total=max_eval_steps):
    eval_data = get_data_bert(GPUS, eval_it)
    eval_result: dict[str, Tensor] = eval_step_bert(model, eval_data["input_ids"], eval_data["segment_ids"], eval_data["input_mask"], \
                                               eval_data["masked_lm_positions"], eval_data["masked_lm_ids"], \
                                               eval_data["masked_lm_weights"], eval_data["next_sentence_labels"])

    mlm_accuracy = eval_result["masked_lm_accuracy"].numpy().item()
    eval_accuracy.append(mlm_accuracy)

  total_lm_accuracy = sum(eval_accuracy) / len(eval_accuracy)
  assert total_lm_accuracy >= 0.34, "Checkpoint loaded incorrectly. Accuracy should be very close to 0.34085 as per MLPerf BERT README."
  print(f"Checkpoint loaded correctly. Accuracy of {total_lm_accuracy*100:.3f}% achieved. (Reference: 34.085%)")
