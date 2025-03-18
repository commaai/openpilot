# USAGE:
# 1.  Download raw text data with `wikipedia_download.py`

# 2.  Install python==3.7.12 and tensorflow==1.15.5
#     Run `create_pretraining_data.py` to create TFRecords on specific part (This will take some time)
#     Command: python3 create_pretraining_data.py --input_file=/path/to/part-00XXX-of-00500 --vocab_file=/path/to/vocab.txt \
#                              --output_file=/path/to/output.tfrecord --max_seq_length=512 --max_predictions_per_seq=76
#
# 2.1 For eval: --input_file=/path/to/eval.txt and
#     Command: python3 pick_eval_samples.py --input_tfrecord=/path/to/eval.tfrecord --output_tfrecord=/path/to/output_eval.tfrecord

# 3.  Run `wikipedia.py` to preprocess the data with tinygrad (Use python > 3.7)
#     Command: BASEDIR=/path/to/basedir python3 wikipedia.py pre-train X (NOTE: part number needs to match part of step 2)
#     This will output to /path/to/basedir/train/X.pkl
#
# 3.1 For eval:
#     Command: BASEDIR=/path/to/basedir python3 wikipedia.py pre-eval
#     This will output to /path/to/basedir/eval.pkl

# 4.  Run this script to verify the correctness of the preprocessing script for specific part
#     Command: python3 external_test_preprocessing_part.py --preprocessed_part=/path/to/basedir/train/X.pkl --tf_records=/path/to/output.tfrecord
import os, argparse, pickle
from tqdm import tqdm

# This is a workaround for protobuf issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def _parse_function(proto, max_seq_length, max_predictions_per_seq):
  feature_description = {
      'input_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([max_seq_length], tf.int64),
      'masked_lm_positions': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_ids': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_weights': tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
      'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
  }
  return tf.io.parse_single_example(proto, feature_description)

def load_dataset(file_path, max_seq_length=512, max_predictions_per_seq=76):
  dataset = tf.data.TFRecordDataset(file_path)
  parse_function = lambda proto: _parse_function(proto, max_seq_length, max_predictions_per_seq) # noqa: E731
  return dataset.map(parse_function)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verify the correctness of the preprocessing script for specific part",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--preprocessed_part", type=str, default=None,
                      help="Path to preprocessed samples file from `wikipedia.py`")
  parser.add_argument("--tf_records", type=str, default=None,
                      help="Path to TFRecords file from `create_pretraining_data.py` (Reference implementation)")
  parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length. For MLPerf keep it at 512")
  parser.add_argument("--max_predictions_per_seq", type=int, default=76, help="Max predictions per sequence. For MLPerf keep it at 76")
  parser.add_argument("--is_eval", type=bool, default=False, help="Whether to run eval or train preprocessing")
  args = parser.parse_args()

  assert os.path.isfile(args.preprocessed_part), f"The specified file {args.preprocessed_part} does not exist."
  assert os.path.isfile(args.tf_records), f"The specified TFRecords file {args.tf_records} does not exist."

  with open(args.preprocessed_part, 'rb') as f:
    preprocessed_samples = pickle.load(f)

  dataset = load_dataset(args.tf_records, args.max_seq_length, args.max_predictions_per_seq)
  tf_record_count = sum(1 for _ in dataset)
  assert tf_record_count == len(preprocessed_samples), f"Samples in reference: {tf_record_count} != Preprocessed samples: {len(preprocessed_samples)}"
  print(f"Total samples in the part: {tf_record_count}")

  for i, (reference_example, preprocessed_sample) in tqdm(enumerate(zip(dataset, preprocessed_samples)), desc="Checking samples", total=len(preprocessed_samples)): # noqa: E501
    feature_keys = ["input_ids", "input_mask", "segment_ids", "masked_lm_positions", "masked_lm_ids", "masked_lm_weights", "next_sentence_labels"]
    for key in feature_keys:
      reference_example_feature = reference_example[key].numpy()
      assert (reference_example_feature == preprocessed_sample[key]).all(), \
      f"{key} are not equal at index {i}\nReference: {reference_example_feature}\nPreprocessed: {preprocessed_sample[key]}"
