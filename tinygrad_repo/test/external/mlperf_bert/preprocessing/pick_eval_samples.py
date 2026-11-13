# https://github.com/mlcommons/training/blob/1c8a098ae3e70962a4f7422c0b0bd35ae639e357/language_model/tensorflow/bert/cleanup_scripts/pick_eval_samples.py
# NOTE: This is a direct copy of the original script
"""Script for picking certain number of sampels.
"""

import argparse
import time
import logging
import collections
import tensorflow as tf

parser = argparse.ArgumentParser(
    description="Eval sample picker for BERT.")
parser.add_argument(
    '--input_tfrecord',
    type=str,
    default='',
    help='Input tfrecord path')
parser.add_argument(
    '--output_tfrecord',
    type=str,
    default='',
    help='Output tfrecord path')
parser.add_argument(
    '--num_examples_to_pick',
    type=int,
    default=10000,
    help='Number of examples to pick')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=512,
    help='The maximum number of tokens within a sequence.')
parser.add_argument(
    '--max_predictions_per_seq',
    type=int,
    default=76,
    help='The maximum number of predictions within a sequence.')
args = parser.parse_args()

max_seq_length = args.max_seq_length
max_predictions_per_seq = args.max_predictions_per_seq
logging.basicConfig(level=logging.INFO)

def decode_record(record):
  """Decodes a record to a TensorFlow example."""
  name_to_features = {
      "input_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "masked_lm_positions":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_ids":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_weights":
          tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
      "next_sentence_labels":
          tf.FixedLenFeature([1], tf.int64),
  }

  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


if __name__ == '__main__':
  tic = time.time()
  tf.enable_eager_execution()

  d = tf.data.TFRecordDataset(args.input_tfrecord)
  num_examples = 0
  records = []
  for record in d:
    records.append(record)
    num_examples += 1

  writer = tf.python_io.TFRecordWriter(args.output_tfrecord)
  i = 0
  pick_ratio = num_examples / args.num_examples_to_pick
  num_examples_picked = 0
  for i in range(args.num_examples_to_pick):
    example = decode_record(records[int(i * pick_ratio)])
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(
        example["input_ids"].numpy())
    features["input_mask"] = create_int_feature(
        example["input_mask"].numpy())
    features["segment_ids"] = create_int_feature(
        example["segment_ids"].numpy())
    features["masked_lm_positions"] = create_int_feature(
        example["masked_lm_positions"].numpy())
    features["masked_lm_ids"] = create_int_feature(
        example["masked_lm_ids"].numpy())
    features["masked_lm_weights"] = create_float_feature(
        example["masked_lm_weights"].numpy())
    features["next_sentence_labels"] = create_int_feature(
        example["next_sentence_labels"].numpy())

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    num_examples_picked += 1

  writer.close()
  toc = time.time()
  logging.info("Picked %d examples out of %d samples in %.2f sec",
               num_examples_picked, num_examples, toc - tic)
