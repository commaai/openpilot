#!/usr/bin/env python3
import sys
import tensorflow as tf  # pylint: disable=import-error

with open(sys.argv[1], "rb") as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())
  #tf.io.write_graph(graph_def, '', sys.argv[1]+".try")
