#!/usr/bin/env python3
import numpy as np
import coremltools as ct
from coremltools.models.neural_network import datatypes, NeuralNetworkBuilder

# KxK GEMM with bias
K = 64

input_features = [('image', datatypes.Array(K))]
input_features2 = [('image2', datatypes.Array(K))]
output_features = [('probs', datatypes.Array(K))]

weights = np.zeros((K, K)) + 3
bias = np.ones(K)

builder = NeuralNetworkBuilder(input_features+input_features2, output_features)

#builder.add_inner_product(name='ip_layer', W=weights, b=None, input_channels=K, output_channels=K, has_bias=False, input_name='image', output_name='med')
#builder.add_inner_product(name='ip_layer_2', W=weights, b=None, input_channels=3, output_channels=3, has_bias=False, input_name='med', output_name='probs')
builder.add_elementwise(name='element', input_names=['image', 'image2'], output_name='probs', mode='ADD')
#builder.add_bias(name='bias', b=bias, input_name='med', output_name='probs', shape_bias=(K,))
#builder.add_activation(name='act_layer', non_linearity='SIGMOID', input_name='med', output_name='probs')

# compile the spec
mlmodel = ct.models.MLModel(builder.spec)

# trigger the ANE!
out = mlmodel.predict({"image": np.zeros(K, dtype=np.float32)+1, "image2": np.zeros(K, dtype=np.float32)+2})
print(out)
mlmodel.save('test.mlmodel')
