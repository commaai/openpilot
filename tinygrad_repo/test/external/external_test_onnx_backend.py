import unittest
from typing import Any, Tuple
from onnx.backend.base import Backend, BackendRep
import onnx.backend.test
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv, OSX
from tinygrad.device import is_dtype_supported
from tinygrad.frontend.onnx import OnnxRunner

# pip3 install tabulate
pytest_plugins = 'onnx.backend.test.report',

class TinygradModel(BackendRep):
  def __init__(self, run_onnx, input_names):
    super().__init__()
    self.fxn = run_onnx
    self.input_names = input_names

  def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
    real_inputs = dict(zip(self.input_names, inputs))
    ret = self.fxn(real_inputs, debug=2)
    return tuple(x.numpy() if isinstance(x, Tensor) else [i.numpy() for i in x] if isinstance(x, list) else np.array(x) for x in ret.values())

class TinygradBackend(Backend):
  @classmethod
  def prepare(cls, model: onnx.ModelProto, device):
    input_all = [x.name for x in model.graph.input]
    input_initializer = [x.name for x in model.graph.initializer]
    net_feed_input = [x for x in input_all if x not in input_initializer]
    print("prepare", cls, device, net_feed_input)
    model = Tensor(model.SerializeToString(), device="PYTHON")
    run_onnx = OnnxRunner(model)
    return TinygradModel(run_onnx, net_feed_input)

  @classmethod
  def supports_device(cls, device: str) -> bool:
    # NOTE: this is onnx CPU
    return device == "CPU"

backend_test = onnx.backend.test.BackendTest(TinygradBackend, __name__)

# BUG: buggy onnx tests
backend_test.exclude('test_adam_multiple_cpu')

# BUG: ORT fails these with runtime error
backend_test.exclude('test_PReLU_1d_multiparam_cpu')
backend_test.exclude('test_PReLU_2d_multiparam_cpu')
backend_test.exclude('test_PReLU_3d_multiparam_cpu')

# BUG: we don't match ORT here due to some div inaccuracy with floats
backend_test.exclude('test_dynamicquantizelinear_cpu')
backend_test.exclude('test_dynamicquantizelinear_expanded_cpu')

# BUG: ORT fails these with numerical error but we match ORT numerically
# see: https://onnx.ai/backend-scoreboard/onnxruntime_details_stable.html
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_qlinearmatmul_2D_int8_float16
backend_test.exclude('test_qlinearmatmul_2D_int8_float16_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_qlinearmatmul_3D_int8_float16
backend_test.exclude('test_qlinearmatmul_3D_int8_float16_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_qlinearmatmul_2D_int8_float32
backend_test.exclude('test_qlinearmatmul_2D_int8_float32_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_qlinearmatmul_3D_int8_float32
backend_test.exclude('test_qlinearmatmul_3D_int8_float32_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_maxunpool_export_with_output_shape
backend_test.exclude('test_maxunpool_export_with_output_shape_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True
backend_test.exclude('test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_True_cpu')
# tested in external_test_onnx_ops.py::TestMainOnnxOps.test_resize_downsample_scales_linear_align_corners
backend_test.exclude('test_resize_downsample_scales_linear_align_corners_cpu')

# about different dtypes
if not is_dtype_supported(dtypes.float64):
  backend_test.exclude('float64')
  backend_test.exclude('DOUBLE')
  # these have float64 inputs
  backend_test.exclude('test_eyelike_with_dtype_cpu')
  backend_test.exclude('test_reduce_log_sum_exp*')
  backend_test.exclude('test_operator_add*')
  backend_test.exclude('test_einsum_*')
  backend_test.exclude('test_cumsum_*')

if not is_dtype_supported(dtypes.float16):
  backend_test.exclude('float16')
  backend_test.exclude('FLOAT16')

# dtype cast
backend_test.exclude('STRING')
backend_test.exclude('FLOAT8')
backend_test.exclude('INT4')
backend_test.exclude('UINT4')
backend_test.exclude('BFLOAT16')  # not supported in numpy
backend_test.exclude('FLOAT4E2M1')

backend_test.exclude('test_dequantizelinear_int4_cpu')
backend_test.exclude('test_dequantizelinear_uint4_cpu')
backend_test.exclude('test_quantizelinear_int4_cpu')
backend_test.exclude('test_quantizelinear_uint4_cpu')

# no support for FLOAT8
backend_test.exclude('test_quantizelinear_e4m3fn_cpu')
backend_test.exclude('test_quantizelinear_e5m2_cpu')
backend_test.exclude('test_quantizelinear_e4m3fn_cpu')
backend_test.exclude('test_quantizelinear_e5m2_cpu')
backend_test.exclude('test_quantizelinear_float4e2m1_cpu')
backend_test.exclude('test_dequantizelinear_e4m3fn_cpu')
backend_test.exclude('test_dequantizelinear_e4m3fn_zero_point_cpu')
backend_test.exclude('test_dequantizelinear_e4m3fn_float16_cpu')
backend_test.exclude('test_dequantizelinear_e5m2_cpu')
backend_test.exclude('test_dequantizelinear_float4e2m1_cpu')

# we don't support indexes
backend_test.exclude('test_nonzero_*')

# no support for int pow
backend_test.exclude('test_pow_types_int32_int32_cpu')
backend_test.exclude('test_pow_types_int64_int64_cpu')

# no boolean ops (2d, 3d, 4d)
backend_test.exclude('test_bitshift_*')

# no string ops
backend_test.exclude('string')
backend_test.exclude('test_strnorm_*')
backend_test.exclude('test_regex_*')

# no rnn
backend_test.exclude('test_gru_*')
backend_test.exclude('test_rnn_*')
backend_test.exclude('test_lstm_*')
backend_test.exclude('test_simple_rnn_*')

# no control flow
# control flow uses AttributeProto.GRAPH
backend_test.exclude('test_if_*')
backend_test.exclude('test_loop*')
backend_test.exclude('test_range_float_type_positive_delta_expanded_cpu') # requires loop
backend_test.exclude('test_affine_grid_2d_align_corners_expanded_cpu')
backend_test.exclude('test_affine_grid_2d_expanded_cpu')
backend_test.exclude('test_affine_grid_3d_align_corners_expanded_cpu')
backend_test.exclude('test_affine_grid_3d_expanded_cpu')
backend_test.exclude('test_range_int32_type_negative_delta_expanded_cpu')

# unsupported (strange) ops
backend_test.exclude('test_blackmanwindow_*')
backend_test.exclude('test_bernoulli_*')
backend_test.exclude('test_det_*')
backend_test.exclude('test_col2im_*')
backend_test.exclude('test_hammingwindow_*')
backend_test.exclude('test_hannwindow_*')
backend_test.exclude('test_hardmax_*')
backend_test.exclude('test_gridsample_*')
backend_test.exclude('test_dft_*')
backend_test.exclude('test_einsum_batch_diagonal_cpu*') # TODO: equation = '...ii ->...i'
backend_test.exclude('test_einsum_inner_prod_cpu*') # TODO: equation = 'i,i'
backend_test.exclude('test_unique_*')
backend_test.exclude('test_sequence_*')
backend_test.exclude('test_nonmaxsuppression_*')
backend_test.exclude('test_reversesequence_*')
backend_test.exclude('test_roialign_*')
backend_test.exclude('test_tfidfvectorizer_*')
backend_test.exclude('test_stft_*')
backend_test.exclude('test_melweightmatrix_*')

# more strange ops
backend_test.exclude('test_basic_deform_conv_*')
backend_test.exclude('test_deform_conv_*')
backend_test.exclude('test_lppool_*')
backend_test.exclude('test_scan_*')
backend_test.exclude('test_split_to_sequence_*')
backend_test.exclude('test_resize_downsample_scales_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_downsample_sizes_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_upsample_scales_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_resize_upsample_sizes_cubic_*') # unsure how to implement cubic
backend_test.exclude('test_ai_onnx_ml_tree_ensemble_*') # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/aionnxml/op_tree_ensemble.py#L121

# rest of the failing tests
backend_test.exclude('test_resize_tf_crop_and_resize_cpu') # tf_crop_and_resize not implemented
backend_test.exclude('test_resize_tf_crop_and_resize_axes_2_3_cpu') # tf_crop_and_resize not implemented
backend_test.exclude('test_resize_tf_crop_and_resize_axes_3_2_cpu') # tf_crop_and_resize not implemented
backend_test.exclude('test_resize_tf_crop_and_resize_extrapolation_value_cpu') # tf_crop_and_resize value not implemented
backend_test.exclude('test_resize_downsample_scales_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_resize_downsample_sizes_linear_antialias_cpu') # antialias not implemented
backend_test.exclude('test_ai_onnx_ml_label_encoder_tensor_value_only_mapping_cpu') # bad data type string
backend_test.exclude('test_ai_onnx_ml_label_encoder_tensor_mapping_cpu') # bad data type string

backend_test.exclude('test_scatternd_min_cpu') # min not yet supported
backend_test.exclude('test_scatternd_max_cpu') # max not yet supported

# regression from removing StrEnum in Domain
backend_test.exclude('test_adam_cpu')
backend_test.exclude('test_gradient_of_add_and_mul_cpu')
backend_test.exclude('test_gradient_of_add_cpu')

if Device.DEFAULT in ['GPU', 'METAL']:
  backend_test.exclude('test_resize_upsample_sizes_nearest_axes_2_3_cpu')
  backend_test.exclude('test_resize_upsample_sizes_nearest_axes_3_2_cpu')
  backend_test.exclude('test_resize_upsample_sizes_nearest_cpu')

if Device.DEFAULT == "METAL" or (OSX and Device.DEFAULT == "GPU"):
  # numerical inaccuracy
  backend_test.exclude('test_mish_cpu')
  backend_test.exclude('test_mish_expanded_cpu')

# disable model tests for now since they are slow
if not getenv("MODELTESTS"):
  for x in backend_test.test_suite:
    if 'OnnxBackendRealModelTest' in str(type(x)):
      backend_test.exclude(str(x).split(" ")[0])
else:
  # model tests all pass!
  backend_test.include('test_resnet50')
  backend_test.include('test_inception_v1')
  backend_test.include('test_inception_v2')
  backend_test.include('test_densenet121')
  backend_test.include('test_shufflenet')
  backend_test.include('test_squeezenet')
  backend_test.include('test_bvlc_alexnet')
  backend_test.include('test_zfnet512')
  backend_test.include('test_vgg19')

globals().update(backend_test.enable_report().test_cases)

if __name__ == '__main__':
  unittest.main()
