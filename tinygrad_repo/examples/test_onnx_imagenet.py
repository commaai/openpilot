import random, sys
import numpy as np
from extra.datasets.imagenet import get_imagenet_categories, get_val_files, center_crop
from examples.benchmark_onnx import load_onnx_model
from PIL import Image
from tinygrad import Tensor, dtypes, GlobalCounters
from tinygrad.helpers import fetch, getenv

# works:
#  ~70% - https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
#  ~43% - https://github.com/onnx/models/raw/refs/heads/main/Computer_Vision/alexnet_Opset16_torch_hub/alexnet_Opset16.onnx
#  ~72% - https://github.com/xamcat/mobcat-samples/raw/refs/heads/master/onnx_runtime/InferencingSample/InferencingSample/mobilenetv2-7.onnx
#  ~71% - https://github.com/axinc-ai/onnx-quantization/raw/refs/heads/main/models/mobilenetv2_1.0.opt.onnx
#  ~67% - https://github.com/xamcat/mobcat-samples/raw/refs/heads/master/onnx_runtime/InferencingSample/InferencingSample/mobilenetv2-7-quantized.onnx
# broken:
#  https://github.com/MTlab/onnx2caffe/raw/refs/heads/master/model/MobileNetV2.onnx
#  https://huggingface.co/qualcomm/MobileNet-v2-Quantized/resolve/main/MobileNet-v2-Quantized.onnx
#  ~35% - https://github.com/axinc-ai/onnx-quantization/raw/refs/heads/main/models/mobilenev2_quantized.onnx

# QUANT=1 python3 examples/test_onnx_imagenet.py
#   https://github.com/xamcat/mobcat-samples/raw/refs/heads/master/onnx_runtime/InferencingSample/InferencingSample/mobilenetv2-7.onnx
# DONT_REALIZE_EXPAND=1 python3 examples/test_onnx_imagenet.py /tmp/model.quant.onnx
# VIZ=1 DONT_REALIZE_EXPAND=1 python3 examples/benchmark_onnx.py /tmp/model.quant.onnx

def imagenet_dataloader(cnt=0):
  input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
  input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
  files = get_val_files()
  random.shuffle(files)
  if cnt != 0: files = files[:cnt]
  cir = get_imagenet_categories()
  for fn in files:
    img = Image.open(fn)
    img = img.convert('RGB') if img.mode != "RGB" else img
    img = center_crop(img)
    img = np.array(img)
    img = Tensor(img).permute(2,0,1).reshape(1,3,224,224)
    img = ((img.cast(dtypes.float32)/255.0) - input_mean) / input_std
    y = cir[fn.split("/")[-2]]
    yield img,y

if __name__ == "__main__":
  fn = sys.argv[1]
  if getenv("QUANT"):
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantFormat, QuantType, CalibrationDataReader
    model_fp32 = fetch(fn)
    fn = '/tmp/model.quant.onnx'
    if getenv("DYNAMIC"):
      quantize_dynamic(model_fp32, fn)
    else:
      class ImagenetReader(CalibrationDataReader):
        def __init__(self):
          self.iter = imagenet_dataloader(cnt=1000)
        def get_next(self) -> dict:
          try:
            img,y = next(self.iter)
          except StopIteration:
            return None
          return {"input": img.numpy()}
      quantize_static(model_fp32, fn, ImagenetReader(), quant_format=QuantFormat.QDQ, per_channel=False,
                      activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                      extra_options={"ActivationSymmetric": False})

  run_onnx_jit, input_specs = load_onnx_model(fetch(fn))
  t_name, t_spec = list(input_specs.items())[0]
  assert t_spec.shape[1:] == (3,224,224), f"shape is {t_spec.shape}"

  hit = 0
  for i,(img,y) in enumerate(imagenet_dataloader(cnt=getenv("CNT", 100))):
    GlobalCounters.reset()
    p = run_onnx_jit(**{t_name:img})
    assert p.shape == (1,1000)
    t = p.argmax().item()
    hit += y==t
    print(f"target: {y:3d}  pred: {t:3d}  acc: {hit/(i+1)*100:.2f}%")

  import pickle
  with open("/tmp/im.pkl", "wb") as f: pickle.dump(run_onnx_jit, f)
