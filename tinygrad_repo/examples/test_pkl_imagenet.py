import sys, pickle
from tinygrad import GlobalCounters
from tinygrad.helpers import fetch, getenv
from examples.test_onnx_imagenet import imagenet_dataloader

if __name__ == "__main__":
  with open(fetch(sys.argv[1]), "rb") as f:
    run_onnx_jit = pickle.load(f)
  input_name = run_onnx_jit.captured.expected_names[0]
  device = run_onnx_jit.captured.expected_st_vars_dtype_device[0][-1]
  print(f"input goes into {input_name=} on {device=}")
  hit = 0
  for i,(img,y) in enumerate(imagenet_dataloader(cnt=getenv("CNT", 100))):
    GlobalCounters.reset()
    p = run_onnx_jit(**{input_name:img.to(device)})
    assert p.shape == (1,1000)
    t = p.to('cpu').argmax().item()
    hit += y==t
    print(f"target: {y:3d}  pred: {t:3d}  acc: {hit/(i+1)*100:.2f}%")
