# load weights from
# https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
# a rough copy of
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
import sys
import ast
import time
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, fetch, Timing
from tinygrad.engine.jit import TinyJit
from extra.models.efficientnet import EfficientNet
np.set_printoptions(suppress=True)

# TODO: you should be able to put these in the jitted function
bias = Tensor([0.485, 0.456, 0.406])
scale = Tensor([0.229, 0.224, 0.225])

@TinyJit
def _infer(model, img):
  img = img.permute((2,0,1))
  img = img / 255.0
  img = img - bias.reshape((1,-1,1,1))
  img = img / scale.reshape((1,-1,1,1))
  return model.forward(img).realize()

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  """
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  """

  # run the net
  out = _infer(model, Tensor(img.astype("float32"))).numpy()

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  # instantiate my net
  model = EfficientNet(getenv("NUM", 0))
  model.load_from_pretrained()

  # category labels
  lbls = ast.literal_eval(fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt").read_text())

  # load image and preprocess
  url = sys.argv[1] if len(sys.argv) >= 2 else "https://raw.githubusercontent.com/tinygrad/tinygrad/master/docs/showcase/stable_diffusion_by_tinygrad.jpg"
  if url == 'webcam':
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])
      lt = time.monotonic_ns()
      out, retimg = infer(model, img)
      print(f"{(time.monotonic_ns()-lt)*1e-6:7.2f} ms", np.argmax(out), np.max(out), lbls[np.argmax(out)])
      SCALE = 3
      simg = cv2.resize(retimg, (224*SCALE, 224*SCALE))
      retimg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
      cv2.imshow('capture', retimg)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
  else:
    img = Image.open(fetch(url))
    for i in range(getenv("CNT", 1)):
      with Timing("did inference in "):
        out, _ = infer(model, img)
        print(np.argmax(out), np.max(out), lbls[np.argmax(out)])
