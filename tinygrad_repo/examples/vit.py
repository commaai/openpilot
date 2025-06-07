import ast
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv, fetch
from extra.models.vit import ViT
"""
fn = "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"
import tensorflow as tf
with tf.io.gfile.GFile(fn, "rb") as f:
  dat = f.read()
  with open("cache/"+ fn.rsplit("/", 1)[1], "wb") as g:
    g.write(dat)
"""

Tensor.training = False
if getenv("LARGE", 0) == 1:
  m = ViT(embed_dim=768, num_heads=12)
else:
  # tiny
  m = ViT(embed_dim=192, num_heads=3)
m.load_from_pretrained()

# category labels
lbls = ast.literal_eval(fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt").read_text())

#url = "https://upload.wikimedia.org/wikipedia/commons/4/41/Chicken.jpg"
url = "https://repository-images.githubusercontent.com/296744635/39ba6700-082d-11eb-98b8-cb29fb7369c0"

# junk
img = Image.open(fetch(url))
aspect_ratio = img.size[0] / img.size[1]
img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
img = np.array(img)
y0,x0=(np.asarray(img.shape)[:2]-224)//2
img = img[y0:y0+224, x0:x0+224]
img = np.moveaxis(img, [2,0,1], [0,1,2])
img = img.astype(np.float32)[:3].reshape(1,3,224,224)
img /= 255.0
img -= 0.5
img /= 0.5

out = m.forward(Tensor(img))
outnp = out.numpy().ravel()
choice = outnp.argmax()
print(out.shape, choice, outnp[choice], lbls[choice])
