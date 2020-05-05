from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
from tools.lib.framereader import FrameReader
import matplotlib
import matplotlib.pyplot as plt
# imgs just a list of images as in YUV format
fr = FrameReader("leon.hevc")

imgs = []
for i in tqdm(range(100)):
  imgs.append(fr.get(i, pix_fmt='yuv420p')[0].reshape((874*3//2, 1164)))

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


from tensorflow.keras.models import load_model
supercombo = load_model('supercombo.keras')
print(supercombo.summary())
# Just passing zeros for desire and state
poses = []
state = np.zeros((1,512))
## Saving lane data 
left_lane = []
right_lane = []

for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], np.zeros((1,8)), state]
  outs = supercombo.predict(inputs)
  poses.append(outs[-2])
  state = outs[-1]
  plt.clf()
  plt.title("lanes and path")
  plt.scatter(parsed["lll"], range(0,192), c="b")
  plt.scatter(parsed["rll"], range(0, 192), c="r")
  plt.scatter(parsed["path"], range(0, 192), c="g")
  plt.gca().invert_xaxis()
  plt.pause(0.05)
  count += 1
plt.show()