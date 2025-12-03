# Implementation of waifu2x vgg7 in tinygrad.
# Obviously, not developed, supported, etc. by the original waifu2x author(s).

import numpy
from tinygrad.tensor import Tensor
from PIL import Image
from tinygrad.helpers import fetch

# File Formats

# tinygrad convolution tensor input layout is (1,c,y,x) - and therefore the form for all images used in the project
# tinygrad convolution tensor weight layout is (outC,inC,H,W) - this matches NCNN (and therefore KINNE), but not waifu2x json

def image_load(path) -> numpy.ndarray:
  """
  Loads an image in the shape expected by other functions in this module.
  Doesn't Tensor it, in case you need to do further work with it.
  """
  # file
  na = numpy.array(Image.open(path))
  if na.shape[2] == 4:
    # RGBA -> RGB (covers opaque images with alpha channels)
    na = na[:,:,0:3]
  # fix shape
  na = numpy.moveaxis(na, [2,0,1], [0,1,2])
  # shape is now (3,h,w), add 1
  na = na.reshape(1,3,na.shape[1],na.shape[2])
  # change type
  na = na.astype("float32") / 255.0
  return na

def image_save(path, na: numpy.ndarray):
  """
  Saves an image of the shape expected by other functions in this module.
  However, note this expects a numpy array.
  """
  # change type
  na = numpy.fmax(numpy.fmin(na * 255.0, 255), 0).astype("uint8")
  # shape is now (1,3,h,w), remove 1
  na = na.reshape(3,na.shape[2],na.shape[3])
  # fix shape
  na = numpy.moveaxis(na, [0,1,2], [2,0,1])
  # shape is now (h,w,3)
  # file
  Image.fromarray(na).save(path)

# The Model

class Conv3x3Biased:
  """
  A 3x3 convolution layer with some utility functions.
  """
  def __init__(self, inC, outC, last = False):
    # The properties must be named as "W" and "b".
    # This is in an attempt to try and be roughly compatible with https://github.com/FHPythonUtils/Waifu2x
    #  though this cannot necessarily account for transposition and other such things.

    # Massively overstate the weights to get them to be focused on,
    #  since otherwise the biases overrule everything
    self.W = Tensor.uniform(outC, inC, 3, 3) * 16.0
    # Layout-wise, blatant cheat, but serious_mnist does it. I'd guess channels either have to have a size of 1 or whatever the target is?
    # Values-wise, entirely different blatant cheat.
    # In most cases, use uniform bias, but tiny.
    # For the last layer, use just 0.5, constant.
    if last:
      self.b = Tensor.zeros(1, outC, 1, 1) + 0.5
    else:
      self.b = Tensor.uniform(1, outC, 1, 1)

  def forward(self, x):
    # You might be thinking, "but what about padding?"
    # Answer: Tiling is used to stitch everything back together, though you could pad the image before providing it.
    return x.conv2d(self.W).add(self.b)

  def get_parameters(self) -> list:
    return [self.W, self.b]

  def load_waifu2x_json(self, layer: dict):
    # Weights in this file are outChannel,inChannel,X,Y.
    # Not outChannel,inChannel,Y,X.
    # Therefore, transpose it before assignment.
    # I have long since forgotten how I worked this out.
    self.W.assign(Tensor(layer["weight"]).reshape(shape=self.W.shape).transpose(2, 3))
    self.b.assign(Tensor(layer["bias"]).reshape(shape=self.b.shape))

class Vgg7:
  """
  The 'vgg7' waifu2x network.
  Lower quality and slower than even upconv7 (nevermind cunet), but is very easy to implement and test.
  """

  def __init__(self):
    self.conv1 = Conv3x3Biased(3, 32)
    self.conv2 = Conv3x3Biased(32, 32)
    self.conv3 = Conv3x3Biased(32, 64)
    self.conv4 = Conv3x3Biased(64, 64)
    self.conv5 = Conv3x3Biased(64, 128)
    self.conv6 = Conv3x3Biased(128, 128)
    self.conv7 = Conv3x3Biased(128, 3, True)

  def forward(self, x):
    """
    Forward pass: Actually runs the network.
    Input format: (1, 3, Y, X)
    Output format: (1, 3, Y - 14, X - 14)
    (the - 14 represents the 7-pixel context border that is lost)
    """
    x = self.conv1.forward(x).leaky_relu(0.1)
    x = self.conv2.forward(x).leaky_relu(0.1)
    x = self.conv3.forward(x).leaky_relu(0.1)
    x = self.conv4.forward(x).leaky_relu(0.1)
    x = self.conv5.forward(x).leaky_relu(0.1)
    x = self.conv6.forward(x).leaky_relu(0.1)
    x = self.conv7.forward(x)
    return x

  def get_parameters(self) -> list:
    return self.conv1.get_parameters() + self.conv2.get_parameters() + self.conv3.get_parameters() + self.conv4.get_parameters() + self.conv5.get_parameters() + self.conv6.get_parameters() + self.conv7.get_parameters()

  def load_from_pretrained(self, intent = "art", subtype = "scale2.0x"):
    """
    Downloads a nagadomi/waifu2x JSON weight file and loads it.
    """
    import json
    data = json.loads(fetch("https://github.com/nagadomi/waifu2x/raw/master/models/vgg_7/" + intent + "/" + subtype + "_model.json").read_bytes())
    self.load_waifu2x_json(data)

  def load_waifu2x_json(self, data: list):
    """
    Loads weights from one of the waifu2x JSON files, i.e. waifu2x/models/vgg_7/art/noise0_model.json
    data (passed in) is assumed to be the output of json.load or some similar on such a file
    """
    self.conv1.load_waifu2x_json(data[0])
    self.conv2.load_waifu2x_json(data[1])
    self.conv3.load_waifu2x_json(data[2])
    self.conv4.load_waifu2x_json(data[3])
    self.conv5.load_waifu2x_json(data[4])
    self.conv6.load_waifu2x_json(data[5])
    self.conv7.load_waifu2x_json(data[6])

  def forward_tiled(self, image: numpy.ndarray, tile_size: int) -> numpy.ndarray:
    """
    Given an ndarray image as loaded by image_load (NOT a tensor), scales it, pads it, splits it up, forwards the pieces, and reconstitutes it.
    Note that you really shouldn't try to run anything not (1, 3, *, *) through this.
    """
    # Constant that only really gets repeated a ton here.
    context = 7
    context2 = context + context

    # Notably, numpy is used here because it makes this fine manipulation a lot simpler.
    # Scaling first - repeat on axis 2 and axis 3 (Y & X)
    image = image.repeat(2, 2).repeat(2, 3)

    # Resulting image buffer. This is made before the input is padded,
    #  since the input has the padded shape right now.
    image_out = numpy.zeros(image.shape)

    # Padding next. Note that this padding is done on the whole image.
    # Padding the tiles would lose critical context, cause seams, etc.
    image = numpy.pad(image, [[0, 0], [0, 0], [context, context], [context, context]], mode = "edge")

    # Now for tiling.
    # The output tile size is the usable output from an input tile (tile_size).
    # As such, the tiles overlap.
    out_tile_size = tile_size - context2
    for out_y in range(0, image_out.shape[2], out_tile_size):
      for out_x in range(0, image_out.shape[3], out_tile_size):
        # Input is sourced from the same coordinates, but some stuff ought to be
        #  noted here for future reference:
        # + out_x/y's equivalent position w/ the padding is out_x + context.
        # + The output, however, is without context. Input needs context.
        # + Therefore, the input rectangle is expanded on all sides by context.
        # + Therefore, the input position has the context subtracted again.
        # + Therefore:
        in_y = out_y
        in_x = out_x
        # not shown: in_w/in_h = tile_size (as opposed to out_tile_size)
        # Extract tile.
        # Note that numpy will auto-crop this at the bottom-right.
        # This will never be a problem, as tiles are specifically chosen within the padded section.
        tile = image[:, :, in_y:in_y + tile_size, in_x:in_x + tile_size]
        # Extracted tile dimensions -> output dimensions
        # This is important because of said cropping, otherwise it'd be interior tile size.
        out_h = tile.shape[2] - context2
        out_w = tile.shape[3] - context2
        # Process tile.
        tile_t = Tensor(tile)
        tile_fwd_t = self.forward(tile_t)
        # Replace tile.
        image_out[:, :, out_y:out_y + out_h, out_x:out_x + out_w] = tile_fwd_t.numpy()

    return image_out

