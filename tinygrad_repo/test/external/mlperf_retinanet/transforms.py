# Copied from https://github.com/mlcommons/training/blob/637c82f9e699cd6caf108f92efb2c1d446b630e0/single_stage_detector/ssd/transforms.py

import torch
import torchvision

from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from typing import Any

try:
    import accimage
except ImportError:
    accimage = None

@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def get_image_size_tensor(img: Tensor) -> List[int]:
    # Returns (w, h) of tensor image
    torchvision.transforms._functional_tensor._assert_image_tensor(img)
    return [img.shape[-1], img.shape[-2]]

@torch.jit.unused
def get_image_size_pil(img: Any) -> List[int]:
    if _is_pil_image(img):
        return list(img.size)
    raise TypeError("Unexpected type {}".format(type(img)))

def get_image_size(img: Tensor) -> List[int]:
    """Returns the size of an image as [width, height].
    Args:
        img (PIL Image or Tensor): The image to be checked.
    Returns:
        List[int]: The image size.
    """
    if isinstance(img, torch.Tensor):
        return get_image_size_tensor(img)

    return get_image_size_pil(img)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target