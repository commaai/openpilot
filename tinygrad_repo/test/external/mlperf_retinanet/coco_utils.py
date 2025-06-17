# Copied from https://github.com/mlcommons/training/blob/637c82f9e699cd6caf108f92efb2c1d446b630e0/single_stage_detector/ssd/coco_utils.py

import os
import torch
import torchvision

from test.external.mlperf_retinanet import transforms as T

class ConvertCocoPolysToMask(object):
    def __init__(self, filter_iscrowd=True):
        self.filter_iscrowd = filter_iscrowd

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        if self.filter_iscrowd:
            anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

def get_openimages(name, root, image_set, transforms):
    PATHS = {
        "train": os.path.join(root, "train"),
        "val":   os.path.join(root, "validation"),
    }

    t = [ConvertCocoPolysToMask(filter_iscrowd=False)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder = os.path.join(PATHS[image_set], "data")
    ann_file = os.path.join(PATHS[image_set], "labels", f"{name}.json")

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    return dataset