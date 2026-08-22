# Copied from https://github.com/mlcommons/training/blob/637c82f9e699cd6caf108f92efb2c1d446b630e0/single_stage_detector/ssd/presets.py

from test.external.mlperf_retinanet import transforms as T

class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123., 117., 104.)):
        if data_augmentation == 'hflip':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)