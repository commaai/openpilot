from test.external.mlperf_retinanet.model.boxes import box_iou
from test.external.mlperf_retinanet.model.utils import Matcher

import torch

# This applies the filtering in https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/model/retinanet.py#L117
# and https://github.com/mlcommons/training/blob/cdd928d4596c142c15a7d86b2eeadbac718c8da2/single_stage_detector/ssd/model/retinanet.py#L203
# to match with tinygrad's dataloader implementation.
def postprocess_targets(targets, anchors):
    proposal_matcher, matched_idxs = Matcher(0.5, 0.4, allow_low_quality_matches=True), []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image['boxes'].numel() == 0:
            matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64,
                                            device=anchors_per_image.device))
            continue

        match_quality_matrix = box_iou(targets_per_image['boxes'], anchors_per_image)
        matched_idxs.append(proposal_matcher(match_quality_matrix))

    for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
        foreground_idxs_per_image = matched_idxs_per_image >= 0
        targets_per_image["boxes"] = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
        targets_per_image["labels"] = targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]]

    return targets
