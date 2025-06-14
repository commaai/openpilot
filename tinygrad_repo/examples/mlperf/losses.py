from examples.mlperf.metrics import dice_score
from tinygrad import Tensor

def dice_ce_loss(pred, tgt):
  ce = pred.permute(0, 2, 3, 4, 1).sparse_categorical_crossentropy(tgt.squeeze(1))
  dice = (1.0 - dice_score(pred, tgt, argmax=False, to_one_hot_x=False)).mean()
  return (dice + ce) / 2

def sigmoid_focal_loss(pred:Tensor, tgt:Tensor, alpha:float=0.25, gamma:float=2.0, reduction:str="none") -> Tensor:
  assert reduction in ["mean", "sum", "none"], f"unsupported reduction {reduction}"
  p, ce_loss = pred.sigmoid(), pred.binary_crossentropy_logits(tgt, reduction="none")
  p_t = p * tgt + (1 - p) * (1 - tgt)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * tgt + (1 - alpha) * (1 - tgt)
    loss = loss * alpha_t

  if reduction == "mean": loss = loss.mean()
  elif reduction == "sum": loss = loss.sum()
  return loss

def l1_loss(pred:Tensor, tgt:Tensor, reduction:str="none") -> Tensor:
  assert reduction in ["mean", "sum", "none"], f"unsupported reduction {reduction}"
  loss = (pred - tgt).abs()

  if reduction == "mean": loss = loss.mean()
  elif reduction == "sum": loss = loss.sum()
  return loss