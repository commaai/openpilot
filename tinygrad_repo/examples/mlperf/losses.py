from examples.mlperf.metrics import dice_score

def dice_ce_loss(pred, tgt):
  ce = pred.permute(0, 2, 3, 4, 1).sparse_categorical_crossentropy(tgt.squeeze(1))
  dice = (1.0 - dice_score(pred, tgt, argmax=False, to_one_hot_x=False)).mean()
  return (dice + ce) / 2
