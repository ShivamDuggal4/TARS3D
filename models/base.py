import torch
import torch.nn.functional as torch_F
from easydict import EasyDict as edict


class Model(torch.nn.Module):
    """
        Base Model. Parent class for all model types.
    """
    def __init__(self, opt):
        super().__init__()

    def forward(self, opt, var, training=False):
        raise NotImplementedError
        return var

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        raise NotImplementedError
        return loss

    def L1_loss(self, pred, label=0, weight=None, mask=None):
        loss = (pred.contiguous()-label).abs()
        return self.aggregate_loss(loss, weight=weight, mask=mask)

    def MSE_loss(self, pred, label=0, weight=None, mask=None):
        loss = (pred.contiguous()-label)**2
        return self.aggregate_loss(loss, weight=weight, mask=mask)

    def BCE_loss(self, pred, label, weight=None, mask=None):
        label = label.expand_as(pred)
        loss = torch_F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        return self.aggregate_loss(loss, weight=weight, mask=mask)

    def aggregate_loss(self, loss, weight=None, mask=None):
        if mask is not None:
            if mask.sum() == 0:
                return 0.*loss.mean()  # keep loss dependent on prediction
            mask = mask.expand_as(loss)
            if weight is not None:
                weight = weight.expand_as(loss)
            loss = loss[mask]
        if weight is not None:
            if mask is not None:
                weight = weight[mask]
            loss = (loss*weight).sum()/weight.sum()
        else:
            loss = loss.mean()
        return loss
