
import torch.nn as nn
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax
from .ohem_loss import OhemCrossEntropy2d
from .softiou_loss import SoftIoULoss
from .MixSoftmax_loss import EncNetLoss, ICNetLoss, MixSoftmaxCrossEntropyOHEMLoss, MixSoftmaxCrossEntropyLoss

class MultiClassCriterion(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        if args.loss_type == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1, **kwargs)
        elif args.loss_type == 'FocalLoss':
            self.criterion = FocalLoss(**kwargs)
        elif args.loss_type == 'LovaszSoftmax':
            self.criterion = LovaszSoftmax(**kwargs)
        elif args.loss_type == 'OhemCrossEntropy2d':
            self.criterion = OhemCrossEntropy2d(**kwargs)
        elif args.loss_type == 'SoftIoULoss':
            self.criterion = SoftIoULoss(n_classes=args.nclasses)
        elif args.loss_type == 'EncNetLoss':
            self.criterion = EncNetLoss(**kwargs)
        elif args.loss_type == 'ICNetLoss':
            self.criterion = ICNetLoss(**kwargs)
        elif args.loss_type == "MixSoftmaxCrossEntropyLoss":
            self.criterion = MixSoftmaxCrossEntropyLoss(**kwargs)
        elif args.loss_type == "MixSoftmaxCrossEntropyOHEMLoss":
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss