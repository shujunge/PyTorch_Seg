"""
Loss.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)
        target = target.transpose(1, 0).contiguous()
        pred = pred.transpose(1,0).contiguous()
        N = target.size(0)
        smooth = 1
        input_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

        return 1 - loss.mean()


class CrossEntropyLoss2d(nn.Module):

    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)


if __name__ =="__main__":

    x= torch.zeros((2,21,224,224))
    y =torch.ones((2,224,224)).long()
    soft = nn.CrossEntropyLoss()
    print(soft(x,y))
    soft_1 = CrossEntropyLoss2d()
    print(soft_1(x,y))