from utils.my_seed import  seed_everything
import numpy as np
import torch
import torch.nn as nn
import copy

def base_IOU(pred , mask):
    y_true = mask.clone().cpu().detach().numpy()
    y_true = [(y_true == v) for v in range(21)]
    y_true = np.stack(y_true, axis=1)
    y_pred = pred.clone().cpu().detach().numpy()
    y_pred = np.transpose(y_pred,(1,0,2,3))
    y_true = np.transpose(y_true,(1,0,2,3))
    N = y_true.shape[0]
    eps=1e-6
    input_flat = y_pred.reshape(N, y_pred.shape[1]*y_pred.shape[2]*y_pred.shape[3])
    target_flat = y_true.reshape(N, y_true.shape[1]*y_true.shape[2]*y_true.shape[3])
    intersection = input_flat * target_flat
    iou = (intersection.sum(1) + eps) / (input_flat.sum(1) + target_flat.sum(1)-intersection.sum(1) + eps)
    # iou[iou>=0.9999] = 0
    return iou.mean()


def IOU(pred , mask):
    mask = mask.transpose(1,0).contiguous()
    pred = pred.transpose(1,0).contiguous()
    N = mask.size(0)
    eps=1e-6
    input_flat = pred.view(N, -1)
    target_flat = mask.view(N, -1)
    intersection = input_flat * target_flat
    iou = (intersection.sum(1) + eps) / (input_flat.sum(1) + target_flat.sum(1)-intersection.sum(1) + eps)
    iou[iou >= 0.9999] = 0
    return iou.mean()

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

if __name__=="__main__":

    np.set_printoptions(precision=3, suppress=True)
    y_preds = torch.ones((2, 21, 343, 565))
    y_labels = torch.ones((2, 343, 565))
    print(base_IOU(y_preds, y_labels))

    y_preds = y_preds.numpy()
    y_labels = y_labels.numpy()
    dd = runningScore(21)
    dd.update(y_labels, y_preds)
    print(dd.get_scores())