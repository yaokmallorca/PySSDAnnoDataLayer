from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
# from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def seg_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    # print(label_trues.dtype, label_preds.dtype)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # if n_class > 2:
    # rec = recall_score(label_trues.flatten(), label_preds.flatten(), average='micro')
    # prec = precision_score(label_trues.flatten(), label_preds.flatten(), average='micro')
    rec_cls = np.diag(hist) / hist.sum(axis=1)
    rec_own = np.nanmean(rec_cls)
    prec_cls = np.diag(hist) / hist.sum(axis=0)
    prec_own = np.nanmean(prec_cls)
    # else:
    #     rec = recall_score(label_trues.flatten(), label_preds.flatten())
    #     prec = precision_score(label_trues.flatten(), label_preds.flatten())

    return mean_iu, rec_own, prec_own