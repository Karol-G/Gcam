"""
### author: Aashis Khanal
### sraashis@gmail.com
### date: 9/10/2018
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from ..utils import img_utils as iu

def plot_confusion_matrix(y_pred=None, y_true=None, classes=None, normalize=False, cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    title = 'Confusion matrix'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


class ScoreAccumulator:
    def __init__(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    def add(self, tn=0, fp=0, fn=0, tp=0):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        return self

    def add_tensor(self, y_pred_tensor, y_true_tensor):

        y_true = y_true_tensor.clone().int().view(1, -1).squeeze()
        y_pred = y_pred_tensor.clone().int().view(1, -1).squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += torch.sum(y_cases == 3).item()
        self.fp += torch.sum(y_cases == 1).item()
        self.tn += torch.sum(y_cases == 0).item()
        self.fn += torch.sum(y_cases == 2).item()
        return self

    def add_array(self, arr_2d=None, truth=None):
        x = arr_2d.copy()
        y = truth.copy()
        x[x == 255] = 1
        y[y == 255] = 1
        xy = x + (y * 2)
        self.tp += xy[xy == 3].shape[0]
        self.fp += xy[xy == 1].shape[0]
        self.tn += xy[xy == 0].shape[0]
        self.fn += xy[xy == 2].shape[0]
        return self

    def accumulate(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn
        return self

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4
        return self

    def get_prfa(self, beta=1):
        try:
            p = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            r = 0
        try:
            f = (1 + beta ** 2) * p * r / (((beta ** 2) * p) + r)
        except ZeroDivisionError:
            f = 0
        try:
            a = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        except ZeroDivisionError:
            a = 0
        return [round(p, 5), round(r, 5), round(f, 5), round(a, 5)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_best_thr(img, y, for_best='F1'):
    best_score = {for_best: 0.0}
    best_thr = 0.0

    if np.sum(y) == 0:
        best_score[for_best] = 1
        return best_score, 255.0

    for thr in np.linspace(1, 255, 255):
        i_best = img.copy()
        i_best[i_best > thr] = 255
        i_best[i_best <= thr] = 0
        current_score = iu.get_praf1(i_best, y)
        if current_score[for_best] > best_score[for_best]:
            best_score = current_score
            best_thr = thr
    return best_score, best_thr
